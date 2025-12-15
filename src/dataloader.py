import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, DATA_DIR, NUM_CLASSES, SEED,
    MIXUP_PROB, CUTMIX_PROB, RANDOM_ERASE_PROB, BLUR_PROB
)

AUTOTUNE = tf.data.AUTOTUNE

# ============================================
#   AUGMENTACIONES (Keras)
# ============================================

def get_augmentations():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.20, seed=SEED),
        tf.keras.layers.RandomZoom(0.15, seed=SEED),
        tf.keras.layers.RandomContrast(0.15, seed=SEED),
        tf.keras.layers.RandomBrightness(0.20, seed=SEED),
        tf.keras.layers.RandomTranslation(0.08, 0.08, seed=SEED),
    ])

# ============================================
#   UTILS
# ============================================

def _to_float(x, y):
    x = tf.cast(x, tf.float32)
    return x, y

def _random_blur(x):
    # blur suave usando avg_pool (rápido y sin tf-addons)
    # x: [B,H,W,C]
    k = 3
    x = tf.nn.avg_pool2d(x, ksize=k, strides=1, padding="SAME")
    return x

def _maybe_blur(x):
    r = tf.random.uniform(())
    return tf.cond(r < BLUR_PROB, lambda: _random_blur(x), lambda: x)

def random_erasing_batch(images, erase_prob=RANDOM_ERASE_PROB):
    # Random erasing simple por batch (aplica a algunas imágenes)
    # images: [B,H,W,C] float32
    b = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    c = tf.shape(images)[3]

    def erase_one(img):
        r = tf.random.uniform(())
        def do_erase():
            # área
            erase_h = tf.cast(tf.round(tf.cast(h, tf.float32) * tf.random.uniform((), 0.08, 0.20)), tf.int32)
            erase_w = tf.cast(tf.round(tf.cast(w, tf.float32) * tf.random.uniform((), 0.08, 0.20)), tf.int32)
            y0 = tf.random.uniform((), 0, tf.maximum(1, h - erase_h), dtype=tf.int32)
            x0 = tf.random.uniform((), 0, tf.maximum(1, w - erase_w), dtype=tf.int32)

            mask = tf.ones([erase_h, erase_w, c], dtype=img.dtype)
            # valor “gris” (puedes usar ruido también)
            fill = tf.zeros([erase_h, erase_w, c], dtype=img.dtype)

            paddings = [[y0, h - y0 - erase_h], [x0, w - x0 - erase_w], [0, 0]]
            patch = tf.pad(fill, paddings, constant_values=0.0)
            patch_mask = tf.pad(mask, paddings, constant_values=0.0)

            return img * (1.0 - patch_mask) + patch

        return tf.cond(r < erase_prob, do_erase, lambda: img)

    return tf.map_fn(erase_one, images, fn_output_signature=images.dtype)

# ============================================
#   MIXUP
# ============================================

def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0.0, 1.0)  # simple y estable
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

# ============================================
#   CUTMIX
# ============================================

def cutmix(images, labels):
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    # ratio del recorte
    r = tf.random.uniform((), 0.3, 0.6)
    cut_h = tf.cast(tf.round(tf.cast(h, tf.float32) * r), tf.int32)
    cut_w = tf.cast(tf.round(tf.cast(w, tf.float32) * r), tf.int32)

    cy = tf.random.uniform((), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((), 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

    # máscara: 1 donde se pega el parche
    patch = tf.ones([y2 - y1, x2 - x1, tf.shape(images)[3]], dtype=images.dtype)
    paddings = [[y1, h - y2], [x1, w - x2], [0, 0]]
    patch_mask = tf.pad(patch, paddings, constant_values=0.0)  # [H,W,C]
    patch_mask = tf.expand_dims(patch_mask, 0)                 # [1,H,W,C]
    patch_mask = tf.tile(patch_mask, [batch_size, 1, 1, 1])    # [B,H,W,C]

    mixed_images = images * (1.0 - patch_mask) + shuffled_images * patch_mask

    # lambda = proporción de pixeles NO reemplazados
    replaced_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    total_area = tf.cast(h * w, tf.float32)
    lam = 1.0 - (replaced_area / (total_area + 1e-7))

    mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels
    return mixed_images, mixed_labels

# ============================================
#   PIPELINE AUG
# ============================================

def apply_augmentations(x, y, aug):
    x = tf.cast(x, tf.float32)

    # Keras aug
    x = aug(x, training=True)

    # Blur a veces
    x = _maybe_blur(x)

    # Decide MixUp / CutMix / nada (con tf.cond)
    r = tf.random.uniform(())
    def do_mixup():
        return mixup(x, y)
    def do_cutmix():
        return cutmix(x, y)
    def do_none():
        return x, y

    x, y = tf.cond(
        r < MIXUP_PROB,
        do_mixup,
        lambda: tf.cond(r < (MIXUP_PROB + CUTMIX_PROB), do_cutmix, do_none)
    )

    # Random erasing al final (sobre el batch)
    x = random_erasing_batch(x)

    return x, y

# ============================================
#   LOAD DATASETS (raw + aug)
# ============================================

def _load_dir(dir_path, shuffle):
    return tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )

def load_datasets():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    # RAW (sin aug) — útil para class_weights
    train_raw = _load_dir(train_dir, shuffle=True).map(_to_float, num_parallel_calls=AUTOTUNE)
    val_ds    = _load_dir(val_dir, shuffle=False).map(_to_float, num_parallel_calls=AUTOTUNE)

    # AUG (para entrenar)
    aug = get_augmentations()
    train_aug = train_raw.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    train_aug = train_aug.map(lambda x, y: apply_augmentations(x, y, aug),
                              num_parallel_calls=AUTOTUNE)

    train_aug = train_aug.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_aug, val_ds, train_raw

# ============================================
#   CLASS WEIGHTS (desde RAW)
# ============================================

def get_class_weights_from_raw(train_raw, num_classes):
    labels = []
    for _, y in train_raw:
        labels.extend(np.argmax(y.numpy(), axis=1))
    labels = np.array(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )
    return {i: float(w) for i, w in enumerate(class_weights)}
