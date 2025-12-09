import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
from config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, DATA_DIR, NUM_CLASSES
AUTOTUNE = tf.data.AUTOTUNE

# ============================================
#   AUGMENTACIONES AVANZADAS
# ============================================

def get_augmentations():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.30),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
    ])


# ============================================
#   MIXUP + CUTMIX IMPLEMENTACIÓN
# ============================================

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1 = tf.random.gamma(shape=[size], alpha=concentration_0)
    gamma_2 = tf.random.gamma(shape=[size], alpha=concentration_1)
    return gamma_1 / (gamma_1 + gamma_2)

def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))

    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * x_l + tf.gather(images, index) * (1 - x_l)
    mixed_labels = labels * l[:, None] + tf.gather(labels, index) * (1 - l[:, None])
    return mixed_images, mixed_labels

def cutmix(images, labels, alpha=0.3):
    batch_size = tf.shape(images)[0]
    W = IMG_WIDTH
    H = IMG_HEIGHT

    l = sample_beta_distribution(batch_size, alpha, alpha)

    cut_rat = tf.math.sqrt(1. - l)
    cut_w = tf.cast(W * cut_rat, tf.int32)
    cut_h = tf.cast(H * cut_rat, tf.int32)

    cx = tf.random.uniform((batch_size,), 0, W, tf.int32)
    cy = tf.random.uniform((batch_size,), 0, H, tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, H)

    index = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, index)
    shuffled_labels = tf.gather(labels, index)

    new_images = []
    for i in range(batch_size):
        image = images[i]
        cut_image = shuffled_images[i]
        image = tf.tensor_scatter_nd_update(
            image,
            tf.reshape(tf.range(y1[i], y2[i]), (-1, 1)),
            tf.reshape(cut_image[y1[i]:y2[i], x1[i]:x2[i], :], (-1, x2[i]-x1[i], 3)),
        )
        new_images.append(image)
    new_images = tf.stack(new_images)

    lam = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
    new_labels = labels * lam + shuffled_labels * (1 - lam)

    return new_images, new_labels


# ============================================
#   CARGA DE DATASETS
# ============================================

def load_datasets():

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    aug = get_augmentations()

    def augment(x, y):
        x = aug(x)
        # Random aplicar MixUp o CutMix
        r = tf.random.uniform(())
        if r < 0.33:
            return mixup(x, y)
        elif r < 0.66:
            return cutmix(x, y)
        else:
            return x, y

    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# ============================================
#   CLASS WEIGHTS
# ============================================

def get_class_weights(train_ds, num_classes):
    labels = []
    for _, y in train_ds:
        labels.extend(np.argmax(y.numpy(), axis=1))

    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )
    return {i: w for i, w in enumerate(class_weights)}
