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
#   MIXUP (VERSIÓN ESTABLE Y SIMPLE)
# ============================================

def mixup(images, labels, alpha=0.2):
    """
    Aplica MixUp a un batch completo.
    """
    batch_size = tf.shape(images)[0]

    # Lambda ~ Uniform(0,1) (simple y funciona bien)
    lam = tf.random.uniform([], 0, 1)

    # Reordenar el batch
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


# ============================================
#   FUNCIÓN DE AUGMENTACIÓN GLOBAL
# ============================================

def apply_augmentations(x, y, aug):
    """
    Aplica augmentaciones de Keras + (a veces) MixUp.
    """
    # Augmentaciones geométricas / fotométricas
    x = aug(x, training=True)

    # Decide aleatoriamente si aplicar MixUp o no
    r = tf.random.uniform(())
    if r < 0.5:
        x, y = mixup(x, y)

    return x, y


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

    # Aplicar augmentaciones + MixUp SOLO al train
    train_ds = train_ds.map(
        lambda x, y: apply_augmentations(x, y, aug),
        num_parallel_calls=AUTOTUNE
    )

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
