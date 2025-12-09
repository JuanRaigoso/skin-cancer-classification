import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
from config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, DATA_DIR

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
#                 DATA AUGMENTATION MEJORADO
# =========================================================
def get_augmentations():
    """
    Data augmentation optimizado para EfficientNetB3_v2.
    Mejora la generalización, robustez y reduce overfitting.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),

        # Nuevos augmentations recomendados
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])


# =========================================================
#                 CARGA DE DATASETS
# =========================================================
def load_datasets():
    """
    Carga los datasets desde DATA_DIR, aplica augmentations al train
    y prepara los datasets para entrenamiento en GPU.
    """

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    # ----------- Training dataset -----------
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # ----------- Validation dataset -----------
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # ----------- Apply augmentations ONLY TO TRAIN -----------
    aug = get_augmentations()
    train_ds = train_ds.map(
        lambda x, y: (aug(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )

    # ----------- Performance optimizations -----------
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# =========================================================
#                 CLASS WEIGHTS (DESBALANCE)
# =========================================================
def get_class_weights(train_ds, num_classes):
    """
    Calcula los class weights para manejar el desbalance del dataset.
    """

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
