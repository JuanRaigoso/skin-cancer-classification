import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES,
    MODEL_NAME, LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
    UNFROZEN_LAYERS, LR_FINE_TUNE
)

# =========================================================
# FOCAL LOSS
# =========================================================

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = alpha * tf.pow((1 - p_t), gamma) * ce
        return loss
    return loss


# =========================================================
# BACKBONE LOADER
# =========================================================

def load_backbone():
    if MODEL_NAME == "EfficientNetB4":
        return tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        )
    else:
        raise ValueError("Modelo no soportado")


# =========================================================
# TRANSFER LEARNING (WARM-UP)
# =========================================================

def create_model(trainable=False):
    backbone = load_backbone()
    backbone.trainable = trainable

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=focal_loss(),
        metrics=["accuracy"]
    )
    return model


# =========================================================
# FINE TUNING PROFUNDO
# =========================================================

def create_finetune_model():
    backbone = load_backbone()

    for layer in backbone.layers[-UNFROZEN_LAYERS:]:
        layer.trainable = True

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=AdamW(learning_rate=LR_FINE_TUNE, weight_decay=WEIGHT_DECAY),
        loss=focal_loss(),
        metrics=["accuracy"]
    )
    return model
