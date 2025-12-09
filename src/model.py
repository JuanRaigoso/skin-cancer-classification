import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW

from config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    NUM_CLASSES, MODEL_NAME,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
    UNFROZEN_LAYERS, LR_FINE_TUNE
)

def load_backbone():
    if MODEL_NAME == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        )
    elif MODEL_NAME == "EfficientNetB3":
        return tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        )
    elif MODEL_NAME == "EfficientNetB7":
        return tf.keras.applications.EfficientNetB7(
            include_top=False, weights="imagenet",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        )
    else:
        raise ValueError("Modelo no soportado")


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
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_finetune_model():
    backbone = load_backbone()

    # Descongelar las últimas UNFROZEN_LAYERS
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
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
