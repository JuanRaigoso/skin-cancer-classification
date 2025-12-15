import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW

from config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES,
    LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE,
    UNFROZEN_LAYERS, LR_FINE_TUNE, LABEL_SMOOTHING, GRAD_CLIPNORM
)

# =========================
# FOCAL LOSS (con smoothing via y_true)
# =========================
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        return alpha * tf.pow((1.0 - p_t), gamma) * ce
    return loss

def smooth_labels(y, smoothing=LABEL_SMOOTHING):
    if smoothing <= 0:
        return y
    num_classes = tf.cast(tf.shape(y)[-1], tf.float32)
    return (1.0 - smoothing) * y + (smoothing / num_classes)

# =========================
# BACKBONE
# =========================
def load_backbone():
    backbone = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )
    return backbone

def build_model():
    backbone = load_backbone()
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # EfficientNetV2 preprocess
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    opt = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, clipnorm=GRAD_CLIPNORM)
    model.compile(
        optimizer=opt,
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(multi_label=True, num_labels=NUM_CLASSES, name="auc_ovr")
        ]
    )
    return model

def enable_finetuning(model):
    # Encontrar backbone (por tipo)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnetv2" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        # fallback: el 2do layer suele ser el backbone, pero mejor explícito
        backbone = model.layers[2] if isinstance(model.layers[2], tf.keras.Model) else None
    if backbone is None:
        raise RuntimeError("No pude localizar el backbone EfficientNetV2 dentro del modelo.")

    # Descongelar últimas N capas
    backbone.trainable = True
    for l in backbone.layers[:-UNFROZEN_LAYERS]:
        l.trainable = False
    for l in backbone.layers[-UNFROZEN_LAYERS:]:
        l.trainable = True

    # Recompilar con LR chico
    opt = AdamW(learning_rate=LR_FINE_TUNE, weight_decay=WEIGHT_DECAY, clipnorm=GRAD_CLIPNORM)
    model.compile(
        optimizer=opt,
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(multi_label=True, num_labels=NUM_CLASSES, name="auc_ovr")
        ]
    )
    return model
