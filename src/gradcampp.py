import numpy as np
import tensorflow as tf
import cv2

# ============================================================
#   UTIL: encontrar última capa convolucional (robusto)
# ============================================================

def _is_conv_layer(layer):
    return isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))

def find_last_conv_layer(model):
    """
    Busca la última capa conv en el modelo, incluyendo submodelos (como EfficientNetV2).
    Devuelve el nombre de la capa.
    """
    # Recorre capas en reversa
    for layer in reversed(model.layers):
        # Si es submodelo (backbone), busca dentro
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if _is_conv_layer(sub):
                    return sub.name
        else:
            if _is_conv_layer(layer):
                return layer.name

    raise ValueError("No encontré ninguna capa Conv2D/DepthwiseConv2D en el modelo.")


# ============================================================
#   GRAD-CAM++ (para clasificación softmax multiclase)
# ============================================================

def make_gradcampp_heatmap(model, img_tensor, class_index=None, layer_name=None, eps=1e-8):
    """
    model: tf.keras.Model (ya cargado)
    img_tensor: [1,H,W,3] float32, YA preprocesado (preprocess_input)
    class_index: int, si None -> usa clase predicha
    layer_name: str, si None -> última conv automática
    returns: heatmap [H,W] float32 en [0,1]
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    # Construimos modelo que entrega activaciones conv + predicción
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape:
        conv_out, preds = grad_model(img_tensor, training=False)  # conv_out: [1,h,w,c]
        if class_index is None:
            class_index = tf.argmax(preds[0])
        score = preds[:, class_index]  # [1]

        # Gradientes de primer orden
        grads = tape.gradient(score, conv_out)  # [1,h,w,c]

        # Gradientes de segundo y tercer orden (para Grad-CAM++)
        grads2 = grads * grads
        grads3 = grads2 * grads

        # α_k = grads2 / (2*grads2 + sum(A * grads3))
        # (sum sobre h,w)
        conv_out_relu = tf.nn.relu(conv_out)  # estabilidad
        sum_conv_grads3 = tf.reduce_sum(conv_out_relu * grads3, axis=[1, 2], keepdims=True)  # [1,1,1,c]

        denom = 2.0 * grads2 + sum_conv_grads3
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom) * eps)

        alphas = grads2 / denom  # [1,h,w,c]

        # pesos = sum(alphas * relu(grads)) sobre h,w
        relu_grads = tf.nn.relu(grads)
        weights = tf.reduce_sum(alphas * relu_grads, axis=[1, 2])  # [1,c]

        # mapa = sum(weights_k * A_k) sobre canales
        cam = tf.reduce_sum(tf.expand_dims(tf.expand_dims(weights, 1), 1) * conv_out, axis=-1)  # [1,h,w]
        cam = tf.nn.relu(cam)[0]  # [h,w]

    del tape

    # Normalización [0,1]
    cam -= tf.reduce_min(cam)
    cam /= (tf.reduce_max(cam) + eps)
    heatmap = cam.numpy().astype(np.float32)
    return heatmap, int(class_index.numpy()), preds.numpy()[0]


# ============================================================
#   OVERLAY BONITO (no “todo azul”)
# ============================================================

def overlay_heatmap_on_image(
    rgb_image_uint8, heatmap, alpha=0.45, colormap=cv2.COLORMAP_TURBO
):
    """
    rgb_image_uint8: [H,W,3] uint8 (0-255) en RGB
    heatmap: [H,W] float32 en [0,1]
    returns: overlay_rgb_uint8
    """
    h, w = rgb_image_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * rgb_image_uint8.astype(np.float32) + alpha * heatmap_color_rgb.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ============================================================
#   PIPELINE SIMPLE (para usar en notebook o script)
# ============================================================

def preprocess_for_efficientnetv2(img_path, img_size):
    """
    Lee imagen con OpenCV y devuelve:
    - rgb_uint8 para visualizar
    - img_tensor preprocesado [1,H,W,3] para el modelo
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    rgb_uint8 = img.copy()

    x = img.astype(np.float32)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return rgb_uint8, tf.convert_to_tensor(x, dtype=tf.float32)
