# src/gradcampp.py
import numpy as np
import tensorflow as tf

def make_gradcampp_heatmap(model, img_tensor, class_index=None, eps=1e-8):
    """
    Grad-CAM++ para un modelo con estructura:
    inputs -> backbone(submodel) -> GAP -> Dropout -> Dense(softmax)

    Retorna:
      heatmap [h,w] float32 en [0,1],
      class_idx int,
      probs np.ndarray [C]
    """

    # 1) Identificar componentes del modelo por nombre (tu modelo los tiene así)
    backbone = None
    gap = None
    dropout = None
    dense = None

    for layer in model.layers:
        name = layer.name.lower()
        if isinstance(layer, tf.keras.Model) and "efficientnet" in name:
            backbone = layer
        elif "global_average_pooling2d" in name:
            gap = layer
        elif "dropout" in name:
            dropout = layer
        elif "dense" == name or name.startswith("dense"):
            dense = layer

    if backbone is None or gap is None or dropout is None or dense is None:
        raise ValueError(
            "No pude detectar backbone/GAP/Dropout/Dense en el modelo. "
            f"Capas encontradas: {[l.name for l in model.layers]}"
        )

    # 2) Forward conectado (preds calculado desde conv_out)
    with tf.GradientTape(persistent=True) as tape:
        conv_out = backbone(img_tensor, training=False)  # [1,h,w,c]
        tape.watch(conv_out)

        x = gap(conv_out)
        x = dropout(x, training=False)
        preds = dense(x)  # [1,C] (softmax)

        if class_index is None:
            class_index = tf.argmax(preds[0])
        score = preds[:, class_index]

        grads = tape.gradient(score, conv_out)  # [1,h,w,c]

        if grads is None:
            raise RuntimeError("Gradientes None: no se pudo conectar score con conv_out para Grad-CAM++.")

        grads2 = grads * grads
        grads3 = grads2 * grads

        conv_out_relu = tf.nn.relu(conv_out)
        sum_conv_grads3 = tf.reduce_sum(conv_out_relu * grads3, axis=[1, 2], keepdims=True)

        denom = 2.0 * grads2 + sum_conv_grads3
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom) * eps)

        alphas = grads2 / denom
        relu_grads = tf.nn.relu(grads)
        weights = tf.reduce_sum(alphas * relu_grads, axis=[1, 2])  # [1,c]

        cam = tf.reduce_sum(tf.expand_dims(tf.expand_dims(weights, 1), 1) * conv_out, axis=-1)  # [1,h,w]
        cam = tf.nn.relu(cam)[0]

    del tape

    cam -= tf.reduce_min(cam)
    cam /= (tf.reduce_max(cam) + eps)

    heatmap = cam.numpy().astype(np.float32)
    return heatmap, int(class_index.numpy()), preds.numpy()[0]
