# src/gradcam.py
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from gradcampp import make_gradcampp_heatmap

def preprocess_for_model(img_pil: Image.Image, img_width: int, img_height: int) -> tf.Tensor:
    img = img_pil.resize((img_width, img_height))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # [1,H,W,3]
    return tf.convert_to_tensor(arr, dtype=tf.float32)

def apply_medical_colormap(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    original_img: PIL RGB
    heatmap: [h,w] float32 en [0,1]
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.convertScaleAbs(heatmap_color, alpha=1.1, beta=10)

    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)

    return Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

def make_gradcampp_overlay(
    model: tf.keras.Model,
    img_pil: Image.Image,
    img_width: int,
    img_height: int,
):
    """
    Retorna:
      overlay_img (PIL),
      probs (np.ndarray [C]),
      pred_class_index (int)
    """
    x = preprocess_for_model(img_pil, img_width, img_height)  # tensor [1,H,W,3]
    heatmap, class_idx, probs = make_gradcampp_heatmap(model, x)
    overlay = apply_medical_colormap(img_pil, heatmap)
    return overlay, probs, class_idx
