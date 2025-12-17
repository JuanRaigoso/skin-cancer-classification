# src/inference.py
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def read_image_file(file) -> Image.Image:
    return Image.open(file).convert("RGB")

def preprocess_pil(img: Image.Image, img_width: int, img_height: int) -> np.ndarray:
    img = img.resize((img_width, img_height))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # [1,H,W,3]

def apply_tta_np(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    img_rgb_uint8: [H,W,3] uint8 (0-255)
    Retorna imagen aumentada (uint8) para TTA simple.
    """
    x = img_rgb_uint8

    # flips
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=1)  # horizontal
    if np.random.rand() < 0.3:
        x = np.flip(x, axis=0)  # vertical

    # rotación 90
    if np.random.rand() < 0.2:
        k = np.random.choice([1, 2, 3])
        x = np.rot90(x, k)

    # jitter brillo
    if np.random.rand() < 0.3:
        x = np.clip(x.astype(np.float32) * (0.9 + 0.2*np.random.rand()), 0, 255).astype(np.uint8)

    return x.copy()

def temperature_scale_probs(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Nota: esto usa log(p) como "logits aproximados".
    Sirve como ajuste suave y estable para la app.
    """
    eps = 1e-8
    logits = np.log(np.clip(probs, eps, 1.0))
    logits = logits / float(T)
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def predict_proba(
    model: tf.keras.Model,
    img_pil: Image.Image,
    img_width: int,
    img_height: int,
    use_tta: bool = True,
    tta_samples: int = 8,
    temperature: float | None = None,
) -> np.ndarray:
    """
    Retorna probs [C]
    """
    # base image (uint8) para TTA
    base = np.array(img_pil.resize((img_width, img_height)), dtype=np.uint8)

    if use_tta:
        probs = []
        for _ in range(int(tta_samples)):
            aug = apply_tta_np(base)  # uint8
            x = preprocess_input(aug.astype(np.float32))
            x = np.expand_dims(x, axis=0)
            p = model.predict(x, verbose=0)[0]
            probs.append(p)
        p = np.mean(probs, axis=0)
    else:
        x = preprocess_pil(img_pil, img_width, img_height)
        p = model.predict(x, verbose=0)[0]

    if temperature is not None:
        p = temperature_scale_probs(p, temperature)

    return p
