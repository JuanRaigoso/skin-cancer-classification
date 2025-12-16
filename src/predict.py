import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, USE_TTA, TTA_SAMPLES

MODEL_PATH = os.path.join("models", "model_final_V2M_v1_robust.keras")

print(f"[INFO] Cargando modelo: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] OK")

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def preprocess_image_raw(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32")
    return img

def apply_tta(img):
    # img: [H,W,C] float32
    x = img

    # flips aleatorios
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=1)  # horizontal
    if np.random.rand() < 0.3:
        x = np.flip(x, axis=0)  # vertical

    # rotación 90 (a veces)
    if np.random.rand() < 0.2:
        k = np.random.choice([1,2,3])
        x = np.rot90(x, k)

    # pequeño jitter de brillo (suave)
    if np.random.rand() < 0.3:
        x = np.clip(x * (0.9 + 0.2*np.random.rand()), 0, 255)

    return x.copy()

def predict_proba(img_path, temperature=None):
    img = preprocess_image_raw(img_path)

    if USE_TTA:
        probs = []
        for _ in range(TTA_SAMPLES):
            x = apply_tta(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            p = model.predict(x, verbose=0)[0]
            probs.append(p)
        p = np.mean(probs, axis=0)
    else:
        x = preprocess_input(img)
        x = np.expand_dims(x, 0)
        p = model.predict(x, verbose=0)[0]

    # calibración por temperatura (si la pasas)
    if temperature is not None:
        p = temperature_scale_probs(p, temperature)

    return p

def temperature_scale_probs(probs, T):
    # aplica temperature scaling en logits aproximados
    # Convertimos probs->logits con log, escalamos, volvemos a softmax
    eps = 1e-8
    logits = np.log(np.clip(probs, eps, 1.0))
    logits = logits / float(T)
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def predict_image(img_path, temperature=None):
    p = predict_proba(img_path, temperature=temperature)
    idx = np.argsort(p)[::-1]
    print("\n==============================")
    print("📌 RESULTADOS DE PREDICCIÓN")
    print("==============================")
    for i in idx[:7]:
        print(f"{CLASS_NAMES[i]:>5} → {p[i]:.4f}")
    print(f"\nTop-1: {CLASS_NAMES[idx[0]].upper()}  ({p[idx[0]]:.4f})")
    return CLASS_NAMES[idx[0]], p

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    predict_image(args.image, temperature=args.temperature)
## Final