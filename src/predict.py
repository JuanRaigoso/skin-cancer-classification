import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES

# ============================================================
#   Cargar modelo
# ============================================================

MODEL_PATH = os.path.join("models", "model_final_B4_v1_advanced.h5")

print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Modelo cargado correctamente.")

# ============================================================
#   Preprocesamiento de imagen
# ============================================================

from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32")

    # MISMO PREPROCESAMIENTO QUE EL ENTRENAMIENTO
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img

# ============================================================
#   Predicción
# ============================================================

def predict_image(img_path):
    """
    Recibe ruta de imagen, la procesa, predice y devuelve probabilidades.
    """
    img = preprocess_image(img_path)
    preds = model.predict(img)[0]  # primer batch

    # Ordenar probabilidades de mayor a menor
    sorted_idx = np.argsort(preds)[::-1]
    sorted_probs = preds[sorted_idx]
    sorted_classes = [CLASS_NAMES[i] for i in sorted_idx]

    # Clase ganadora
    top_class = sorted_classes[0]
    top_prob = sorted_probs[0]

    print("\n==============================")
    print("📌 RESULTADOS DE PREDICCIÓN")
    print("==============================")

    print(f"🔮 Clase más probable: **{top_class.upper()}**")
    print(f"📊 Probabilidad: {top_prob:.4f}")
    print("\n=== Probabilidades por clase ===")

    for cls, prob in zip(sorted_classes, sorted_probs):
        print(f"{cls:>5}  →  {prob:.4f}")

    return top_class, preds


# ============================================================
#   Ejecución directa desde terminal
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clasificación de cáncer de piel")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen a predecir")

    args = parser.parse_args()
    predict_image(args.image)
