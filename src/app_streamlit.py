# ============================================================
# src/app_streamlit.py — versión completa con TODAS las mejoras
# ============================================================

import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================
try:
    from config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
except:
    CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    IMG_HEIGHT = 380
    IMG_WIDTH = 380

NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    "akiec": "Queratosis actínica / enfermedad de Bowen (precancerosa).",
    "bcc":   "Carcinoma basocelular.",
    "bkl":   "Lesión benigna tipo queratosis.",
    "df":    "Dermatofibroma (benigno).",
    "mel":   "Melanoma (alto riesgo).",
    "nv":    "Nevus melanocítico (lunar).",
    "vasc":  "Lesión vascular (angioma, hemangioma)."
}

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_final_B4_v1_advanced.h5"
REPORT_PATH = BASE_DIR / "classification_report_B4_v1_advanced.json"


# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ============================================================
# PREPROCESAMIENTO
# ============================================================
def read_image_file(file):
    return Image.open(file).convert("RGB")


def preprocess_image(img: Image.Image):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ============================================================
# HEATMAP — Grad-CAM ROBUSTO + SmoothGrad
# ============================================================
def compute_smooth_grad(img_array, model, class_idx, samples=15, noise_level=0.2):

    aggregated_grads = 0

    for i in range(samples):
        noise = tf.random.normal(shape=img_array.shape, mean=0.0,
                                 stddev=noise_level * tf.math.reduce_std(img_array))
        noised_img = tf.cast(img_array + noise, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(noised_img)
            preds = model(noised_img, training=False)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, noised_img)
        grads = tf.reduce_mean(grads, axis=-1)[0]  # reduce channels
        aggregated_grads += grads

    smooth_grad = aggregated_grads / samples

    heatmap = tf.maximum(smooth_grad, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def apply_medical_colormap(original_img, heatmap, alpha=0.45):

    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # medical jet: JET + small contrast boost
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.convertScaleAbs(heatmap_color, alpha=1.1, beta=10)

    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)

    return Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))


def make_gradcam(img_array, model):
    preds = model(img_array, training=False)
    class_idx = int(tf.argmax(preds[0]))

    heatmap = compute_smooth_grad(img_array, model, class_idx)
    probs = preds.numpy()[0]

    return heatmap, probs


# ============================================================
# ESTILO Y MODO OSCURO/CLARO
# ============================================================
def apply_theme(dark_mode=True):

    if dark_mode:
        bg = "#0c1020"
        text = "#f5f7ff"
        panel = "#111425"
    else:
        bg = "#f5f5f5"
        text = "#111"
        panel = "#e8e8e8"

    st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {bg}; color: {text}; }}

    h1, h2, h3, h4 {{ color: {text} !important; }}

    .confidence-bar {{
        height: 18px;
        background-color: #444;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 6px;
    }}
    .confidence-inner {{
        height: 100%;
        color: white;
        text-align: right;
        padding-right: 5px;
        line-height: 18px;
        font-size: 12px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


def get_color(prob):
    if prob >= 0.60: return "🟢"
    if prob >= 0.30: return "🟡"
    return "🔴"


# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================
def main():

    st.set_page_config(page_title="Clasificación de cáncer de piel — EfficientNetB4",
                       page_icon="🧬",
                       layout="wide")

    # Theming switch
    mode = st.sidebar.radio("🌓 Tema", ["Modo oscuro", "Modo claro"])
    apply_theme(dark_mode = (mode == "Modo oscuro"))
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 🔗 Enlaces del proyecto
        - 📁 [Repositorio en GitHub](https://github.com/JuanRaigoso/skin-cancer-classification/tree/main)
        """
    )
    
    st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/JuanRaigoso/skin-cancer-classification/tree/main" target="_blank">
            <button style="
                background-color:#1f77b4;
                color:white;
                border:none;
                padding:10px 20px;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;">
                🌐 Ver proyecto en GitHub
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


    st.title("🧬 Clasificación de Cáncer de Piel (EfficientNetB4)")
    st.info("Esta herramienta es solamente para fines educativos. No es diagnóstico médico.")

    model = load_model()

    col_left, col_center, col_right = st.columns([1.2, 1.1, 1.0])

    with col_left:
        st.subheader("📁 Sube una imagen dermatoscópica")
        uploaded = st.file_uploader(
        "Selecciona una imagen (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader_main"
    )

    if uploaded:
        img_pil = read_image_file(uploaded)
        img_array = preprocess_image(img_pil)

        heatmap, probs = make_gradcam(img_array, model)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = probs[pred_idx]

        grad_img = apply_medical_colormap(img_pil, heatmap)
        sorted_idx = np.argsort(probs)[::-1]

        # --------------------------------------------------
        # Imagen original y Heatmap
        # --------------------------------------------------
        with col_center:
            st.subheader("🩻 Imagen original")
            st.image(img_pil, use_container_width=True)

            st.subheader("🔥 Grad-CAM (SmoothGrad + Medical Jet)")
            st.image(grad_img, use_container_width=True)

        # --------------------------------------------------
        # Resultados
        # --------------------------------------------------
        with col_right:
            st.subheader("📌 Predicción principal")
            st.markdown(f"### {pred_class.upper()} — {pred_prob*100:.1f}%")
            st.write(CLASS_DESCRIPTIONS[pred_class])

            # -------------------------------
            # Nivel de confianza + Confidence Meter
            # -------------------------------
            if pred_prob >= 0.60:
                level = "🟢 Alta"
                color = "#00cc44"
            elif pred_prob >= 0.40:
                level = "🟡 Media"
                color = "#ffcc00"
            else:
                level = "🔴 Baja"
                color = "#cc0000"

            st.markdown(f"### 🔍 Nivel de confianza: {level}")

            width = int(pred_prob * 100)

            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-inner" style="width:{width}%; background:{color};">
                    {width}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------------
            # Probabilidades por clase
            # -------------------------------
            st.markdown("---")
            st.subheader("📊 Probabilidades")
            for i in sorted_idx:
                cls = CLASS_NAMES[i]
                pr = probs[i]
                st.markdown(f"- {get_color(pr)} **{cls.upper()}**: {pr*100:.1f}%")

    # =======================================================
    # PANEL DE MÉTRICAS DEL MODELO
    # =======================================================
    st.markdown("---")
    st.header("📈 Métricas del modelo (Validación HAM10000)")

    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)

        st.table({
            "Clase": list(CLASS_NAMES),
            "Precision": [report[c]["precision"] for c in CLASS_NAMES],
            "Recall": [report[c]["recall"] for c in CLASS_NAMES],
            "F1-score": [report[c]["f1-score"] for c in CLASS_NAMES],
        })

        # Botón para ver el JSON completo
        if st.button("📄 Mostrar classification_report.json"):
            st.json(report)

    else:
        st.warning("No se encontró el archivo classification_report_B4_v1_advanced.json.")

    # =======================================================
    # EXPLICACIÓN DE INTERPRETACIÓN
    # =======================================================
    st.markdown("---")
    st.header("ℹ️ ¿Cómo interpretar los resultados?")

    st.markdown("""
    - **Confianza alta (🟢)**: el modelo está razonablemente seguro.
    - **Confianza media (🟡)**: resultado incierto; examinar otras clases y Grad-CAM.
    - **Confianza baja (🔴)**: el modelo no está seguro; no debe usarse para decisiones.
    - El heatmap muestra **regiones que influyen más en la predicción**.
    - Recuerda: este sistema es **solo educativo** y no sustituye a una evaluación dermatológica real.
    """)


if __name__ == "__main__":
    main()
