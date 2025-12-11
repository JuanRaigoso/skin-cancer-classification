# ============================================================
# src/app_streamlit.py — versión con ejemplos HAM10000 + externos
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
# CONFIGURACIÓN DE IMÁGENES DE EJEMPLO
# ============================================================
EXAMPLE_IMAGES_HAM = [
    {"id": "ham_akiec_1", "label": "akiec", "title": "AKIEC – Ejemplo 1 (HAM10000)", "path": "assets/examples/ham/akiec_1.jpg"},
    {"id": "ham_akiec_2", "label": "akiec", "title": "AKIEC – Ejemplo 2 (HAM10000)", "path": "assets/examples/ham/akiec_2.jpg"},
    {"id": "ham_bcc_1",   "label": "bcc",   "title": "BCC – Ejemplo 1 (HAM10000)",   "path": "assets/examples/ham/bcc_1.jpg"},
    {"id": "ham_bcc_2",   "label": "bcc",   "title": "BCC – Ejemplo 2 (HAM10000)",   "path": "assets/examples/ham/bcc_2.jpg"},
    {"id": "ham_bkl_1",   "label": "bkl",   "title": "BKL – Ejemplo 1 (HAM10000)",   "path": "assets/examples/ham/bkl_1.jpg"},
    {"id": "ham_bkl_2",   "label": "bkl",   "title": "BKL – Ejemplo 2 (HAM10000)",   "path": "assets/examples/ham/bkl_2.jpg"},
    {"id": "ham_df_1",    "label": "df",    "title": "DF – Ejemplo 1 (HAM10000)",    "path": "assets/examples/ham/df_1.jpg"},
    {"id": "ham_df_2",    "label": "df",    "title": "DF – Ejemplo 2 (HAM10000)",    "path": "assets/examples/ham/df_2.jpg"},
    {"id": "ham_mel_1",   "label": "mel",   "title": "MEL – Ejemplo 1 (HAM10000)",   "path": "assets/examples/ham/mel_1.jpg"},
    {"id": "ham_mel_2",   "label": "mel",   "title": "MEL – Ejemplo 2 (HAM10000)",   "path": "assets/examples/ham/mel_2.jpg"},
    {"id": "ham_nv_1",    "label": "nv",    "title": "NV – Ejemplo 1 (HAM10000)",    "path": "assets/examples/ham/nv_1.jpg"},
    {"id": "ham_nv_2",    "label": "nv",    "title": "NV – Ejemplo 2 (HAM10000)",    "path": "assets/examples/ham/nv_2.jpg"},
    {"id": "ham_vasc_1",  "label": "vasc",  "title": "VASC – Ejemplo 1 (HAM10000)",  "path": "assets/examples/ham/vasc_1.jpg"},
    {"id": "ham_vasc_2",  "label": "vasc",  "title": "VASC – Ejemplo 2 (HAM10000)",  "path": "assets/examples/ham/vasc_2.jpg"},
]

EXAMPLE_IMAGES_EXTERNAL = [
    {"id": "ext_df_1", "label": "df",   "title": "DF – Caso clínico externo",   "path": "assets/examples/external/df_web_1.jpg"},
    {"id": "ext_nv_1", "label": "nv",   "title": "NV – Caso clínico externo",   "path": "assets/examples/external/nv_web_1.jpg"},
    {"id": "ext_vasc_1", "label": "vasc", "title": "VASC – Caso clínico externo", "path": "assets/examples/external/vasc_web_1.jpg"},
    {"id": "ext_vasc_2", "label": "vasc", "title": "VASC – Caso clínico externo", "path": "assets/examples/external/vasc_web_2.jpg"},
]


# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ============================================================
# PREPROCESAMIENTO
# ============================================================
def read_image_file(file) -> Image.Image:
    return Image.open(file).convert("RGB")


def preprocess_image(img: Image.Image):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def load_example_image(relative_path: str):
    abs_path = BASE_DIR / relative_path
    if not abs_path.exists():
        return None
    return Image.open(abs_path).convert("RGB")


# ============================================================
# HEATMAP — SmoothGrad + Grad-CAM
# ============================================================
def compute_smooth_grad(img_array, model, class_idx, samples=15, noise_level=0.2):
    aggregated_grads = 0
    for i in range(samples):
        noise = tf.random.normal(
            shape=img_array.shape,
            mean=0.0,
            stddev=noise_level * tf.math.reduce_std(img_array)
        )
        noised_img = tf.cast(img_array + noise, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(noised_img)
            preds = model(noised_img, training=False)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, noised_img)
        grads = tf.reduce_mean(grads, axis=-1)[0]
        aggregated_grads += grads

    smooth_grad = aggregated_grads / samples
    heatmap = tf.maximum(smooth_grad, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def apply_medical_colormap(original_img, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

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
    else:
        bg = "#f5f5f5"
        text = "#111"

    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )


def get_color(prob):
    if prob >= 0.60:
        return "🟢"
    if prob >= 0.30:
        return "🟡"
    return "🔴"


# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================
def main():

    st.set_page_config(
        page_title="Clasificación de cáncer de piel — EfficientNetB4",
        page_icon="🧬",
        layout="wide",
    )

    mode = st.sidebar.radio("🌓 Tema", ["Modo oscuro", "Modo claro"])
    apply_theme(dark_mode=(mode == "Modo oscuro"))

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
        unsafe_allow_html=True,
    )

    st.title("🧬 Clasificación de Cáncer de Piel (EfficientNetB4)")
    st.info("Esta herramienta es solo para fines educativos. No es un diagnóstico médico.")

    model = load_model()

    col_left, col_center, col_right = st.columns([1.6, 1.1, 1.0])

    # ===========================
    # PANEL IZQUIERDO
    # ===========================
    with col_left:

        st.subheader("📁 Sube una imagen dermatoscópica")
        uploaded = st.file_uploader(
            "Selecciona una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        st.markdown("### 🧪 O prueba con imágenes de ejemplo")

        if "selected_example_path" not in st.session_state:
            st.session_state["selected_example_path"] = None

        # HAM
        with st.expander("📦 Ejemplos del dataset HAM10000"):
            cols = st.columns(2)
            for idx, ex in enumerate(EXAMPLE_IMAGES_HAM):
                c = cols[idx % 2]
                with c:
                    img = load_example_image(ex["path"])
                    if img:
                        st.image(img, caption=ex["title"], width=180)
                        if st.button("Usar esta imagen", key=f"use_{ex['id']}"):
                            st.session_state["selected_example_path"] = ex["path"]
                    else:
                        st.warning(f"No encontrado: {ex['path']}")

        # Externos
        with st.expander("🌍 Ejemplos externos"):
            cols2 = st.columns(2)
            for idx, ex in enumerate(EXAMPLE_IMAGES_EXTERNAL):
                c = cols2[idx % 2]
                with c:
                    img = load_example_image(ex["path"])
                    if img:
                        st.image(img, caption=ex["title"], width=180)
                        if st.button("Usar esta imagen", key=f"use_{ex['id']}"):
                            st.session_state["selected_example_path"] = ex["path"]
                    else:
                        st.warning(f"No encontrado: {ex['path']}")

    # ===========================
    # SELECCIÓN DE IMAGEN
    # ===========================
    img_pil = None

    if uploaded:
        img_pil = read_image_file(uploaded)
        st.session_state["selected_example_path"] = None
    else:
        selected = st.session_state.get("selected_example_path")
        if selected:
            img_pil = load_example_image(selected)

    # ===========================
    # PREDICCIÓN
    # ===========================
    if img_pil:
        img_array = preprocess_image(img_pil)
        heatmap, probs = make_gradcam(img_array, model)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = float(probs[pred_idx])

        grad_img = apply_medical_colormap(img_pil, heatmap)
        sorted_idx = np.argsort(probs)[::-1]

        with col_center:
            st.subheader("🩻 Imagen original")
            st.image(img_pil, use_container_width=True)

            st.subheader("🔥 Grad-CAM (SmoothGrad + Medical Jet)")
            st.image(grad_img, use_container_width=True)

        with col_right:
            st.subheader("📌 Predicción principal")
            st.markdown(f"### {pred_class.upper()} — {pred_prob*100:.1f}%")
            st.write(CLASS_DESCRIPTIONS[pred_class])

            # Confianza
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
            st.markdown(
                f"""
<div class="confidence-bar">
    <div class="confidence-inner" style="width:{width}%; background:{color};">
        {width}% 
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.subheader("📊 Probabilidades por clase")

            for i in sorted_idx:
                cls = CLASS_NAMES[i]
                pr = float(probs[i])
                st.markdown(f"- {get_color(pr)} **{cls.upper()}**: {pr*100:.1f}%")

    else:
        with col_center:
            st.subheader("Esperando imagen…")
            st.write("Sube una imagen o elige un ejemplo.")

    # ============================================================
    # MÉTRICAS DEL MODELO
    # ============================================================
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

        if st.button("📄 Mostrar classification_report.json"):
            st.json(report)

    else:
        st.warning("No se encontró el archivo classification_report_B4_v1_advanced.json.")

    # ============================================================
    # TARJETAS CLÍNICAS PREMIUM (CORREGIDAS)
    # ============================================================
    st.markdown("---")
    st.header("🧾 Fichas clínicas de cada tipo de lesión")

    # CSS premium — sin sangría
    st.markdown(
        """
<style>
.card-clinical {
    border-radius: 14px;
    padding: 20px;
    margin-top: 20px;
    background: rgba(255,255,255,0.08);
    border: 1px solid #2d3553;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.45);
    transition: all 0.25s ease-out;
}
.card-clinical:hover {
    transform: translateY(-6px);
    box-shadow: 0px 8px 28px rgba(0,0,0,0.45);
    border-color: #6a8dff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    LESION_INFO = {
        "akiec": {
            "name": "Actinic Keratosis / Bowen disease",
            "risk": "🟡 Precancerosa",
            "risk_color": "#f1c232",
            "desc": "Lesión intraepitelial con riesgo de progresar a carcinoma escamocelular."
        },
        "bcc": {
            "name": "Basal Cell Carcinoma",
            "risk": "🔴 Maligno",
            "risk_color": "#cc0000",
            "desc": "Cáncer cutáneo de crecimiento lento."
        },
        "bkl": {
            "name": "Benign Keratosis",
            "risk": "🟢 Benigno",
            "risk_color": "#6aa84f",
            "desc": "Incluye queratosis seborreicas, lentigos benignos y queratosis solares."
        },
        "df": {
            "name": "Dermatofibroma",
            "risk": "🟢 Benigno",
            "risk_color": "#6aa84f",
            "desc": "Tumor benigno, firme y estable."
        },
        "mel": {
            "name": "Melanoma",
            "risk": "🔴 Altamente maligno",
            "risk_color": "#e06666",
            "desc": "Neoplasia agresiva que requiere atención urgente."
        },
        "nv": {
            "name": "Melanocytic Nevus",
            "risk": "🟢 Benigno",
            "risk_color": "#6aa84f",
            "desc": "Nevus común (‘lunar’). Normalmente estable."
        },
        "vasc": {
            "name": "Vascular lesion",
            "risk": "🟢 Benigno",
            "risk_color": "#6aa84f",
            "desc": "Incluye angiomas y hemangiomas."
        }
    }

    # TARJETAS — HTML CORRECTO SIN SANGRÍA
    for cls in CLASS_NAMES:
        info = LESION_INFO[cls]

        st.markdown(
            f"""
<div class="card-clinical">
    <h3 style="margin-bottom: 6px; font-size: 24px;">
        {cls.upper()} — {info['name']}
    </h3>

    <p style="font-size:17px; margin-top:-5px; margin-bottom:12px;
              font-weight:bold; color:{info['risk_color']};">
        {info['risk']}
    </p>

    <p style="font-size:16px; line-height:1.45;">
        {info['desc']}
    </p>
</div>
""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
