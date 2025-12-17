# ============================================================
# src/app_streamlit.py — versión estable para EfficientNetV2 (V2M)
# ============================================================

import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Importa config e inferencia (archivos nuevos)
from config_app import (
    BASE_DIR, MODEL_PATH, REPORT_PATH,
    ASSETS_DIR,
    load_labels_config, load_temperature
)
from inference import read_image_file, predict_proba
from gradcam import make_gradcampp_overlay


# ============================================================
# CLASES Y CONFIG
# ============================================================
CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, MODEL_NAME, MODEL_VERSION = load_labels_config()
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

# ============================================================
# EJEMPLOS
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


def load_example_image(relative_path: str):
    abs_path = BASE_DIR / relative_path
    if not abs_path.exists():
        return None
    return Image.open(abs_path).convert("RGB")


# ============================================================
# CARGA DEL MODELO (cacheado)
# ============================================================
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)


# ============================================================
# ESTILO Y TEMA
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
# APP
# ============================================================
def main():

    title_model = "EfficientNetV2"
    if MODEL_NAME and MODEL_VERSION:
        title_model = f"{MODEL_NAME} — {MODEL_VERSION}"
    elif MODEL_VERSION:
        title_model = f"Modelo — {MODEL_VERSION}"

    st.set_page_config(
        page_title="Clasificación de cáncer de piel",
        page_icon="🧬",
        layout="wide",
    )

    mode = st.sidebar.radio("🌓 Tema", ["Modo oscuro", "Modo claro"])
    apply_theme(dark_mode=(mode == "Modo oscuro"))

    st.sidebar.markdown(
        """
### 🔗 Enlaces del proyecto
- 📁 Repositorio (configurable)

---
### 👤 Diseñador
**Juan David Raigoso Espinosa**
📚 Economista  
📊 Mag. Ciencia de Datos
"""
    )

    st.title(f"🧬 Clasificación de Cáncer de Piel ({title_model})")
    st.info("Esta herramienta es solo para fines educativos. No es un diagnóstico médico.")

    # Validaciones de artifacts
    if not MODEL_PATH.exists():
        st.error(f"No encontré el modelo en: {MODEL_PATH}")
        st.stop()

    model = load_model_cached()
    temperature = load_temperature()  # puede ser None

    # Controles (sin dañar tu UI)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Inferencia")
    use_tta = st.sidebar.toggle("Usar TTA", value=True)
    tta_samples = st.sidebar.slider("Muestras TTA", min_value=1, max_value=16, value=8, step=1)
    use_calibration = st.sidebar.toggle("Usar temperatura (si existe)", value=True)

    col_left, col_center, col_right = st.columns([1.6, 1.1, 1.0])

    # ===========================
    # PANEL IZQUIERDO
    # ===========================
    with col_left:
        st.subheader("📁 Sube una imagen dermatoscópica")
        uploaded = st.file_uploader("Selecciona una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
        st.markdown("### 🧪 O prueba con imágenes de ejemplo")

        if "selected_example_path" not in st.session_state:
            st.session_state["selected_example_path"] = None

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
        # 1) Probabilidades (con TTA + temperatura opcional)
        T = temperature if (use_calibration and temperature is not None) else None
        probs = predict_proba(
            model=model,
            img_pil=img_pil,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            use_tta=use_tta,
            tta_samples=tta_samples,
            temperature=T,
        )

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = float(probs[pred_idx])
        sorted_idx = np.argsort(probs)[::-1]

        # 2) Grad-CAM++ overlay (real) — mantiene tu estética
        grad_img = None
        try:
             grad_img, cam_probs, cam_idx = make_gradcampp_overlay(
                model=model,
                img_pil=img_pil,
                img_width=IMG_WIDTH,
                img_height=IMG_HEIGHT,
            )
        except Exception as e:
          st.warning(f"No se pudo generar Grad-CAM++ (se mostrará solo la imagen original): {e}")

        with col_center:
            st.subheader("🩻 Imagen original")
            st.image(img_pil, use_container_width=True)

            st.subheader("🔥 Grad-CAM++ (Mapa de atención)")
            if grad_img is not None:
                st.image(grad_img, use_container_width=True)
            else:
                st.info("Grad-CAM++ no disponible (se muestra solo la imagen original).")

        with col_right:
            st.subheader("📌 Predicción principal")
            st.markdown(f"### {pred_class.upper()} — {pred_prob*100:.1f}%")
            st.write(CLASS_DESCRIPTIONS.get(pred_class, "Descripción no disponible."))

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
    st.header("📈 Métricas del modelo (Validación)")

    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)

        st.table({
            "Clase": list(CLASS_NAMES),
            "Precision": [report[c]["precision"] for c in CLASS_NAMES if c in report],
            "Recall": [report[c]["recall"] for c in CLASS_NAMES if c in report],
            "F1-score": [report[c]["f1-score"] for c in CLASS_NAMES if c in report],
        })

        if st.button("📄 Mostrar classification_report.json"):
            st.json(report)
    else:
        st.warning(f"No se encontró el archivo: {REPORT_PATH.name}")

    # ============================================================
    # TARJETAS CLÍNICAS
    # ============================================================
    st.markdown("---")
    st.header("🧾 Fichas clínicas de cada tipo de lesión")

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
        "akiec": {"name": "Actinic Keratosis / Bowen disease", "risk": "🟡 Precancerosa", "risk_color": "#f1c232",
                  "desc": "Lesión intraepitelial con riesgo de progresar a carcinoma escamocelular."},
        "bcc":   {"name": "Basal Cell Carcinoma", "risk": "🔴 Maligno", "risk_color": "#cc0000",
                  "desc": "Cáncer cutáneo de crecimiento lento."},
        "bkl":   {"name": "Benign Keratosis", "risk": "🟢 Benigno", "risk_color": "#6aa84f",
                  "desc": "Incluye queratosis seborreicas, lentigos benignos y queratosis solares."},
        "df":    {"name": "Dermatofibroma", "risk": "🟢 Benigno", "risk_color": "#6aa84f",
                  "desc": "Tumor benigno, firme y estable."},
        "mel":   {"name": "Melanoma", "risk": "🔴 Altamente maligno", "risk_color": "#e06666",
                  "desc": "Neoplasia agresiva que requiere atención urgente."},
        "nv":    {"name": "Melanocytic Nevus", "risk": "🟢 Benigno", "risk_color": "#6aa84f",
                  "desc": "Nevus común (‘lunar’). Normalmente estable."},
        "vasc":  {"name": "Vascular lesion", "risk": "🟢 Benigno", "risk_color": "#6aa84f",
                  "desc": "Incluye angiomas y hemangiomas."}
    }

    for cls in CLASS_NAMES:
        info = LESION_INFO.get(cls)
        if not info:
            continue
        st.markdown(
            f'<div class="card-clinical">'
            f'<h3 style="margin-bottom: 6px; font-size: 24px;">{cls.upper()} — {info["name"]}</h3>'
            f'<p style="font-size:17px; margin-top:-5px; margin-bottom:12px; font-weight:bold; color:{info["risk_color"]};">{info["risk"]}</p>'
            f'<p style="font-size:16px; line-height:1.45;">{info["desc"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ============================================================
    # INFO SOBRE EJEMPLOS
    # ============================================================
    st.markdown("---")
    st.markdown(
        """
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 14px; padding: 24px; margin-top: 20px; 
            box-shadow: 0px 6px 20px rgba(102, 126, 234, 0.3);">
    <h3 style="color: white; margin-bottom: 16px; font-size: 20px;">📸 Sobre las imágenes de ejemplo</h3>
    <div style="color: #f0f0f0; font-size: 15px; line-height: 1.6;">
        <p><strong>• Ejemplos del dataset HAM10000:</strong> Provienen del dataset de entrenamiento y representan el dominio para el que fue optimizado el modelo.</p>
        <p><strong>• Ejemplos externos:</strong> Son ilustrativos y pueden diferir en luz, enfoque o técnica. El modelo puede caer en precisión fuera de dominio.</p>
        <p><strong>• Comportamiento real:</strong> En modelos clínicos, la exactitud puede disminuir en dominios distintos al entrenamiento.</p>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
