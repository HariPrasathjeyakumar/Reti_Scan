import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import onnxruntime as ort
import altair as alt

# ======================================================
# PAGE CONFIG — EMERALD GREEN CLINICAL THEME
# ======================================================
st.set_page_config(page_title="Retina DR Scanner", layout="wide")

PRIMARY = "#2ecc71"
SECONDARY = "#145A32"
ACCENT = "#27AE60"
BACKGROUND = "#F0FAF3"
CARD_BG = "#FFFFFF"

st.markdown(f"""
<style>
body {{ background: {BACKGROUND}; }}
.section-title {{
    font-size: 24px; 
    font-weight: 700;
    margin-bottom: 4px;
    color: {SECONDARY};
}}
.card {{
    background: white; 
    padding: 18px; 
    border-radius: 12px;
    box-shadow: 0 5px 14px rgba(0,0,0,0.07);
}}
.kpi {{
    font-size: 20px;
    font-weight: 700;
    color: {PRIMARY};
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 800;
    margin-top: -6px;
    color: {SECONDARY};
}}
</style>
""", unsafe_allow_html=True)


# ======================================================
# DOWNLOAD ONNX MODEL
# ======================================================
def download_from_drive(file_id, dest):
    if os.path.exists(dest):
        return

    URL = "https://drive.google.com/uc?export=download"
    sess = requests.Session()
    resp = sess.get(URL, params={'id': file_id}, stream=True)

    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            resp = sess.get(URL, params={'id': file_id, 'confirm': v}, stream=True)
            break

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    MODEL = "ddr_model.onnx"
    if not os.path.exists(MODEL):
        FILE_ID = "1lzIEhnZhpRzMhiZqOY-vlwRr8avHstGm"   # your ONNX model
        download_from_drive(FILE_ID, MODEL)

    return ort.InferenceSession(MODEL)


session = load_model()
INPUT_NAME = session.get_inputs()[0].name


# ======================================================
# CLASS MAPPING (Clinical Stages)
# ======================================================
def map_stage(output_class):
    if output_class == 0:
        return "No DR"
    elif output_class == 1:
        return "Mild DR"
    elif output_class == 2:
        return "Moderate DR"
    else:
        return "Severe DR"


STAGE_COLORS = {
    "No DR": "#2ecc71",
    "Mild DR": "#f1c40f",
    "Moderate DR": "#e67e22",
    "Severe DR": "#e74c3c"
}


# ======================================================
# PREPROCESS
# ======================================================
def preprocess(img):
    img = img.resize((300, 300))
    arr = np.array(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, 0)
    return arr


# ======================================================
# PREDICT
# ======================================================
def predict(img):
    arr = preprocess(img)
    preds = session.run(None, {INPUT_NAME: arr})[0][0]
    probs = np.exp(preds) / np.sum(np.exp(preds))
    top_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    stage = map_stage(top_idx)
    return stage, confidence, probs


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=40)
note = st.sidebar.text_area("Doctor Notes (Optional)")


# ======================================================
# MAIN LAYOUT
# ======================================================
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.markdown("<div class='section-title'>Upload Retina Image</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Fundus Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Retina", use_container_width=True):
            with st.spinner("Diagnosing..."):
                stage, conf, probs = predict(img)

            # KPI CARDS
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"""
            <div class='card'>
                <div class='kpi'>Stage</div>
                <div class='kpi-value' style='color:{STAGE_COLORS[stage]}'>{stage}</div>
            </div>
            """, unsafe_allow_html=True)

            k2.markdown(f"""
            <div class='card'>
                <div class='kpi'>Confidence</div>
                <div class='kpi-value'>{conf*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            classes = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]
            k3.markdown(f"""
            <div class='card'>
                <div class='kpi'>Top Class</div>
                <div class='kpi-value'>{classes[np.argmax(probs)]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Probability Chart (Altair)
            df = {
                "Class": classes,
                "Probability": (probs * 100).tolist()
            }
            import pandas as pd
            df = pd.DataFrame(df)

            chart = alt.Chart(df).mark_bar(color=PRIMARY).encode(
                x="Class",
                y="Probability"
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

            st.success(f"Stage: {stage} ({conf*100:.1f}%)")


with col_right:
    st.markdown("<div class='section-title'>Stage Description</div>", unsafe_allow_html=True)

    if uploaded:
        stage, conf, probs = predict(img)

        if stage == "No DR":
            st.info("""
            ### **🟢 No DR**
            - Healthy retina  
            - No microaneurysms  
            - No hemorrhages  
            """)
        elif stage == "Mild DR":
            st.warning("""
            ### **🟡 Mild DR**
            - Microaneurysms present  
            - Early-stage damage  
            """)
        elif stage == "Moderate DR":
            st.warning("""
            ### **🟠 Moderate DR**
            - Hemorrhages  
            - Exudates  
            - Possible macular edema  
            """)
        else:
            st.error("""
            ### **🔴 Severe DR**
            - Severe hemorrhages  
            - Cotton wool spots  
            - High risk of vision loss  
            """)

    st.markdown("---")
    st.markdown("<div class='section-title'>Clinical Basis</div>", unsafe_allow_html=True)

    st.write("""
    Diabetic Retinopathy stages are determined using:
    - Microaneurysms  
    - Retinal hemorrhages  
    - IRMA  
    - Exudates  
    - Neovascularization  
    - Cotton wool spots  
    """)

