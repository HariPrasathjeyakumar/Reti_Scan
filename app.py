import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import onnxruntime as ort

# ===============================
# 0) CUSTOM CSS
# ===============================
st.set_page_config(page_title="Retina DR Classifier", layout="centered")

st.markdown("""
    <style>
        body {background-color: #F7F9FC;}
        .title {text-align: center; font-size: 40px; font-weight: 700; margin-bottom: -10px;}
        .subtitle {text-align: center; color: #555; font-size: 18px; margin-bottom: 30px;}
        .upload-box {border: 2px dashed #4a90e2; padding: 25px; border-radius: 12px;}
        .result-card {
            background: white; 
            padding: 20px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 25px;
        }
        .pred-label {
            font-size: 26px; 
            font-weight: 700; 
            color: #1a1a1a;
        }
        .confidence {
            font-size: 18px; 
            color: #4a4a4a;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 1) DOWNLOAD ONNX MODEL
# ===============================

def download_from_drive(file_id, dest):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Check GDrive confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    with open(dest, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    MODEL_FILE = "ddr_model.onnx"
    if not os.path.exists(MODEL_FILE):
        FILE_ID = "1lzIEhnZhpRzMhiZqOY-vlwRr8avHstGm"
        with st.spinner("Downloading model..."):
            download_from_drive(FILE_ID, MODEL_FILE)
    return ort.InferenceSession(MODEL_FILE)

session = load_model()

IMG_SIZE = (300, 300)

label_map = {
    0: "Grade 0 — No Diabetic Retinopathy",
    1: "Grade 1 — Mild Retinopathy",
    2: "Grade 2 — Moderate Retinopathy",
    3: "Grade 3 — Severe Retinopathy",
    4: "Grade 4 — Proliferative Retinopathy",
}

# ===============================
# 2) MODEL PREDICTION
# ===============================

def predict(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0      # EfficientNet preprocessing
    arr = np.expand_dims(arr, axis=0)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    return cls, conf


# ===============================
# 3) UI LAYOUT
# ===============================

st.markdown("<div class='title'>👁️ Retina DR Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a retinal fundus image to detect diabetic retinopathy severity.</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a retinal image", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 4) PROCESS IMAGE
# ===============================

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350, use_column_width=False)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing retina image..."):
            cls, conf = predict(image)
        
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='pred-label'>{label_map[cls]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence'>Confidence: <b>{conf*100:.2f}%</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
