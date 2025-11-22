import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import onnxruntime as ort

# ======================================================
# Custom UI Styling (Professional Theme)
# ======================================================
st.set_page_config(page_title="Retina DR Classifier", layout="centered")

st.markdown("""
    <style>
        body {background-color: #F7F9FC;}
        .title {text-align: center; font-size: 40px; font-weight: 700; margin-bottom: -10px;}
        .subtitle {text-align: center; color: #444; font-size: 17px; margin-bottom: 30px;}
        .upload-box {border: 2px dashed #4a90e2; padding: 25px; border-radius: 12px;}
        .result-card {
            background: white; 
            padding: 20px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 25px;
        }
        .pred-label {
            font-size: 28px; 
            font-weight: 800; 
        }
        .confidence {
            font-size: 18px; 
            margin-top: 4px;
        }
        .symptoms-title {
            font-size: 20px; 
            font-weight: 600; 
            margin-top: 15px;
        }
        .symptoms-text {
            color: #333;
            font-size: 16px;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)


# ======================================================
# 1) DOWNLOAD ONNX MODEL
# ======================================================
def download_from_drive(file_id, dest):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Virus-scan confirmation token
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
        FILE_ID = "1lzIEhnZhpRzMhiZqOY-vlwRr8avHstGm"  # your ONNX file ID
        with st.spinner("Downloading model..."):
            download_from_drive(FILE_ID, MODEL_FILE)
    
    return ort.InferenceSession(MODEL_FILE)


session = load_model()

IMG_SIZE = (300, 300)

# ======================================================
# Clinical Stages + Symptoms
# ======================================================
stage_map = {
    "No DR": {
        "color": "#2ecc71",
        "symptoms": [
            "Normal vision",
            "No visible retinal changes",
            "No hemorrhages or microaneurysms"
        ]
    },
    "Mild DR": {
        "color": "#f1c40f",
        "symptoms": [
            "Presence of microaneurysms",
            "Tiny retinal bleeding spots",
            "Usually no major vision symptoms"
        ]
    },
    "Moderate DR": {
        "color": "#e67e22",
        "symptoms": [
            "Blurry vision may occur",
            "Multiple hemorrhages",
            "Retinal hard exudates",
            "Possible macular edema"
        ]
    },
    "Severe DR": {
        "color": "#e74c3c",
        "symptoms": [
            "Significant blurry or reduced vision",
            "Large hemorrhages",
            "Cotton wool spots",
            "Abnormal new blood vessels (Stage 4)"
        ]
    }
}

# ======================================================
# 2) MODEL PREDICTION
# ======================================================
def predict(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # ---- New stage mapping ----
    if pred_class == 0:
        stage = "No DR"
    elif pred_class == 1:
        stage = "Mild DR"
    elif pred_class == 2:
        stage = "Moderate DR"
    else:   # 3 or 4
        stage = "Severe DR"

    return stage, confidence


# ======================================================
# 3) UI
# ======================================================
st.markdown("<div class='title'>👁️ Retina DR Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a retinal image to detect the stage of diabetic retinopathy.</div>", unsafe_allow_html=True)

st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("🔍 Analyze Retina"):
        with st.spinner("Analyzing image..."):
            stage, conf = predict(image)

        info = stage_map[stage]

        # Result card
        st.markdown(f"""
            <div class='result-card'>
                <div class='pred-label' style='color:{info["color"]};'>
                    {stage}
                </div>
                <div class='confidence'>
                    Confidence: <b>{conf*100:.2f}%</b>
                </div>
                <div class='symptoms-title'>Symptoms:</div>
                <div class='symptoms-text'>
                    <ul>
                        { "".join([f"<li>{sym}</li>" for sym in info["symptoms"]]) }
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
