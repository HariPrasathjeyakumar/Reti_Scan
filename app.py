import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import onnxruntime as ort

# ===============================
# 1) DOWNLOAD ONNX MODEL
# ===============================

def download_from_drive(file_id, dest):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

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
        FILE_ID = "1lzIEhnZhpRzMhiZqOY-vlwRr8avHstGm"   # <--- your ONNX file ID
        st.write("Downloading ONNX model...")
        download_from_drive(FILE_ID, MODEL_FILE)
    return ort.InferenceSession(MODEL_FILE)

session = load_model()

IMG_SIZE = (300, 300)

label_map = {
    0: "Grade 0 — No DR",
    1: "Grade 1 — Mild DR",
    2: "Grade 2 — Moderate DR",
    3: "Grade 3 — Severe DR",
    4: "Grade 4 — Proliferative DR",
}

# ===============================
# 2) PREDICTION FUNCTION
# ===============================
def predict(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # EfficientNet preprocessing
    arr = np.expand_dims(arr, axis=0)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    return cls, conf

# ===============================
# 3) STREAMLIT UI
# ===============================
st.title("👁️ Diabetic Retinopathy Detection (ONNX Version)")
st.write("Upload a retinal image to classify DR severity.")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)
    if st.button("Predict"):
        cls, conf = predict(image)
        st.success(f"### Prediction: **{label_map[cls]}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")
