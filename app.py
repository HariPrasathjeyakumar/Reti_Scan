import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tflite_runtime.interpreter as tflite

# ===============================
# 1) DOWNLOAD TFLITE MODEL
# ===============================

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Check for Google Drive confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    # Save the file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    MODEL_PATH = "ddr_model.tflite"

    if not os.path.exists(MODEL_PATH):
        FILE_ID = "1bckOwYULzekNNGAZ6rzf0krvkBSBCTIH"  # <-- TFLite file ID
        st.write("Downloading TFLite model...")
        download_file_from_google_drive(FILE_ID, MODEL_PATH)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (300, 300)

# Labels
label_map = {
    0: "Grade 0 — No DR",
    1: "Grade 1 — Mild DR",
    2: "Grade 2 — Moderate DR",
    3: "Grade 3 — Severe DR",
    4: "Grade 4 — Proliferative DR",
}

# ===============================
# 2) PREDICT FUNCTION
# ===============================
def predict_tflite(image):
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)

    # EfficientNet preprocessing
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    return cls, conf


# ===============================
# 3) STREAMLIT UI
# ===============================
st.title("👁️ Diabetic Retinopathy Detection (TFLite Version)")
st.write("Upload a retinal fundus image to classify DR severity.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        cls, conf = predict_tflite(image)
        st.success(f"### Prediction: **{label_map[cls]}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")
