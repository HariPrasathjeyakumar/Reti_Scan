import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ===============================
# 1) DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Check for Google Drive virus scan confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    # Save file in chunks
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    MODEL_PATH = "ddr_efficientnetb3_final.h5"

    if not os.path.exists(MODEL_PATH):
        FILE_ID = "1CrgDM9BONOITwSOQnyzbnJf7RoItRfS1"  # <-- YOUR MODEL ID
        st.write("Downloading model from Google Drive... (this happens only once)")
        download_file_from_google_drive(FILE_ID, MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# Load the model
model = load_model()

# ===============================
# 2) LABELS + IMAGE SIZE
# ===============================
label_map = {
    0: "Grade 0 — No DR",
    1: "Grade 1 — Mild DR",
    2: "Grade 2 — Moderate DR",
    3: "Grade 3 — Severe DR",
    4: "Grade 4 — Proliferative DR"
}

IMG_SIZE = (300, 300)

# ===============================
# 3) PREDICTION FUNCTION
# ===============================
def predict(image):
    img = image.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    return cls, conf

# ===============================
# 4) STREAMLIT UI
# ===============================
st.title("👁️ Diabetic Retinopathy Detection App")
st.write("Upload a retinal fundus image to classify DR severity.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        cls, conf = predict(image)

        st.success(f"### Prediction: **{label_map[cls]}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")
