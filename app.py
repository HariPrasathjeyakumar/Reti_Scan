import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ===============================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================

@st.cache_resource
def load_model():
    MODEL_PATH = "ddr_efficientnetb3_final.h5"

    if not os.path.exists(MODEL_PATH):
        # Replace with your file ID
        FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

        download_url = f"https://drive.google.com/uc?export=download&id=1CrgDM9BONOITwSOQnyzbnJf7RoItRfS1"
        response = requests.get(download_url)

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddr_efficientnetb3_final.h5")
    return model

model = load_model()

# Label mapping
label_map = {
    0: "Grade 0 — No DR",
    1: "Grade 1 — Mild DR",
    2: "Grade 2 — Moderate DR",
    3: "Grade 3 — Severe DR",
    4: "Grade 4 — Proliferative DR"
}

IMG_SIZE = (300, 300)

# ----------------------------
# Prediction Function
# ----------------------------
def predict(image):
    img = image.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    cls = int(np.argmax(preds))
    conf = float(np.max(preds))

    return cls, conf


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("👁️ Diabetic Retinopathy Detection App")
st.write("Upload a retinal fundus image to predict the DR grade.")

uploaded_file = st.file_uploader("Upload a retinal image (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Predict"):
        cls, conf = predict(image)

        st.success(f"### Prediction: **{label_map.get(cls)}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")


