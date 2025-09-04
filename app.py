import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

IMG_HEIGHT, IMG_WIDTH = 180, 180
CLASS_NAMES = ["Kucing 🐱", "Anjing 🐶"]

# -----------------------------
# Load Model (.h5 dari Hugging Face)
# -----------------------------
@st.cache_resource
def load_trained_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/klasifikasikucing",  # ganti dengan repo kamu
        filename="klasifikasikucing.h5"            # nama file di Hugging Face
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_trained_model()

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(img):
    img = img.convert("RGB")  # pastikan 3 channel
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# -----------------------------
# Prediksi
# -----------------------------
def predict_image(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    prob = preds[0][0]
    pred_class = 1 if prob > 0.5 else 0
    return CLASS_NAMES[pred_class]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🐶🐱", layout="centered")
st.title("🐶🐱 Klasifikasi Gambar: Anjing vs Kucing")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    with st.spinner("Sedang memproses..."):
        label = predict_image(img)

    st.success(f"Hasil Prediksi: {label}")
