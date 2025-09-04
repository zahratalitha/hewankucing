import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === Konfigurasi ===
MODEL_PATH = "anjingkucing.h5"
IMG_SIZE = 180  # samakan dengan saat training

# === Load model langsung ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Preprocessing & Prediksi ===
def predict(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    return prediction

st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

uploaded_file = st.file_uploader("Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    pred = predict(image)
    label = "🐶 Anjing" if pred < 0.5 else "🐱 Kucing"
    st.subheader(f"Hasil Prediksi: {label} ({pred:.2f})")
