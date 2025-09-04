import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# === Konfigurasi ===
REPO_ID = "zahratalitha/hewankucing"  # ganti dengan repo HuggingFace kamu
FILENAME = "klasifikasihewan.h5"
IMG_SIZE = 180

# === Download model dari HuggingFace ===
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# === Preprocessing & Prediksi ===
def predict(image):
    # pastikan RGB
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    return prediction

# === Streamlit UI ===
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

uploaded_file = st.file_uploader("Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    pred = predict(image)
    label = "🐶 Anjing" if pred < 0.5 else "🐱 Kucing"
    st.subheader(f"Hasil Prediksi: {label} ({pred:.2f})")
