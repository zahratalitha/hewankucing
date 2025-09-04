import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

# === Konfigurasi ===
REPO_ID = "username/anjingkucing"   # ganti dengan repo HuggingFace kamu
FILENAME = "anjingkucing.h5"
IMG_SIZE = 180

# === Bangun ulang arsitektur model ===
def build_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model

# === Load model dengan weights ===
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = build_model()
    model.load_weights(model_path)  # pakai load_weights, bukan load_model
    return model

model = load_model()

# === Prediksi ===
def predict(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    return prediction

# === UI ===
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

uploaded_file = st.file_uploader("Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    pred = predict(image)
    label = "🐶 Anjing" if pred < 0.5 else "🐱 Kucing"
    st.subheader(f"Hasil Prediksi: {label} ({pred:.2f})")
