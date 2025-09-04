import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from huggingface_hub import from_pretrained_keras

IMG_HEIGHT, IMG_WIDTH = 180, 180
@st.cache_resource
def load_trained_model():
    model = from_pretrained_keras("zahratalitha/klasifikasikucing")
    return model

model = load_trained_model()

class_names = ["Kucing 🐱", "Anjing 🐶"]
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(model, img, class_names):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)

    prob = preds[0][0]  # sigmoid
    pred_class = 1 if prob > 0.5 else 0
    return class_names[pred_class]


st.set_page_config(page_title="Klasifikasi Gambar", page_icon="🐶🐱", layout="centered")

st.title("🐶🐱 Klasifikasi Gambar: Anjing vs Kucing")
st.write("Upload gambar")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_column_width=True)
    with st.spinner("Sedang memproses..."):
        label = predict_image(model, img, class_names)

    st.markdown(
        f"<h2 style='text-align: center; color: green;'>Hasil Prediksi: {label}</h2>",
        unsafe_allow_html=True
    )
