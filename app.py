import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

IMG_HEIGHT, IMG_WIDTH = 180, 180
@st.cache_resource
def load_trained_model():
    model = load_model("best_model.h5")  
    return model
model = load_trained_model()

class_names = ["Kucing 🐱", "Anjing 🐶"]  

def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # resize sesuai training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(model, img, class_names):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)

    # Sigmoid (Dense(1, activation="sigmoid"))
    prob = preds[0][0]
    pred_class = 1 if prob > 0.5 else 0
    class_label = class_names[pred_class]

    return class_label, float(prob if pred_class == 1 else 1 - prob)


st.title("🐶🐱 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar")

# Upload gambar
uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    with st.spinner("Sedang memproses..."):
        label, prob = predict_image(model, img, class_names)

    st.success(f"Hasil Prediksi: **{label}**")
    st.write(f"Tingkat keyakinan: **{prob:.2%}**")
