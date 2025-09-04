import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow import keras
from tensorflow.keras import layers

# === Konfigurasi halaman ===
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# === Parameter input ===
IMG_SIZE = 180
MODEL_REPO = "zahratalitha/hewankucing"  
MODEL_FILENAME = "klasifikasihewan.h5"   

def build_model():
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    model = build_model()
    model.load_weights(model_path)
    return model

model = load_model()

uploaded_file = st.file_uploader("📂 Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")   # EfficientNet butuh RGB
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

   
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

 
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])

    if prob > 0.5:
        st.success(f"🐶 Ini anjing (probabilitas {prob:.2f})")
    else:
        st.success(f"🐱 Ini kucing (probabilitas {1 - prob:.2f})")
