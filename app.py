import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# === Download & Load Model ===
MODEL_REPO = "zahratalitha/hewankucing" 
MODEL_FILENAME = "klasifikasihewan.h5"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
model = tf.keras.models.load_model(model_path)

uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    if prob > 0.5:
        st.success(f"🐶 Ini anjing (probabilitas {prob:.2f})")
    else:
        st.success(f"🐱 Ini kucing (probabilitas {1-prob:.2f})")
