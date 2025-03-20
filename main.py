import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("skin_cancer_cnn.h5")


def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img

st.title("Medical Diagnosis System for Skin Cancer Detection")
st.markdown(""""This is skin cancer detection application. Upload an image and model will give result""")

uploaded_image = st.file_uploader("choose an image....", type=["jpg","jpeg","png"])

if uploaded_image is not None:
    class_label, img = predict_skin_cancer(uploaded_image, model)

    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
    st.write(f"**Prediction: {class_label}**")

st.markdown("""
    ### About the Model:
    This model uses CNN architecture for predicting whether a skin lesion is **Benign** or **Malignant**
    
    ###Features:
    - **Input**: Skin lesion images
    - **Output**: **Benign** or **Malignant** classification
    
    How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.
    """)