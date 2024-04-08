import os
import cv2
import numpy as np
import tensorflow as tf

import streamlit as st
from PIL import Image

import tempfile


def classify_digit(model, image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # resize to match input shape
    image = np.invert(image)
    image = image / 255.0  # normalize to [0,1] range
    image = image.reshape(1, 28, 28)  # reshape to match input shape
    prediction = model.predict(image)
    return np.argmax(prediction)


def resize_image(image, target_size):
    image = Image.open(image)
    image = image.resize(target_size)
    return image

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🧠")

st.title("Handwritten Digit Recognition")
st.caption("Upload an image of a handwritten digit and let the model predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_np = np.array(Image.open(uploaded_file))
    temp_image_path = os.path.join(tempfile.gettempdir(), "temp_image.jpg")
    cv2.imwrite(temp_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    resized_image = resize_image(temp_image_path, (300, 300))

    col1, col2, col3 = st.columns(3)

    with col2:
        st.image(resized_image, caption="Uploaded Image", use_column_width=True)

    submit = st.button("Predict Digit")

    if submit:
        model = tf.keras.models.load_model("handwrittendigit.keras")
        prediction = classify_digit(model, temp_image_path)
        st.subheader("Prediction Result")
        st.success(f"The predicted digit is: {prediction}")

    os.remove(temp_image_path)