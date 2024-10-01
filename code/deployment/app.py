import streamlit as st
import requests
from PIL import Image

st.title("Happy and Sad classification App")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    if st.button('Classify'):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://fastapi:8000/predict/", files=files)

        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted Class: {result}")
        else:
            st.write("Error in prediction")
