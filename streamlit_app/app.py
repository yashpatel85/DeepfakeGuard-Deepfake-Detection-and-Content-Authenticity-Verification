# streamlit_app/app.py
import streamlit as st
import requests

st.set_page_config(page_title="DeepfakeGuard", page_icon="🛡️", layout="wide")

st.title("🛡️ DeepfakeGuard – Deepfake Detection Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Analyzing..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['predicted_label']}")
                st.json(result)
            else:
                st.error("API request failed.")
