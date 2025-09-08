import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = timm.create_model("xception", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("models/xception_best.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# Define Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception input size
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =========================
# Streamlit UI
# =========================
st.title("🛡️ DeepfakeGuard - Deepfake Detection & Content Authenticity Verification")
st.write("Upload an image and the model will predict whether it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    predicted_label = "REAL" if real_prob > fake_prob else "FAKE"

    # Show Results
    st.subheader("🔎 Prediction Results")
    st.write(f"**Real:** {real_prob:.4f}")
    st.write(f"**Fake:** {fake_prob:.4f}")
    st.success(f"✅ Predicted Label: **{predicted_label}**")
