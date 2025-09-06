import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from timm import create_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("D:/DeepfakeGuard/models/xception_best.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Preprocessing pipeline
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="DeepfakeGuard API", version="1.2")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold between real/fake"),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        predicted_label = "REAL" if real_prob >= threshold else "FAKE"

        return JSONResponse({
            "real": real_prob,
            "fake": fake_prob,
            "decision_threshold": threshold,
            "predicted_label": predicted_label
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -----------------------------
# Grad-CAM utilities
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (299, 299))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam


# Init Grad-CAM on last conv block of Xception
target_layer = model.conv4  # adjust depending on architecture
grad_cam = GradCAM(model, target_layer)


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        cam = grad_cam.generate(img_tensor)

        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Overlay on image
        img_np = np.array(image.resize((299, 299))) / 255.0
        overlay = 0.5 * heatmap + 0.5 * img_np
        overlay = np.uint8(255 * overlay)

        # Save to buffer
        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
