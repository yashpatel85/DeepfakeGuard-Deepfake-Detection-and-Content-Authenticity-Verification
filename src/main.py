# src/main.py (FastAPI backend)
import uvicorn
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import timm
import torchvision.transforms as transforms

# Load model
model = timm.create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("D:/DeepfakeGuard/models/xception_best.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1).squeeze().tolist()

    result = {"real": probs[0], "fake": probs[1]}
    label = "REAL" if probs[0] > probs[1] else "FAKE"
    result["predicted_label"] = label

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
