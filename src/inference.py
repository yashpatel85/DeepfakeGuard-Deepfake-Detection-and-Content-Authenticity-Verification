import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm import create_model
import sys
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- CONFIG ---
DATA_DIR = "D:/DeepfakeGuard/data_faces_split/test"  # Path to your test folder
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "D:/DeepfakeGuard/models/xception_best.pth"  # Update if different

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# --- LOAD TEST DATA ---
test_dataset = ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- LOAD MODEL ---
model = create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- EVALUATION ---
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# --- REPORT ---
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
