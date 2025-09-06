import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os



DATA_DIR = "D:/DeepfakeGuard/data_split"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
MODEL_SAVE_PATH = "D:/DeepfakeGuard/models/resnet18.pth"



def get_data_loaders(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform = transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)

    return train_loader, val_loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    train_loader, val_loader = get_data_loaders(DATA_DIR)

    model = models.resnet18(pretrained = True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")


        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok = True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()