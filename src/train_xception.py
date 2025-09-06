import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from tqdm import tqdm
import timm

# -------------------------
# Config
# -------------------------
DATA_DIR = "D:/DeepfakeGuard/data_faces_split"   # faces-only splits
MODEL_OUT = "D:/DeepfakeGuard/models/xception_best.pth"
THRESH_OUT = "D:/DeepfakeGuard/models/threshold.json"
EPOCHS = 50
BATCH_SIZE = 16
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # Windows-safe

# -------------------------
# Data transforms
# -------------------------
train_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    # Simulate video compression noise (mild)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Model, loss, optim, sched
# -------------------------
model = timm.create_model("xception", pretrained=True, num_classes=2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

# Cosine schedule with warmup
total_steps = EPOCHS * math.ceil(len(train_loader))
warmup_steps = max(100, int(0.05 * total_steps))

def cosine_warmup(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup)

scaler = torch.cuda.amp.GradScaler()

# -------------------------
# Helpers
# -------------------------
def eval_epoch():
    model.eval()
    val_losses, probs_all, labels_all, preds_all = [], [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            val_losses.append(loss.item())
            p = torch.softmax(out, 1)[:, 1]         # probability of class "fake" (index 1)
            pred = torch.argmax(out, 1)
            probs_all.extend(p.cpu().numpy().tolist())
            labels_all.extend(y.cpu().numpy().tolist())
            preds_all.extend(pred.cpu().numpy().tolist())

    val_loss = sum(val_losses) / len(val_losses)
    val_acc  = accuracy_score(labels_all, preds_all)
    val_f1   = f1_score(labels_all, preds_all)
    val_auc  = roc_auc_score(labels_all, probs_all)
    return val_loss, val_acc, val_f1, val_auc, probs_all, labels_all

def best_threshold_from_val(probs, labels):
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    j_best_idx = j.argmax()
    return float(thr[j_best_idx]), float(tpr[j_best_idx]), float(1 - fpr[j_best_idx])

# -------------------------
# Train
# -------------------------
best_auc = 0.0
global_step = 0
patience = 5
epochs_no_improve = 0

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses, train_correct, train_total = [], 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for x, y in loop:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        train_losses.append(loss.item())
        pred = torch.argmax(out, 1)
        train_correct += (pred == y).sum().item()
        train_total += y.size(0)

        loop.set_postfix(loss=sum(train_losses)/len(train_losses), acc=100.0*train_correct/train_total)

    # ----- Validation -----
    val_loss, val_acc, val_f1, val_auc, val_probs, val_labels = eval_epoch()
    print(f"Epoch {epoch}: "
          f"TrainLoss {sum(train_losses)/len(train_losses):.4f} | "
          f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} | ValAUC {val_auc:.4f}")

    # Checkpoint on AUC
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), MODEL_OUT)
        thr, tpr, tnr = best_threshold_from_val(val_probs, val_labels)
        with open(THRESH_OUT, "w") as f:
            json.dump({"optimal_threshold_fake": thr, "val_auc": best_auc, "val_tpr": tpr, "val_tnr": tnr}, f, indent=2)
        print(f"✅ New best AUC {best_auc:.4f}. Saved model to {MODEL_OUT} and threshold to {THRESH_OUT}.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹ Early stopping: no AUC improvement.")
            break

print("🎯 Training finished.")
