"""Simplified classification trainer - robust single-file script"""
import sys, os, json, time, math
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import numpy as np

# Paths
BASE = Path(__file__).parent.parent.parent
DATA_PATH = BASE / "public" / "datasets" / "classification_enhanced"
MODEL_PATH = BASE / "public" / "models" / "trained" / "resnet50_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}", flush=True)

# Load data (no transform yet - we apply transforms differently)
raw_dataset = datasets.ImageFolder(str(DATA_PATH))
classes = raw_dataset.classes
num_classes = len(classes)
print(f"Dataset: {len(raw_dataset)} images, {num_classes} classes: {classes}", flush=True)

# Split indices
np.random.seed(42)
indices = np.arange(len(raw_dataset))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

# Transforms
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = SubsetDataset(raw_dataset, train_idx, train_tf)
val_ds = SubsetDataset(raw_dataset, val_idx, val_tf)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Model
print("Loading ResNet50...", flush=True)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_acc = 0.0
for epoch in range(30):
    model.train()
    t_loss, t_correct = 0.0, 0
    t0 = time.time()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * imgs.size(0)
        t_correct += outputs.argmax(1).eq(labels).sum().item()

    scheduler.step()
    t_acc = 100 * t_correct / len(train_ds)
    t_loss /= len(train_ds)

    model.eval()
    v_loss, v_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            v_loss += criterion(outputs, labels).item() * imgs.size(0)
            v_correct += outputs.argmax(1).eq(labels).sum().item()

    v_acc = 100 * v_correct / len(val_ds)
    v_loss /= len(val_ds)

    print(f"Epoch {epoch+1:2d}/30 | Train {t_acc:5.1f}% | Val {v_acc:5.1f}% | {time.time()-t0:.0f}s", flush=True)

    if v_acc > best_acc:
        best_acc = v_acc
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'classes': classes,
                     'val_accuracy': v_acc, 'num_classes': num_classes}, MODEL_PATH)
        print(f"  -> Saved (best={best_acc:.1f}%)", flush=True)

print(f"\nDone! Best val accuracy: {best_acc:.1f}%", flush=True)

# Test full dataset
model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
model.eval()
test_ds = SubsetDataset(raw_dataset, np.arange(len(raw_dataset)), val_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
print(f"Final accuracy: {100*correct/total:.1f}% ({correct}/{total})", flush=True)