"""Classification trainer - writes progress to log file"""
import sys, os, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
import numpy as np

LOG = Path(__file__).parent / "train_log.txt"
def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

BASE = Path(__file__).parent.parent.parent
DATA_PATH = BASE / "public" / "datasets" / "classification_enhanced"
MODEL_PATH = BASE / "public" / "models" / "trained" / "resnet50_classifier.pth"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"DEVICE: {DEVICE}")

raw_ds = datasets.ImageFolder(str(DATA_PATH))
classes = raw_ds.classes
log(f"Dataset: {len(raw_ds)} images, {len(classes)} classes")

np.random.seed(42)
indices = np.arange(len(raw_ds)); np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]
log(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# Transform chains
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Apply transform per sample
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, ds, idxs, tf):
        self.ds = ds; self.idxs = idxs; self.tf = tf
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        img, lab = self.ds[self.idxs[i]]
        return self.tf(img), lab

train_ds = TransformedSubset(raw_ds, train_idx, train_tf)
val_ds = TransformedSubset(raw_ds, val_idx, val_tf)
train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_ldr = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

log("Loading ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

best_acc = 0.0
log("Starting 30 epochs...")
for ep in range(30):
    model.train()
    t_loss, t_corr = 0.0, 0
    t0 = time.time()
    for imgs, labs in train_ldr:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        opt.zero_grad()
        out = model(imgs); loss = criterion(out, labs)
        loss.backward(); opt.step()
        t_loss += loss.item() * imgs.size(0)
        t_corr += out.argmax(1).eq(labs).sum().item()
    sched.step()

    model.eval()
    v_loss, v_corr = 0.0, 0
    with torch.no_grad():
        for imgs, labs in val_ldr:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            out = model(imgs); v_loss += criterion(out, labs).item() * imgs.size(0)
            v_corr += out.argmax(1).eq(labs).sum().item()

    t_acc = 100 * t_corr / len(train_ds)
    v_acc = 100 * v_corr / len(val_ds)
    log(f"Epoch {ep+1:2d}/30 | Train {t_acc:.1f}% Val {v_acc:.1f}% | {time.time()-t0:.0f}s")

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save({'model_state_dict': model.state_dict(), 'classes': classes,
                     'val_accuracy': v_acc}, MODEL_PATH)
        log(f"  -> Saved best model ({best_acc:.1f}%)")

log(f"DONE! Best val: {best_acc:.1f}%")

# Final eval
model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
model.eval()
test_ds = TransformedSubset(raw_ds, np.arange(len(raw_ds)), val_tf)
test_ldr = DataLoader(test_ds, batch_size=64, num_workers=0)
corr = tot = 0
with torch.no_grad():
    for imgs, labs in test_ldr:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        corr += model(imgs).argmax(1).eq(labs).sum().item()
        tot += labs.size(0)
log(f"Final accuracy: {100*corr/tot:.1f}% ({corr}/{tot})")