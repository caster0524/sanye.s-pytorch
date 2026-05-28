"""Quick training - 5 epochs, saves model, suitable for sandbox constraints"""
import os, json, time, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np

BASE = Path(r"c:\sanyepytorch\projects\pytorch-web-app")
DATA = BASE / "public" / "datasets" / "classification_enhanced"
MODEL = BASE / "public" / "models" / "trained" / "resnet50_classifier.pth"
MODEL.parent.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")

raw = datasets.ImageFolder(str(DATA))
cls = raw.classes
print(f"Data: {len(raw)} imgs, {len(cls)} classes")

np.random.seed(42); idxs = np.arange(len(raw)); np.random.shuffle(idxs)
sp = int(0.8 * len(idxs)); tr_idx, vl_idx = idxs[:sp], idxs[sp:]
print(f"Train: {len(tr_idx)}, Val: {len(vl_idx)}")

tf_tr = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tf_vl = transforms.Compose([transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Sub(torch.utils.data.Dataset):
    def __init__(s, ds, i, t): s.ds=ds; s.i=i; s.t=t
    def __len__(s): return len(s.i)
    def __getitem__(s, j): img, lab = s.ds[s.i[j]]; return s.t(img), lab

ds_tr = Sub(raw, tr_idx, tf_tr)
ds_vl = Sub(raw, vl_idx, tf_vl)
ld_tr = DataLoader(ds_tr, 64, True, num_workers=0, pin_memory=True)
ld_vl = DataLoader(ds_vl, 64, False, num_workers=0, pin_memory=True)

m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, len(cls)))
m = m.to(DEV)
crit = nn.CrossEntropyLoss()
opt = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)

best = 0.0
for ep in range(5):
    m.train(); tl=0.0; tc=0; t0=time.time()
    for im, lb in ld_tr:
        im, lb = im.to(DEV), lb.to(DEV)
        opt.zero_grad(); o=m(im); l=crit(o,lb); l.backward(); opt.step()
        tl+=l.item()*im.size(0); tc+=o.argmax(1).eq(lb).sum().item()
    ta=100*tc/len(ds_tr); tl/=len(ds_tr)

    m.eval(); vl=0.0; vc=0
    with torch.no_grad():
        for im, lb in ld_vl:
            im, lb = im.to(DEV), lb.to(DEV)
            o=m(im); vl+=crit(o,lb).item()*im.size(0); vc+=o.argmax(1).eq(lb).sum().item()
    va=100*vc/len(ds_vl); vl/=len(ds_vl)

    dt=time.time()-t0
    print(f"Ep {ep+1}/5 | Train {ta:.1f}% | Val {va:.1f}% | {dt:.0f}s")
    if va > best:
        best = va
        torch.save({"model_state_dict": m.state_dict(), "classes": cls, "val_accuracy": va, "num_classes": len(cls)}, MODEL)
        print(f"  Saved! Best={best:.1f}%")

print(f"Done! Best val={best:.1f}%")