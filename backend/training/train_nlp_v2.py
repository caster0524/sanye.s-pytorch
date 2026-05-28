"""NLP sentiment - 10 epochs, saves to file for verification"""
import json, time, re
from pathlib import Path
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(r"c:\sanyepytorch\projects\pytorch-web-app")
DATA = BASE / "public" / "datasets" / "nlp_enhanced" / "sentiment.json"
MODEL = BASE / "public" / "models" / "trained" / "nlp_sentiment.pth"
MODEL.parent.mkdir(parents=True, exist_ok=True)
LOG = BASE / "backend" / "training" / "nlp_train_log.txt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

log(f"Device: {DEV}")

with open(DATA, encoding="utf-8") as f:
    raw = json.load(f)
texts = [d["text"] for d in raw]
labels = [d["label"] for d in raw]
log(f"Data: {len(texts)} samples")

def clean(t):
    return re.sub(r"[^\u4e00-\u9fff\w\s]", "", t).lower().strip()

np.random.seed(42)
idxs = np.arange(len(texts))
np.random.shuffle(idxs)
sp = int(0.8 * len(idxs))
tr_i, vl_i = idxs[:sp], idxs[sp:]
log(f"Train: {len(tr_i)}, Val: {len(vl_i)}")

words = []
for i in tr_i:
    words.extend(clean(texts[i]).split())
wc = Counter(words)
vocab = {w: idx + 1 for idx, (w, _) in enumerate(wc.most_common(3000))}
vsize = len(vocab) + 1
log(f"Vocab: {vsize}")

MAXL = 100
def vec(txt):
    ws = clean(txt).split()
    v = np.zeros(MAXL, dtype=np.int64)
    for j, w in enumerate(ws[:MAXL]):
        v[j] = vocab.get(w, 0)
    return v

X_tr = np.stack([vec(texts[i]) for i in tr_i])
Y_tr = np.array([labels[i] for i in tr_i])
X_vl = np.stack([vec(texts[i]) for i in vl_i])
Y_vl = np.array([labels[i] for i in vl_i])

class SentLSTM(nn.Module):
    def __init__(self, vs, ed=128, ncls=3):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm = nn.LSTM(ed, 128, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, ncls)
    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        x = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.drop(x))

model = SentLSTM(vsize).to(DEV)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
ds_vl = TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(Y_vl))
ld_tr = DataLoader(ds_tr, 32, True)
ld_vl = DataLoader(ds_vl, 32, False)

names = {0: "negative", 1: "positive", 2: "neutral"}
best = 0.0
for ep in range(10):
    model.train(); tl=0.0; tc=0; t0=time.time()
    for xb, yb in ld_tr:
        xb, yb = xb.to(DEV), yb.to(DEV)
        opt.zero_grad()
        o=model(xb); l=crit(o,yb); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl+=l.item()*xb.size(0); tc+=o.argmax(1).eq(yb).sum().item()
    ta=100*tc/len(ds_tr); tl/=len(ds_tr)

    model.eval(); vl=0.0; vc=0
    with torch.no_grad():
        for xb, yb in ld_vl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            o=model(xb); vl+=crit(o,yb).item()*xb.size(0); vc+=o.argmax(1).eq(yb).sum().item()
    va=100*vc/len(ds_vl); vl/=len(ds_vl)

    dt=time.time()-t0
    log(f"Ep {ep+1:2d}/10 | Train {ta:.1f}% | Val {va:.1f}% | {dt:.0f}s")
    if va > best:
        best = va
        torch.save({
            "model_state_dict": model.state_dict(), "vocab": vocab,
            "vocab_size": vsize, "val_accuracy": va, "max_len": MAXL,
            "label_names": names
        }, MODEL)
        log(f"  Saved! Best={best:.1f}%")

# Evaluate per-class
model.eval()
cm = np.zeros((3, 3), dtype=int)
with torch.no_grad():
    for xb, yb in ld_vl:
        xb = xb.to(DEV)
        preds = model(xb).argmax(1).cpu().numpy()
        for p, t in zip(preds, yb.numpy()):
            cm[t][p] += 1
log("Confusion matrix (rows=true, cols=pred): neg/pos/neu")
for i, row in enumerate(cm):
    acc = 100 * row[i] / row.sum() if row.sum() > 0 else 0
    log(f"  {names[i]:8s}: {row.tolist()} acc={acc:.1f}%")

log(f"DONE! Best val={best:.1f}%")