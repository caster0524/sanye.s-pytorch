"""Ultra-fast NLP sentiment trainer - 5 epochs, small batch"""
import json, torch, torch.nn as nn, numpy as np
import re, time
from pathlib import Path
from collections import Counter

BASE = Path(r"c:\sanyepytorch\projects\pytorch-web-app")
DATA = BASE / "public" / "datasets" / "nlp_enhanced" / "sentiment.json"
MODEL = BASE / "public" / "models" / "trained" / "nlp_sentiment.pth"
MODEL.parent.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATA, encoding="utf-8") as f:
    raw = json.load(f)
texts, labels = [d["text"] for d in raw], [d["label"] for d in raw]

def clean(t):
    return re.sub(r"[^\u4e00-\u9fff\w\s]", "", t).lower().strip()

np.random.seed(42); idxs = np.arange(len(texts)); np.random.shuffle(idxs)
sp = int(0.8 * len(idxs))
tr_i, vl_i = idxs[:sp], idxs[sp:]

# Build vocab
words = []
for i in tr_i:
    words.extend(clean(texts[i]).split())
vc = Counter(words)
vocab = {w: i+1 for i, (w, _) in enumerate(vc.most_common(2000))}
vsize = len(vocab)+1

MAXL = 80
def vec(txt):
    ws = clean(txt).split()
    v = np.zeros(MAXL, dtype=np.int64)
    for j, w in enumerate(ws[:MAXL]):
        v[j] = vocab.get(w, 0)
    return v

X_tr = torch.from_numpy(np.stack([vec(texts[i]) for i in tr_i]))
Y_tr = torch.tensor([labels[i] for i in tr_i])
X_vl = torch.from_numpy(np.stack([vec(texts[i]) for i in vl_i]))
Y_vl = torch.tensor([labels[i] for i in vl_i])

class FastLSTM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(vsize, 64, padding_idx=0)
        s.lstm = nn.LSTM(64, 64, 1, bidirectional=True, batch_first=True)
        s.fc = nn.Linear(128, 3)
    def forward(s, x):
        x = s.emb(x)
        _, (h, _) = s.lstm(x)
        return s.fc(torch.cat((h[-2], h[-1]), dim=1))

model = FastLSTM().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

N = len(tr_i)
names = {0: "negative", 1: "positive", 2: "neutral"}
best = 0.0
for ep in range(5):
    model.train(); tl=0.0; tc=0; t0=time.time()
    for i in range(0, N, 64):
        end = min(i+64, N)
        x, y = X_tr[i:end].to(DEV), Y_tr[i:end].to(DEV)
        opt.zero_grad()
        o = model(x); l = crit(o, y); l.backward()
        opt.step()
        tl += l.item()*(end-i); tc += o.argmax(1).eq(y).sum().item()
    ta = 100*tc/N

    model.eval(); vl=0.0; vc=0; V=len(vl_i)
    with torch.no_grad():
        for i in range(0, V, 64):
            end = min(i+64, V)
            x, y = X_vl[i:end].to(DEV), Y_vl[i:end].to(DEV)
            o = model(x); vl += crit(o,y).item()*(end-i); vc += o.argmax(1).eq(y).sum().item()
    va = 100*vc/V

    t=time.time()-t0
    print(f"Ep{ep+1}/5 T:{ta:.0f}% V:{va:.0f}% {t:.0f}s")
    if va > best:
        best=va
        torch.save({"model_state_dict": model.state_dict(),"vocab":vocab,
                    "vocab_size":vsize,"val_accuracy":va,"max_len":MAXL,
                    "label_names":names},MODEL)

print(f"DONE={best:.0f}%")