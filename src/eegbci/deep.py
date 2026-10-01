"""Deep models (EEGNet, ShallowConvNet) on causally aligned raw windows.

Input per window: (1, 30 channels, T) at 125 Hz, 1-40 Hz causal band-pass, whitened by the
running (causal) mean covariance of the same recording ("causal Euclidean alignment").

python -m eegbci.deep
"""
from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config as C
from .evaluation import ema_causal
from .features import invsqrtm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def causal_euclidean_align(raw, tau_s=60.0, step_s=C.STEP_S, warmup=40):
    """raw: (n, ch, T) time-ordered windows of one recording."""
    covs = np.einsum("nct,ndt->ncd", raw.astype(np.float64), raw.astype(np.float64)) / raw.shape[2]
    a = 1.0 - np.exp(-step_s / tau_s)
    out = np.empty_like(raw)
    m = covs[0]
    for i in range(len(raw)):
        out[i] = (invsqrtm(m) @ raw[i]).astype(np.float32)
        m = m + (covs[i] - m) / (i + 1) if i < warmup else (1 - a) * m + a * covs[i]
    return out


class EEGNet(nn.Module):
    def __init__(self, n_ch=30, n_t=250, n_cls=2, F1=8, D=2, F2=16, k1=64, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n, n_cls)

    def forward(self, x):
        return self.fc(self.net(x))


class ShallowNet(nn.Module):
    """ShallowConvNet: temporal conv -> spatial conv -> square -> mean pool -> log (band-power learner)."""

    def __init__(self, n_ch=30, n_t=250, n_cls=2, F=40, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, F, (1, 13)), nn.Conv2d(F, F, (n_ch, 1), bias=False), nn.BatchNorm2d(F))
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            n = self.pool(self.net(torch.zeros(1, 1, n_ch, n_t)) ** 2).numel()
        self.fc = nn.Linear(n, n_cls)

    def forward(self, x):
        h = torch.log(torch.clamp(self.pool(self.net(x) ** 2), 1e-6))
        return self.fc(self.drop(h.flatten(1)))


def augment(x):
    x = x * (1 + 0.1 * torch.randn(x.shape[0], 1, x.shape[2], 1, device=x.device))      # channel gain jitter
    x = x + 0.05 * torch.randn_like(x)
    shift = np.random.randint(-12, 13)
    return torch.roll(x, shift, dims=3)


def train(model, Xtr, ytr, Xva, yva, epochs=40, lr=1e-3, bs=256, patience=8):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr[:, None]), torch.tensor(ytr)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = augment(Xt[idx].to(DEV)), yt[idx].to(DEV)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        sched.step()
        auc = roc_auc_score(yva, predict(model, Xva)[:, 1])
        if auc > best:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    out = [torch.softmax(model(torch.tensor(X[i:i + bs, None]).to(DEV)), 1).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def main(arch="eegnet", split="loso"):
    from .data import load_recordings
    from .evaluation import Bundle
    from .windows import build_windows
    torch.manual_seed(0); np.random.seed(0)
    recs = load_recordings()
    W = {r.rec_id: build_windows(r) for r in recs}
    B = Bundle(recs, W, {rid: np.zeros((len(W[rid]["meta"]), 1)) for rid in W})
    Xa = {}
    for rid in W:
        x = W[rid]["raw"] / 10.0
        Xa[rid] = causal_euclidean_align(x)
    lab = {rid: B.labels(rid, "task_rest") for rid in W}
    subj = {rid: B.recs[rid].subject for rid in W}
    subjects = sorted(set(subj.values()))
    Net = {"eegnet": EEGNet, "shallow": ShallowNet}[arch]
    res = {}
    for k, s in enumerate(subjects):
        t0 = time.time()
        val_s = subjects[(k + 1) % len(subjects)]
        tr = [r for r in W if subj[r] not in (s, val_s)]
        va = [r for r in W if subj[r] == val_s]
        cat = lambda rs: (np.concatenate([Xa[r][lab[r][1]] for r in rs]), np.concatenate([lab[r][0][lab[r][1]] for r in rs]))
        Xtr, ytr = cat(tr); Xva, yva = cat(va)
        model, vauc, n_ep = train(Net(n_t=Xtr.shape[2]), Xtr, ytr, Xva, yva)
        for r in [r for r in W if subj[r] == s]:
            y, ok = lab[r][0], lab[r][1]
            P = predict(model, Xa[r])
            res[r] = dict(auc_raw=float(roc_auc_score(y[ok], P[ok, 1])),
                          auc=float(roc_auc_score(y[ok], ema_causal(P, 8.0)[ok, 1])))
        print(f"{arch} {s}: val {vauc:.3f} ({n_ep} ep, {time.time() - t0:.0f}s) test " +
              " ".join(f"{r}:{res[r]['auc']:.3f}" for r in W if subj[r] == s), flush=True)
    v = np.array([d["auc"] for d in res.values()]); v0 = np.array([d["auc_raw"] for d in res.values()])
    print(f"{arch} LOSO task_rest AUC mean {v.mean():.3f} median {np.median(v):.3f} (raw {v0.mean():.3f})")
    json.dump(res, open(C.RESULTS_DIR / f"deep_{arch}_{split}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
