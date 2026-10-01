"""Deep branch v2: bagged ShallowConvNets with inner-subject early stopping (GPU-resident windows).

Outer loop = leave-one-subject-out (test subject s_k, k = 0..9). Inside each fold, K members are trained:
member j (1..K) trains on the 8 subjects other than s_k and v_j = subjects[(k + j) % 10] and is early-stopped
on v_j, using the test metric itself as the criterion (mean per-recording AUC of the 8 s causally smoothed
P(active)). The probabilities of the K members are averaged. Member 1 uses exactly the train/validation split
of ensemble.py / deep.py, so "member 1 alone" is a re-run of the single-net baseline.

The ensemble with the fusion LR is the same as ensemble.py: 0.5 * P_lr + 0.5 * P_net, then 8 s causal EMA, AUC
per recording on the 4 s window grid (2 s net outputs are taken at the identical window end times).

Everything is causal: per-recording causal Euclidean alignment (deep.causal_euclidean_align) and causal EMA.

    python -m eegbci.deep2 feats                                  # fusion-LR features (4 s windows), cached
    python -m eegbci.deep2 lr none 1 2 3                          # LOSO fusion LR, true labels + drift-nulls
    python -m eegbci.deep2 net bag5_w2 --win 2 --K 5 --nulls none 1 2 3
    python -m eegbci.deep2 report bag5_w2 [bag5_w4 ...] --nulls none 1 2 3

Null 'none' = true labels; integer n = every recording's protocol circularly shifted, with the shifts drawn
exactly as ensemble.main(null_seed=n), so the nulls are paired with the orchestrator's ensemble nulls.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from . import config as C
from .data import load_recordings
from .deep import EEGNet, ShallowNet, augment, causal_euclidean_align
from .ensemble import adaptive_decision
from .evaluation import Bundle, ema_causal, make_model
from .windows import build_windows

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOOTH = 8.0
SCHEME = "task_rest"
OUT = C.RESULTS_DIR
torch.backends.cudnn.benchmark = True


def null_name(null):
    return "true" if null is None else f"null{null}"


def parse_null(s):
    return None if str(s).lower() in ("none", "true") else int(s)


def get_shifts(recs, null):
    """Identical to ensemble.main(null_seed): one uniform draw per recording, in load order."""
    if null is None:
        return {r.rec_id: 0.0 for r in recs}
    rng = np.random.default_rng(int(null))
    return {r.rec_id: float(rng.uniform(60, r.duration - 60)) for r in recs}


def meta_bundle(recs, win_s):
    """Bundle that only carries window metadata (labels); covariances are not kept in RAM."""
    return Bundle(recs, {r.rec_id: dict(meta=build_windows(r, win_s=win_s)["meta"]) for r in recs}, {})


def smoothed_auc(y, ok, p, smooth=SMOOTH):
    ps = ema_causal(np.asarray(p, np.float64)[:, None], smooth)[:, 0]
    return float(roc_auc_score(y[ok], ps[ok]))


# ----------------------------------------------------------------------------------------------- data
class NetData:
    """Causally aligned windows of all recordings in one (N, 1, n_ch, T) float32 tensor on the GPU."""

    def __init__(self, recs, win_s, dtype=torch.float32, channels=None):
        self.win_s = win_s
        self.B = meta_bundle(recs, win_s)
        self.offset, n0 = {}, 0
        for r in recs:
            n = len(self.B.W[r.rec_id]["meta"])
            self.offset[r.rec_id] = (n0, n0 + n)
            n0 += n
        self.T = int(round(win_s * 125))
        self.channels = channels
        n_ch = 30 if channels is None else len(channels)
        self.X = torch.empty((n0, 1, n_ch, self.T), dtype=dtype, device=DEV)
        for r in recs:
            raw = build_windows(r, win_s=win_s)["raw"]
            x = np.asarray(raw) / 10.0
            if channels is not None:                                   # align on the selected channels only
                x = np.ascontiguousarray(x[:, channels])
            xa = causal_euclidean_align(x)
            a, b = self.offset[r.rec_id]
            self.X[a:b, 0] = torch.from_numpy(xa).to(DEV, dtype)
            del raw, x, xa
        self.n_ch = n_ch

    def labels(self, shifts):
        return {rid: self.B.labels(rid, SCHEME, shifts[rid])[:2] for rid in self.offset}


@torch.no_grad()
def predict_rec(model, D, rid, bs=None):
    model.eval()
    bs = bs or (512 if D.T <= 250 else 256)
    a, b = D.offset[rid]
    out = [torch.softmax(model(D.X[i:min(i + bs, b)].float()), 1)[:, 1] for i in range(a, b, bs)]
    return torch.cat(out).cpu().numpy().astype(np.float64)


def val_score(model, D, val_rids, lab):
    s = [smoothed_auc(lab[r][0], lab[r][1], predict_rec(model, D, r)) for r in val_rids
         if len(np.unique(lab[r][0][lab[r][1]])) == 2]
    return float(np.mean(s)) if s else float("nan")


class ShallowGAP(nn.Module):
    """ShallowConvNet whose squared output is averaged over the WHOLE window (global average pooling):
    log-power of F learned spatio-temporal filters -> linear layer (an FBCSP-like learner). Sliding windows
    over a continuous state carry no within-window time-locking, so the 30 x F position-specific weights of
    ShallowNet are replaced by F weights."""

    def __init__(self, n_ch=30, n_t=250, n_cls=2, F=40, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, F, (1, 13)), nn.Conv2d(F, F, (n_ch, 1), bias=False), nn.BatchNorm2d(F))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(F, n_cls)

    def forward(self, x):
        h = torch.log(torch.clamp((self.net(x) ** 2).mean(dim=(2, 3)), 1e-6))
        return self.fc(self.drop(h))


def make_net(arch, n_ch, n_t):
    if arch == "shallow":
        return ShallowNet(n_ch=n_ch, n_t=n_t)
    if arch == "gap":
        return ShallowGAP(n_ch=n_ch, n_t=n_t)
    if arch == "eegnet":
        return EEGNet(n_ch=n_ch, n_t=n_t)
    raise ValueError(arch)


def train_member(D, train_rids, val_rids, lab, seed, arch="shallow", epochs=40, lr=1e-3, bs=256, patience_ep=8,
                 evals_per_epoch=2, wd=1e-2, amp=False):
    """Same optimiser/augmentation as deep.train; validation = smoothed per-recording AUC, evaluated
    ``evals_per_epoch`` times per epoch, patience counted in epochs. amp: float16 autocast for training only
    (square/log/softmax/loss stay float32 under autocast)."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx = torch.cat([torch.arange(*D.offset[r], device=DEV)[torch.from_numpy(lab[r][1]).to(DEV)] for r in train_rids])
    yall = torch.cat([torch.from_numpy(lab[r][0][lab[r][1]].astype(np.int64)) for r in train_rids]).to(DEV)
    model = make_net(arch, D.n_ch, D.T).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cnt = torch.bincount(yall, minlength=2).float()
    lossf = nn.CrossEntropyLoss(weight=len(yall) / (2 * cnt))
    amp = bool(amp) and DEV.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n = len(idx)
    n_steps = (n + bs - 1) // bs
    eval_at = {int(round(n_steps * (q + 1) / evals_per_epoch)) for q in range(evals_per_epoch)}
    best, best_state, best_at, bad, hist = -np.inf, None, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for s, i in enumerate(range(0, n, bs), 1):
            b = perm[i:i + bs]
            xb, yb = augment(D.X[idx[b]].float()), yall[b]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=amp):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if s in eval_at:
                v = val_score(model, D, val_rids, lab)
                hist.append(round(v, 4))
                model.train()
                if v > best:
                    best, best_state, best_at, bad = v, copy.deepcopy(model.state_dict()), ep + s / n_steps, 0
                else:
                    bad += 1
        sched.step()
        if bad >= patience_ep * evals_per_epoch:
            break
    if best_state is None:                    # validation never finite (single-class val labels): keep last model
        best, best_at = float("nan"), float(ep + 1)
    else:
        model.load_state_dict(best_state)
    return model, dict(val=float(best), best_epoch=float(best_at), epochs=ep + 1, sec=round(time.time() - t0, 1),
                       hist=hist)


# ----------------------------------------------------------------------------------------------- runs
def run_net(tag, win_s=2.0, K=5, nulls=(None,), arch="shallow", channels=None, **train_kw):
    recs = load_recordings()
    t0 = time.time()
    D = NetData(recs, win_s, channels=channels)
    print(f"[{tag}] data on {DEV}: {tuple(D.X.shape)} in {time.time() - t0:.0f}s", flush=True)
    subj = {r.rec_id: r.subject for r in recs}
    subjects = sorted(set(subj.values()))
    cfg = dict(tag=tag, win_s=win_s, K=K, arch=arch, channels=channels, **train_kw)
    for null in nulls:
        lab = D.labels(get_shifts(recs, null))
        path = OUT / f"deep2_net_{tag}_{null_name(null)}.npz"
        P, info, states = {}, [], {}
        for k, s in enumerate(subjects):
            test = [r for r in D.offset if subj[r] == s]
            preds = {r: [] for r in test}
            for j in range(1, K + 1):
                v = subjects[(k + j) % len(subjects)]
                tr = [r for r in D.offset if subj[r] not in (s, v)]
                va = [r for r in D.offset if subj[r] == v]
                model, inf = train_member(D, tr, va, lab, seed=100 * k + j, arch=arch, **train_kw)
                for r in test:
                    preds[r].append(predict_rec(model, D, r))
                if null is None:
                    states[f"{s}_m{j}"] = {n: t.detach().cpu() for n, t in model.state_dict().items()}
                info.append(dict(test=s, member=j, val_subject=v, **inf))
                print(f"[{tag} {null_name(null)}] {s} m{j} val {v}: {inf['val']:.3f} @ep {inf['best_epoch']:.2f} "
                      f"({inf['epochs']} ep, {inf['sec']:.0f}s)", flush=True)
                del model
            for r in test:
                P[r] = np.stack(preds[r]).astype(np.float32)
            aucs = []
            for r in test:
                y, ok = lab[r]
                if len(np.unique(y[ok])) == 2:
                    aucs.append(f"{r}: m1 {smoothed_auc(y, ok, P[r][0]):.3f} bag {smoothed_auc(y, ok, P[r].mean(0)):.3f}")
            print(f"[{tag} {null_name(null)}] TEST {s} ({time.time() - t0:.0f}s) " + " | ".join(aucs), flush=True)
            np.savez(path, **P)                                   # incremental (resumable by inspection)
        json.dump(dict(cfg=cfg, null=null_name(null), members=info), open(path.with_suffix(".json"), "w"), indent=1)
        if states:
            torch.save(dict(cfg=cfg, states=states), OUT / f"deep2_net_{tag}_true_models.pt")
    del D
    torch.cuda.empty_cache()


class DeepBag:
    """Causal streaming inference for a bag of nets (drop-in for the single ShallowNet of realtime.BrainSwitchDecoder).

    members: list of dict(arch, win_s, channels, state). Call once per decision with the broadband buffer
    (30, >= max T) in µV at 125 Hz whose last sample is the window end. The Euclidean-alignment reference is a
    running mean of the 30-channel covariance per window length, updated exactly as deep.causal_euclidean_align
    (plain mean over the first 40 windows, then EMA tau 60 s; the current window is whitened by the PAST mean);
    a channel-subset member uses the matching sub-block (= alignment on its own channels)."""

    def __init__(self, members, step_s=C.STEP_S, tau_s=60.0, warmup=40):
        from .features import invsqrtm
        self.invsqrtm = invsqrtm
        self.members = []
        for m in members:
            T = int(round(m["win_s"] * 125))
            net = make_net(m["arch"], 30 if m["channels"] is None else len(m["channels"]), T)
            net.load_state_dict(m["state"])
            self.members.append((net.eval(), T, None if m["channels"] is None else np.asarray(m["channels"])))
        self.a, self.warmup, self.ea = 1.0 - np.exp(-step_s / tau_s), warmup, {}

    @torch.no_grad()
    def __call__(self, buf):
        xs, refs = {}, {}
        for T in {T for _, T, _ in self.members}:
            x = np.asarray(buf[:, -T:], np.float32) / np.float32(10.0)
            c = x.astype(np.float64) @ x.astype(np.float64).T / T
            st = self.ea.get(T)
            refs[T] = c if st is None else st[0]
            if st is None:
                self.ea[T] = [c, 1]
            else:
                st[1] += 1
                st[0] = st[0] + (c - st[0]) / st[1] if st[1] <= self.warmup else (1 - self.a) * st[0] + self.a * c
            xs[T] = x
        ps = []
        for net, T, ch in self.members:
            x, ref = (xs[T], refs[T]) if ch is None else (xs[T][ch], refs[T][np.ix_(ch, ch)])
            xa = (self.invsqrtm(ref) @ x).astype(np.float32)
            ps.append(float(torch.softmax(net(torch.from_numpy(xa)[None, None]), 1)[0, 1]))
        return float(np.mean(ps))


def train_bag(recs, configs=(dict(arch="shallow", win_s=2.0, channels=None),), K=5, **train_kw):
    """Deployment: for each config, K members trained on all subjects but one validation subject (spread over the
    subject list), each early-stopped on its validation subject. Returns the member list for DeepBag."""
    subjects = sorted({r.subject for r in recs})
    vals = [subjects[int(i)] for i in np.linspace(0, len(subjects) - 1, K).round()]
    members = []
    for c in configs:
        D = NetData(recs, c["win_s"], channels=c["channels"])
        lab = D.labels({r.rec_id: 0.0 for r in recs})
        subj = {r.rec_id: r.subject for r in recs}
        for j, v in enumerate(vals, 1):
            tr = [r for r in D.offset if subj[r] != v]
            va = [r for r in D.offset if subj[r] == v]
            model, inf = train_member(D, tr, va, lab, seed=1000 + j, arch=c["arch"], **train_kw)
            members.append(dict(arch=c["arch"], win_s=c["win_s"], channels=c["channels"], val_subject=v,
                                val_auc=inf["val"], state={n: t.detach().cpu() for n, t in model.state_dict().items()}))
        del D
        torch.cuda.empty_cache()
    return members


def parity(tags, subject="S05", n_win=400):
    """Offline (GPU, whole recording) vs streaming DeepBag (CPU, window by window) for the saved true-label members
    of one held-out subject; both read the same cached broadband windows."""
    recs = load_recordings(verbose=False)
    members, cfgs = [], {}
    for tag in tags:
        z = torch.load(OUT / f"deep2_net_{tag}_true_models.pt", weights_only=True)
        cfgs[tag] = z["cfg"]
        members += [dict(arch=z["cfg"]["arch"], win_s=z["cfg"]["win_s"], channels=z["cfg"]["channels"], state=st)
                    for name, st in z["states"].items() if name.startswith(subject + "_")]
    r = next(r for r in recs if r.subject == subject)
    # offline
    p_off = []
    for tag in tags:
        cfg = cfgs[tag]
        D = NetData([r], cfg["win_s"], channels=cfg["channels"])
        for name, st in torch.load(OUT / f"deep2_net_{tag}_true_models.pt", weights_only=True)["states"].items():
            if name.startswith(subject + "_"):
                net = make_net(cfg["arch"], D.n_ch, D.T).to(DEV)
                net.load_state_dict(st)
                p_off.append(predict_rec(net, D, r.rec_id)[:n_win])
        del D
    p_off = np.mean(p_off, 0)
    # streaming (all members must share one window grid for this check)
    win = {cfgs[t]["win_s"] for t in tags}
    assert len(win) == 1, "parity check needs members with the same window length"
    raw = build_windows(r, win_s=win.pop())["raw"]
    bag = DeepBag([dict(m, state={k: v.cpu() for k, v in m["state"].items()}) for m in members])
    p_on = np.array([bag(np.asarray(raw[i])) for i in range(n_win)])
    print(f"parity {r.rec_id} ({len(members)} members, {n_win} windows): max |dp| {np.abs(p_on - p_off).max():.2e}")


def fusion_features():
    """FINAL_CFG features of the 4 s windows (cached; same values as ensemble.py)."""
    from .realtime import offline_features
    path = OUT / "deep2_fusion_feats.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    recs = load_recordings(verbose=False)
    F = {}
    for r in recs:
        W = build_windows(r, win_s=4.0)
        F[r.rec_id] = offline_features(W)
        del W
    np.savez(path, **F)
    return F


def run_lr(nulls=(None,)):
    from .realtime import FINAL_CFG
    recs = load_recordings()
    F = fusion_features()
    B4 = meta_bundle(recs, 4.0)
    B4.F = F
    subj = {r.rec_id: r.subject for r in recs}
    subjects = sorted(set(subj.values()))
    for null in nulls:
        t0 = time.time()
        shifts = None if null is None else get_shifts(recs, null)
        P = {}
        for s in subjects:
            test = [r for r in F if subj[r] == s]
            train_r = [r for r in F if subj[r] != s]
            X, y, _ = B4.subset(train_r, SCHEME, shifts)
            lr = make_model("lr", FINAL_CFG["C_reg"]).fit(np.concatenate(X), np.concatenate(y))
            for r in test:
                P[r] = lr.predict_proba(F[r])[:, 1]
        np.savez(OUT / f"deep2_lr_{null_name(null)}.npz", **P)
        print(f"LR {null_name(null)} done in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- report
def _to_grid4(p, t_src, t4):
    """Nearest source window end for every 4 s window end (identical to ensemble.py)."""
    if len(t_src) == len(t4) and np.allclose(t_src, t4):
        return p
    j = np.clip(np.searchsorted(t_src, t4), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t4) < np.abs(t_src[j] - t4), j - 1, j)
    return p[..., j]


def metrics(y, ok, p):
    ps = ema_causal(np.asarray(p, np.float64)[:, None], SMOOTH)[:, 0]
    return dict(auc=float(roc_auc_score(y[ok], ps[ok])),
                bacc_fixed=float(balanced_accuracy_score(y[ok], ps[ok] > 0.5)),
                bacc_adaptive=float(balanced_accuracy_score(y[ok], adaptive_decision(ps)[ok])))


def summarize(rows, subj, name):
    a = np.array([o[name]["auc"] for o in rows.values()])
    subjects = sorted({subj[r] for r in rows})
    sa = np.array([np.mean([rows[r][name]["auc"] for r in rows if subj[r] == s]) for s in subjects])
    return dict(auc=float(a.mean()), median=float(np.median(a)), subj_auc=float(sa.mean()),
                n_subj_above=int((sa > 0.5).sum()), n_subj=len(sa),
                bacc_fixed=float(np.mean([o[name]["bacc_fixed"] for o in rows.values()])),
                bacc_adaptive=float(np.mean([o[name]["bacc_adaptive"] for o in rows.values()])),
                per_subject={s: round(float(v), 3) for s, v in zip(subjects, sa)})


def report(tags, nulls=(None,), out_tag=None, verbose=True):
    """Combines saved LR and net predictions. Models: lr, <tag>:m1 (single net, member 1), <tag>:bag (mean of
    K members), ens:<tag>:m1 / ens:<tag>:bag (0.5 LR + 0.5 net), and, with several tags, the multi-tag bag."""
    recs = load_recordings(verbose=False)
    subj = {r.rec_id: r.subject for r in recs}
    t4 = {r.rec_id: build_windows(r, win_s=4.0)["meta"]["t_end"].to_numpy() for r in recs}
    t2 = {r.rec_id: build_windows(r, win_s=2.0)["meta"]["t_end"].to_numpy() for r in recs}
    B4 = Bundle(recs, {r.rec_id: dict(meta=build_windows(r, win_s=4.0)["meta"]) for r in recs}, {})
    allres = {}
    for null in nulls:
        shifts = get_shifts(recs, null)
        plr = np.load(OUT / f"deep2_lr_{null_name(null)}.npz")
        nets, inner = {}, {}
        for tag in tags:
            z = np.load(OUT / f"deep2_net_{tag}_{null_name(null)}.npz")
            nets[tag] = {k: z[k] for k in z.files}
            js = OUT / f"deep2_net_{tag}_{null_name(null)}.json"
            if js.exists():           # inner-validation score of the config in each outer fold (val subjects only)
                mem = json.load(open(js))["members"]
                inner[tag] = {s: float(np.nanmean([m["val"] for m in mem if m["test"] == s])) for s in {m["test"] for m in mem}}
        nested = len(tags) > 1 and len(inner) == len(tags)
        rows, chosen = {}, {}
        for r in recs:
            rid = r.rec_id
            if rid not in plr.files or any(rid not in nets[t] for t in tags):
                continue
            y, ok, _, _ = B4.labels(rid, SCHEME, shifts[rid])
            if len(np.unique(y[ok])) < 2:
                continue
            P = {"lr": plr[rid]}
            bags = []
            for tag in tags:
                pm = nets[tag][rid]
                src = t2[rid] if pm.shape[1] == len(t2[rid]) else t4[rid]
                pm = _to_grid4(pm, src, t4[rid])
                P[f"{tag}:m1"] = pm[0]
                P[f"{tag}:bag"] = pm.mean(0)
                bags.append(pm.mean(0))
                P[f"ens:{tag}:m1"] = 0.5 * (plr[rid] + pm[0])
                P[f"ens:{tag}:bag"] = 0.5 * (plr[rid] + pm.mean(0))
            if len(tags) > 1:
                P["multi:bag"] = np.mean(bags, 0)
                P["ens:multi:bag"] = 0.5 * (plr[rid] + P["multi:bag"])
            if nested:
                sc = [inner[t].get(r.subject, -np.inf) for t in tags]
                chosen[r.subject] = tags[int(np.argmax(sc))]
                P["nested:bag"] = P[f"{chosen[r.subject]}:bag"]
                P["ens:nested:bag"] = 0.5 * (plr[rid] + P["nested:bag"])
            rows[rid] = {k: metrics(y, ok, v) for k, v in P.items()}
        names = list(next(iter(rows.values())).keys())
        summ = {n: summarize(rows, subj, n) for n in names}
        allres[null_name(null)] = dict(per_rec=rows, summary=summ, nested_choice=chosen, inner=inner)
        if verbose and nested:
            print(f"  nested choice per test subject: {chosen}")
        if verbose:
            print(f"== {null_name(null)} ({len(rows)} recordings)")
            for n in names:
                s = summ[n]
                print(f"  {n:<26} AUC {s['auc']:.3f} med {s['median']:.3f} | subj {s['subj_auc']:.3f} "
                      f"({s['n_subj_above']}/{s['n_subj']}>0.5) | bacc fix {s['bacc_fixed']:.3f} ad {s['bacc_adaptive']:.3f}")
    if out_tag:
        json.dump(allres, open(OUT / f"deep2_report_{out_tag}.json", "w"), indent=1)
    return allres


def compare(report_tag, models):
    """Paired comparison with ensemble.py's saved results (ensemble_loso.json / ensemble_loso_null<n>.json):
    same recordings, same windows, same protocol shifts. Wilcoxon signed-rank on the 10 subject-level differences."""
    from scipy.stats import wilcoxon
    R = json.load(open(OUT / f"deep2_report_{report_tag}.json"))
    subj = {r.rec_id: r.subject for r in load_recordings(verbose=False)}
    for key, res in R.items():
        bf = OUT / ("ensemble_loso.json" if key == "true" else f"ensemble_loso_{key}.json")
        if not bf.exists():
            print(f"{key}: no baseline file {bf.name}")
            continue
        base = json.load(open(bf))
        common = [r for r in res["per_rec"] if r in base]
        for bname in ("shallow", "ensemble"):
            b = np.array([base[r][bname]["auc"] for r in common])
            line = f"{key:<6} baseline {bname:<8} {b.mean():.3f}"
            for m in models:
                a = np.array([res["per_rec"][r][m]["auc"] for r in common])
                ds = np.array([np.mean([a[i] - b[i] for i, r in enumerate(common) if subj[r] == s])
                               for s in sorted({subj[r] for r in common})])
                p = wilcoxon(ds).pvalue if np.any(ds != 0) else 1.0
                line += (f" | {m} {a.mean():.3f} (d {a.mean() - b.mean():+.3f}, subj better {int((ds > 0).sum())}/{len(ds)},"
                         f" p={p:.3f})")
            print(line)


def bench(win_s=2.0, n_epochs=2, variants=(("shallow", False), ("shallow", True), ("gap", False), ("gap", True))):
    """Timing of a few epochs on the full data (true labels, fold 0, member 1)."""
    recs = load_recordings()
    t0 = time.time()
    D = NetData(recs, win_s)
    print(f"data {tuple(D.X.shape)} in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB",
          flush=True)
    subj = {r.rec_id: r.subject for r in recs}
    subjects = sorted(set(subj.values()))
    lab = D.labels(get_shifts(recs, None))
    tr = [r for r in D.offset if subj[r] not in subjects[:2]]
    va = [r for r in D.offset if subj[r] == subjects[1]]
    for arch, amp in variants:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _, inf = train_member(D, tr, va, lab, seed=1, arch=arch, epochs=n_epochs, patience_ep=99, amp=amp)
        print(f"win {win_s} {arch} amp={amp}: {n_epochs} epochs {time.time() - t0:.1f}s, val hist {inf['hist']}, "
              f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--arch", default="shallow")
    ap.add_argument("--nulls", nargs="*", default=["none"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--evals", type=int, default=2)
    ap.add_argument("--po", action="store_true", help="12 parieto-occipital channels only (aligned on their own)")
    ap.add_argument("--amp", action="store_true", help="float16 autocast during training")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subject", default="S05")
    a = ap.parse_args()
    nulls = [parse_null(s) for s in a.nulls]
    if a.cmd == "feats":
        fusion_features()
    elif a.cmd == "lr":
        run_lr([parse_null(s) for s in a.tags] or [None])
    elif a.cmd == "net":
        from .realtime import PO
        run_net(a.tags[0], win_s=a.win, K=a.K, nulls=nulls, arch=a.arch, channels=PO if a.po else None,
                epochs=a.epochs, lr=a.lr, patience_ep=a.patience, evals_per_epoch=a.evals, amp=a.amp)
    elif a.cmd == "report":
        report(a.tags, nulls, out_tag=a.out)
    elif a.cmd == "compare":                       # python -m eegbci.deep2 compare <report_tag> <model> [<model> ...]
        compare(a.tags[0], a.tags[1:])
    elif a.cmd == "parity":                        # python -m eegbci.deep2 parity <tag> [<tag> ...] --subject S05
        parity(a.tags, a.subject)
    elif a.cmd == "bench":
        bench(a.win)
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
