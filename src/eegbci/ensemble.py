"""LOSO ensemble: Riemannian fusion model + ShallowConvNet, equal weights fixed a priori.

Also tests an unsupervised adaptive threshold for the on/off decision: a window is 'active' when
the smoothed probability exceeds its own causal running mean (tau = 120 s). This removes the
per-person calibration offset without any labels.

python -m eegbci.ensemble
"""
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from . import config as C
from .data import load_recordings
from .deep import ShallowNet, causal_euclidean_align, predict, train
from .evaluation import Bundle, ema_causal, make_model
from .realtime import FINAL_CFG, offline_features
from .windows import build_windows

SMOOTH = 8.0


def adaptive_decision(p, tau_s=120.0, step_s=C.STEP_S, warm=40):
    a = 1.0 - np.exp(-step_s / tau_s)
    out = np.zeros(len(p), bool)
    m = p[0]
    for i in range(len(p)):
        out[i] = p[i] > m
        m = m + (p[i] - m) / (i + 1) if i < warm else (1 - a) * m + a * p[i]
    return out


def main(null_seed=None):
    """null_seed: if given, every recording's protocol is circularly shifted (drift-matched null)."""
    torch.manual_seed(0); np.random.seed(0)
    recs = load_recordings()
    shifts = None
    if null_seed is not None:
        rng = np.random.default_rng(int(null_seed))
        shifts = {r.rec_id: float(rng.uniform(60, r.duration - 60)) for r in recs}
    sh = lambda rid: 0.0 if shifts is None else shifts[rid]
    W2 = {r.rec_id: build_windows(r, win_s=2.0) for r in recs}
    W4 = {r.rec_id: build_windows(r, win_s=4.0) for r in recs}
    B4 = Bundle(recs, W4, {rid: offline_features(W4[rid]) for rid in W4})
    B2 = Bundle(recs, W2, {rid: np.zeros((len(W2[rid]["meta"]), 1)) for rid in W2})
    Xa = {rid: causal_euclidean_align(np.asarray(W2[rid]["raw"]) / 10.0) for rid in W2}
    lab2 = {rid: B2.labels(rid, "task_rest", sh(rid)) for rid in W2}
    subj = {r.rec_id: r.subject for r in recs}
    subjects = sorted(set(subj.values()))
    out = {}
    for k, s in enumerate(subjects):
        test = [r for r in W4 if subj[r] == s]
        train_r = [r for r in W4 if subj[r] != s]
        X, y, _ = B4.subset(train_r, "task_rest", shifts)
        lr = make_model("lr", FINAL_CFG["C_reg"]).fit(np.concatenate(X), np.concatenate(y))
        val_s = subjects[(k + 1) % len(subjects)]
        tr = [r for r in W2 if subj[r] not in (s, val_s)]; va = [r for r in W2 if subj[r] == val_s]
        cat = lambda rs: (np.concatenate([Xa[r][lab2[r][1]] for r in rs]), np.concatenate([lab2[r][0][lab2[r][1]] for r in rs]))
        (Xtr, ytr), (Xva, yva) = cat(tr), cat(va)
        net, vauc, _ = train(ShallowNet(n_t=Xtr.shape[2]), Xtr, ytr, Xva, yva)
        for r in test:
            t4 = W4[r]["meta"]["t_end"].to_numpy(); t2 = W2[r]["meta"]["t_end"].to_numpy()
            p_lr = lr.predict_proba(B4.F[r])[:, 1]
            p_sh_all = predict(net, Xa[r])[:, 1]
            j = np.clip(np.searchsorted(t2, t4), 1, len(t2) - 1)
            j = np.where(np.abs(t2[j - 1] - t4) < np.abs(t2[j] - t4), j - 1, j)   # nearest window end (<0.25 s)
            p_sh = p_sh_all[j]
            y, ok, _, _ = B4.labels(r, "task_rest", sh(r))
            if len(np.unique(y[ok])) < 2:
                continue
            row = {}
            for name, p in (("riemann", p_lr), ("shallow", p_sh), ("ensemble", 0.5 * (p_lr + p_sh))):
                ps = ema_causal(p[:, None], SMOOTH)[:, 0]
                row[name] = dict(auc=float(roc_auc_score(y[ok], ps[ok])),
                                 bacc_fixed=float(balanced_accuracy_score(y[ok], ps[ok] > 0.5)),
                                 bacc_adaptive=float(balanced_accuracy_score(y[ok], adaptive_decision(ps)[ok])))
            out[r] = row
            print(f"{r}: " + " | ".join(f"{n} AUC {v['auc']:.3f} bacc {v['bacc_fixed']:.2f}/{v['bacc_adaptive']:.2f}"
                                        for n, v in row.items()), flush=True)
    for n in ("riemann", "shallow", "ensemble"):
        a = np.array([o[n]["auc"] for o in out.values()])
        bf = np.array([o[n]["bacc_fixed"] for o in out.values()]); ba = np.array([o[n]["bacc_adaptive"] for o in out.values()])
        subj_auc = [np.mean([out[r][n]["auc"] for r in out if subj[r] == s]) for s in subjects]
        print(f"{n:<9} LOSO AUC mean {a.mean():.3f} median {np.median(a):.3f} | subject-level {np.mean(subj_auc):.3f} "
              f"({sum(x > 0.5 for x in subj_auc)}/10 > 0.5) | bacc fixed {bf.mean():.3f} adaptive {ba.mean():.3f}")
    tag = "" if null_seed is None else f"_null{null_seed}"
    json.dump(out, open(C.RESULTS_DIR / f"ensemble_loso{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
