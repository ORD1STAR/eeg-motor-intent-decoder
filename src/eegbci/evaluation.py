"""Leakage-free evaluation.

Splits (a block is never split between train and test):
  within   per recording, leave-one-cycle-out
  xsess    per subject, leave-one-recording-out (needs >= 2 recordings)
  loso     leave-one-subject-out

Controls:
  null     the same evaluation with each recording's protocol timeline circularly shifted
           (labels no longer match the brain activity, but drift and autocorrelation are intact).
           A real effect must beat this distribution, not just chance.
  time     a model that only sees the window's timestamp.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .windows import label_windows

SCHEMES = {
    "cmd3": lambda lab: (lab, lab >= 0),
    "task_rest": lambda lab: ((lab > 0).astype(int), lab >= 0),
    "push_walk": lambda lab: (lab - 1, (lab == C.PUSH) | (lab == C.WALK)),
}


def make_model(name="lr", C_reg=0.1):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C_reg, class_weight="balanced", max_iter=3000))
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise ValueError(name)


class Bundle:
    """Features + metadata for many recordings, relabelable on demand."""

    def __init__(self, recs, windows, feats):
        self.recs = {r.rec_id: r for r in recs}
        self.W = windows
        self.F = feats

    def labels(self, rid, scheme, shift=0.0, reject_artifacts=True):
        r = self.recs[rid]
        meta = self.W[rid]["meta"]
        lab, blk, cyc, tib = label_windows(meta, r.blocks, r.duration, shift)
        y, ok = SCHEMES[scheme](lab)
        if reject_artifacts:
            ok &= meta["ptp_max"].to_numpy() <= C.PTP_REJECT_UV
        return y, ok, cyc, blk

    def subset(self, rids, scheme, shifts=None, feats=None):
        X, Y, G = [], [], []
        for rid in rids:
            y, ok, cyc, blk = self.labels(rid, scheme, 0.0 if shifts is None else shifts[rid])
            F = (feats or self.F)[rid]
            X.append(F[ok]); Y.append(y[ok]); G.append(np.c_[cyc[ok], blk[ok]])
        return X, Y, G


def _score(y, p, proba=None):
    out = dict(bacc=float(balanced_accuracy_score(y, p)), n=int(len(y)))
    if proba is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, proba))
    return out


def eval_within(B, scheme, model="lr", shifts=None, feats=None, C_reg=0.1, rids=None):
    res = {}
    for rid in (rids or B.F):
        (X,), (y,), (G,) = B.subset([rid], scheme, shifts, feats)
        cyc = G[:, 0]
        yt, yp, pp = [], [], []
        for c in np.unique(cyc):
            tr, te = cyc != c, cyc == c
            if len(np.unique(y[tr])) < len(np.unique(y)) or te.sum() == 0:
                continue
            m = make_model(model, C_reg).fit(X[tr], y[tr])
            yt.append(y[te]); yp.append(m.predict(X[te]))
            if len(np.unique(y)) == 2:
                pp.append(m.predict_proba(X[te])[:, 1])
        if yt:
            res[rid] = _score(np.concatenate(yt), np.concatenate(yp), np.concatenate(pp) if pp else None)
    return res


def ema_causal(P, tau_s, step_s=C.STEP_S):
    """Exponential moving average over time-ordered rows (causal)."""
    if not tau_s:
        return P
    a = 1.0 - np.exp(-step_s / tau_s)
    out = np.empty_like(P)
    m = P[0]
    for i in range(len(P)):
        m = (1 - a) * m + a * P[i]
        out[i] = m
    return out


def eval_groups(B, scheme, groups, model="lr", shifts=None, feats=None, C_reg=0.1, smooth_s=0.0):
    """groups: list of (test_rids, train_rids). The test recording is decoded window by window over
    its whole timeline (as online); ``smooth_s`` applies a causal EMA to the class probabilities."""
    res = {}
    F_all = feats or B.F
    for test, train in groups:
        Xtr, ytr, _ = B.subset(train, scheme, shifts, feats)
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        m = make_model(model, C_reg).fit(Xtr, ytr)
        for rid in test:
            y, ok, _, _ = B.labels(rid, scheme, 0.0 if shifts is None else shifts[rid])
            if len(np.unique(y[ok])) < 2:
                continue
            P = ema_causal(m.predict_proba(F_all[rid]), smooth_s)[ok]
            pred = m.classes_[P.argmax(1)]
            res[rid] = _score(y[ok], pred, P[:, 1] if P.shape[1] == 2 else None)
    return res


def xsess_groups(B, same_scenario=False):
    out = []
    for rid, r in B.recs.items():
        train = [q for q, s in B.recs.items() if s.subject == r.subject and q != rid
                 and (not same_scenario or s.scenario == r.scenario)]
        if train:
            out.append(([rid], train))
    return out


def loso_groups(B):
    subs = sorted({r.subject for r in B.recs.values()})
    return [([q for q, r in B.recs.items() if r.subject == s], [q for q, r in B.recs.items() if r.subject != s])
            for s in subs]


def summarize(res, key="bacc"):
    v = np.array([d[key] for d in res.values() if key in d])
    return dict(mean=float(v.mean()), median=float(np.median(v)), n=len(v)) if len(v) else dict(mean=np.nan, median=np.nan, n=0)


def random_shifts(B, rng, min_s=60.0):
    """Shifts drawn within the analysed span (after any excluded eyes-closed baseline)."""
    out = {}
    for rid, r in B.recs.items():
        lo = min((b[0] for b in r.blocks if b[2] is not None), default=0.0)
        out[rid] = float(rng.uniform(min_s, (r.duration - lo) - min_s))
    return out


def null_distribution(fn, B, n_perm=50, seed=0, n_jobs=8, key="bacc"):
    """fn(shifts) -> result dict. Returns mean score for each of n_perm circular shifts."""
    rng = np.random.default_rng(seed)
    shift_sets = [random_shifts(B, rng) for _ in range(n_perm)]
    vals = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(lambda s: summarize(fn(s), key)["mean"])(s) for s in shift_sets)
    return np.array(vals)


def time_features(B):
    """Control features: only the window timestamp (and its square)."""
    out = {}
    for rid in B.F:
        t = B.W[rid]["meta"]["t_end"].to_numpy() / 500.0
        out[rid] = np.c_[t, t ** 2]
    return out


def report(name, obs, null=None, key="bacc"):
    s = summarize(obs, key)
    line = f"{name:<52} {key} mean {s['mean']:.3f} median {s['median']:.3f} (n={s['n']})"
    if null is not None and len(null):
        p = (1 + np.sum(null >= s["mean"])) / (1 + len(null))
        line += f" | null {null.mean():.3f}±{null.std():.3f} p={p:.3f}"
        s.update(null_mean=float(null.mean()), null_std=float(null.std()), p=float(p))
    print(line, flush=True)
    return s
