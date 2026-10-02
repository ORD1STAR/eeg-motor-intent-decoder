"""Experiment runner: python -m eegbci.explore <round>"""
import json
import sys
import time
from functools import partial

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .data import load_recordings
from .evaluation import (Bundle, eval_groups, eval_within, loso_groups, null_distribution, report, time_features,
                         xsess_groups)
from .features import recording_features
from .windows import build_windows, fill_cache

LOG = C.RESULTS_DIR / "explore_log.jsonl"


def load_bundle(feat_cfg, reref="none", win_s=C.WIN_S):
    """feat_cfg: one config dict, or a list of them (features concatenated)."""
    recs = load_recordings(verbose=False)
    Parallel(n_jobs=6)(delayed(fill_cache)(r, reref=reref, win_s=win_s) for r in recs)
    W = {r.rec_id: build_windows(r, reref=reref, win_s=win_s) for r in recs}                  # cheap: from cache
    cfgs = feat_cfg if isinstance(feat_cfg, list) else [feat_cfg]
    lite = {rid: dict(covs=W[rid]["covs"], meta=W[rid]["meta"][["t_end", "t_start", "scenario"]]) for rid in W}
    parts = [Parallel(n_jobs=8)(delayed(recording_features)(lite[rid], **c) for rid in W) for c in cfgs]
    F = {rid: np.concatenate([p[i] for p in parts], axis=1).astype(np.float32) for i, rid in enumerate(W)}
    W = {rid: dict(meta=W[rid]["meta"]) for rid in W}       # only metadata is needed from here on
    return Bundle(recs, W, F)


def run(name, B, scheme, split="within", n_perm=20, model="lr", C_reg=0.1, feats=None, key="bacc", smooth_s=0.0):
    if split == "within":
        fn = partial(eval_within, B, scheme, model, feats=feats, C_reg=C_reg)
        obs = fn(shifts=None)
    else:
        groups = xsess_groups(B) if split == "xsess" else loso_groups(B)
        fn = partial(eval_groups, B, scheme, groups, model, feats=feats, C_reg=C_reg, smooth_s=smooth_s)
        obs = fn(shifts=None)
        name = f"{name} smooth={smooth_s:g}s"
    null = null_distribution(lambda s: fn(shifts=s), B, n_perm=n_perm, key=key) if n_perm else None
    s = report(f"{name} | {scheme} | {split}", obs, null, key)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(name=name, scheme=scheme, split=split, model=model, C=C_reg, key=key, summary=s,
                                per_rec=obs, time=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    return obs, s


def round1():
    for label, cfg in [("fb-ts align=rec", dict(bands="fb", align="recording")),
                       ("fb-ts align=causal60 detrend=causal120", dict(bands="fb", align="causal", tau_s=60,
                                                                        detrend_mode="causal", detrend_tau=120))]:
        t = time.time(); B = load_bundle(cfg); print(f"[{label}] features in {time.time() - t:.0f}s")
        for scheme in ("cmd3", "task_rest", "push_walk"):
            run(label, B, scheme, "within")
        run("time-only", B, "cmd3", "within", feats=time_features(B))
        run("time-only", B, "task_rest", "within", feats=time_features(B))
        run("time-only", B, "push_walk", "within", feats=time_features(B))


PARIETO_OCC = ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def round2():
    """Brain switch (task vs rest): compact features, pooled training, smoothing. No nulls yet (screen)."""
    po = [C.CHANNELS.index(c) for c in PARIETO_OCC]
    cfgs = [("logbp PO alpha+beta", dict(bands=[1, 2, 3, 4], align="recording", kind="logbp", channels=po)),
            ("ts PO alpha+beta", dict(bands=[1, 2, 3, 4], align="recording", kind="ts", channels=po)),
            ("ts mi all-ch", dict(bands="mi", align="recording", kind="ts")),
            ("logbp all-ch fb causal-align60", dict(bands="fb", align="causal", tau_s=60, kind="logbp"))]
    for label, cfg in cfgs:
        B = load_bundle(cfg)
        run(label, B, "task_rest", "within", n_perm=0, key="auc")
        for split in ("xsess", "loso"):
            for sm in (0.0, 4.0):
                run(label, B, "task_rest", split, n_perm=0, key="auc", smooth_s=sm)


def round3():
    """Grid: causal re-centering time constant x feature set x smoothing (screen, no nulls)."""
    po = [C.CHANNELS.index(c) for c in PARIETO_OCC]
    for tau in (20, 40, 60, 120):
        for label, cfg in [(f"logbp all fb causal{tau}", dict(bands="fb", align="causal", tau_s=tau, kind="logbp")),
                           (f"logbp PO fb causal{tau}", dict(bands="fb", align="causal", tau_s=tau, kind="logbp", channels=po)),
                           (f"ts PO a+b causal{tau}", dict(bands=[1, 2, 3, 4], align="causal", tau_s=tau, kind="ts", channels=po))]:
            B = load_bundle(cfg)
            run(label, B, "task_rest", "within", n_perm=0, key="auc")
            for split in ("xsess", "loso"):
                for sm in (0.0, 4.0, 8.0):
                    run(label, B, "task_rest", split, n_perm=0, key="auc", smooth_s=sm)


def confirm():
    """Null-controlled confirmation of the two best brain-switch configurations."""
    po = [C.CHANNELS.index(c) for c in PARIETO_OCC]
    B = load_bundle(dict(bands="fb", align="causal", tau_s=60, kind="logbp"))
    run("CONFIRM logbp all fb causal60", B, "task_rest", "xsess", n_perm=30, key="auc", smooth_s=8.0)
    run("CONFIRM logbp all fb causal60", B, "task_rest", "within", n_perm=30, key="auc")
    B = load_bundle(dict(bands=[1, 2, 3, 4], align="causal", tau_s=60, kind="ts", channels=po))
    run("CONFIRM ts PO a+b causal60", B, "task_rest", "loso", n_perm=30, key="auc", smooth_s=8.0)
    run("CONFIRM ts PO a+b causal60", B, "task_rest", "xsess", n_perm=30, key="auc", smooth_s=8.0)


def round4():
    po = [C.CHANNELS.index(c) for c in PARIETO_OCC]
    A = dict(bands="fb", align="causal", tau_s=60, kind="logbp")
    T = dict(bands=[1, 2, 3, 4], align="causal", tau_s=60, kind="ts", channels=po)
    for win in (2.0, 4.0):
        for label, cfg in [("fusion logbp-all+ts-PO", [A, T]), ("logbp all fb", A), ("ts PO a+b", T)]:
            B = load_bundle(cfg, win_s=win)
            for model, creg in (("lr", 0.01), ("lr", 0.1), ("lr", 1.0), ("lda", 0)):
                for split in ("xsess", "loso"):
                    run(f"{label} win{win:g} {model}{creg:g}", B, "task_rest", split, n_perm=0, key="auc",
                        smooth_s=8.0, model=model, C_reg=creg)


def calibration(n_null=10):
    """Per-user calibration on the first minutes of a session (S1: 300 s, S2: 180 s, both classes present),
    tested on the rest of that session. Compares: cross-subject model, cross-subject + calibration data
    (calibration windows up-weighted), calibration-only model. Drift-null by circular protocol shift."""
    from sklearn.metrics import roc_auc_score
    from .evaluation import ema_causal, make_model, random_shifts
    po = [C.CHANNELS.index(c) for c in PARIETO_OCC]
    A = dict(bands="fb", align="causal", tau_s=60, kind="logbp")
    T = dict(bands=[1, 2, 3, 4], align="causal", tau_s=60, kind="ts", channels=po)
    B = load_bundle([A, T], win_s=4.0)
    T_CAL = {1: 300.0, 2: 180.0}

    def evaluate(shifts):
        out = {"loso": [], "loso+cal": [], "cal-only": []}
        for s in sorted({r.subject for r in B.recs.values()}):
            train = [q for q, r in B.recs.items() if r.subject != s]
            Xtr, ytr, _ = B.subset(train, "task_rest", shifts)
            Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
            base = make_model("lr", 0.01).fit(Xtr, ytr)
            for rid in [q for q, r in B.recs.items() if r.subject == s]:
                y, ok, _, _ = B.labels(rid, "task_rest", 0.0 if shifts is None else shifts[rid])
                t = B.W[rid]["meta"]["t_end"].to_numpy()
                cal = ok & (t <= T_CAL[B.recs[rid].scenario]); test = ok & (t > T_CAL[B.recs[rid].scenario] + 4)
                if len(np.unique(y[cal])) < 2 or len(np.unique(y[test])) < 2:
                    continue
                F = B.F[rid]
                w = np.r_[np.ones(len(ytr)), np.full(cal.sum(), 0.3 * len(ytr) / cal.sum())]
                both = make_model("lr", 0.01)
                both.fit(np.r_[Xtr, F[cal]], np.r_[ytr, y[cal]], logisticregression__sample_weight=w)
                own = make_model("lr", 0.01).fit(F[cal], y[cal])
                for name, m in (("loso", base), ("loso+cal", both), ("cal-only", own)):
                    p = ema_causal(m.predict_proba(F)[:, 1:], 8.0)[:, 0]
                    out[name].append(roc_auc_score(y[test], p[test]))
        return {k: float(np.mean(v)) for k, v in out.items()}, len(out["loso"])

    obs, n = evaluate(None)
    print(f"calibration (n={n} recordings): " + " | ".join(f"{k} {v:.3f}" for k, v in obs.items()), flush=True)
    rng = np.random.default_rng(0)
    nulls = [evaluate(random_shifts(B, rng))[0] for _ in range(n_null)]
    for k in obs:
        nv = np.array([d[k] for d in nulls])
        print(f"  {k:<9} {obs[k]:.3f}  null {nv.mean():.3f} ± {nv.std():.3f}  p={(1 + (nv >= obs[k]).sum()) / (1 + len(nv)):.3f}")
    json.dump(dict(obs=obs, nulls=nulls), open(C.RESULTS_DIR / "calibration.json", "w"), indent=1)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
