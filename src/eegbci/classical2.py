"""Part 1 (classical models): can new features/models beat the fusion-LR LOSO baseline (AUC 0.671,
8 s causal smoothing, realtime.FINAL_CFG)?

All 13 candidates reuse ``realtime.offline_features`` with a cfg dict (so they are guaranteed to use
exactly the same causal recentering / tangent-space / log-power code as the established baseline) plus,
for three candidates, a small causal post-processing step (stretch / feature-smoothing) or a different
model. Final selection is NESTED: for every held-out subject, the best candidate is chosen by an inner
LOSO on the other 9 subjects only (mirrors benchmark.nested), then trained on those 9 and tested once on
the held-out subject. We also report, for transparency, the plain (non-nested) LOSO AUC of every
candidate -- descriptive only, never used to pick the final answer.

The per-window LOSO probabilities of candidate "fusion4" (= the established baseline, FINAL_CFG) are
cached to ``results/classical2_predictions.joblib`` for the decision-layer module (Part 2) to reuse.

python -m eegbci.classical2 nested      # main experiment (nested selection across 13 candidates)
python -m eegbci.classical2 plain       # descriptive: plain LOSO AUC of every candidate
python -m eegbci.classical2 null        # >=10 drift-null shifts of the nested winner's fixed config
python -m eegbci.classical2 all         # plain + nested + null, in that order
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import numpy as np
import joblib
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from . import config as C
from .data import load_recordings
from .evaluation import Bundle, ema_causal, make_model, loso_groups, null_distribution, summarize
from .features import recording_features
from .realtime import FINAL_CFG, offline_features, PO
from .windows import build_windows

RESULTS = C.RESULTS_DIR
PRED_PATH = RESULTS / "classical2_predictions.joblib"
SMOOTH = 8.0          # same causal smoothing as the baseline
C_REG = 0.01          # same LR regularisation as FINAL_CFG
N_NULL = 12

# ---------------------------------------------------------------- distinct feature "pieces" (each computed once per
# recording, then concatenated to build the 13 candidates -- avoids recomputing e.g. ts-PO-tau60 five times).
# (bands, channels, kind, tau_s, win_s) -> candidates that use it
PIECES = dict(
    logbp_all_60=dict(bands=[0, 1, 2, 3, 4], channels=None, kind="logbp", tau_s=60.0, win_s=4.0),
    logbp_all_120=dict(bands=[0, 1, 2, 3, 4], channels=None, kind="logbp", tau_s=120.0, win_s=4.0),
    logbp_all_noTheta_60=dict(bands=[1, 2, 3, 4], channels=None, kind="logbp", tau_s=60.0, win_s=4.0),
    logbp_PO_60=dict(bands=[0, 1, 2, 3, 4], channels=PO, kind="logbp", tau_s=60.0, win_s=4.0),
    ts_PO_60=dict(bands=[1, 2, 3, 4], channels=PO, kind="ts", tau_s=60.0, win_s=4.0),
    ts_PO_120=dict(bands=[1, 2, 3, 4], channels=PO, kind="ts", tau_s=120.0, win_s=4.0),
    logbp_all_60_w2=dict(bands=[0, 1, 2, 3, 4], channels=None, kind="logbp", tau_s=60.0, win_s=2.0),
    ts_PO_60_w2=dict(bands=[1, 2, 3, 4], channels=PO, kind="ts", tau_s=60.0, win_s=2.0),
)
# candidate name -> (grid, [piece names to concatenate])
RECIPES = dict(
    fusion4=(4, ["logbp_all_60", "ts_PO_60"]),
    logbp120=(4, ["logbp_all_120", "ts_PO_60"]),
    tspo120=(4, ["logbp_all_60", "ts_PO_120"]),
    both120=(4, ["logbp_all_120", "ts_PO_120"]),
    logbpPO=(4, ["logbp_PO_60", "ts_PO_60"]),
    bandsel=(4, ["logbp_all_noTheta_60", "ts_PO_60"]),
    fusion2=(2, ["logbp_all_60_w2", "ts_PO_60_w2"]),
)


# ---------------------------------------------------------------- causal post-processing (new, not in features.py)
def causal_stretch(F, tau_s=60.0, step_s=C.STEP_S, warmup_n=40, eps=1e-6, per_dim=False):
    """Approximate causal Riemannian-Procrustes 'stretch': divide each row by a causal running RMS
    computed from strictly PAST rows only (re-centering is already done upstream, in features.py, at the
    covariance level; this adds a second-order dispersion-normalisation on top, scalar or per-dimension)."""
    a = 1.0 - np.exp(-step_s / tau_s)
    out = np.empty_like(F)
    m2, k = None, 0
    for i in range(len(F)):
        if m2 is None:
            scale = 1.0
        else:
            scale = np.sqrt(np.maximum(m2, eps))
        out[i] = F[i] / scale
        v = F[i] ** 2 if per_dim else float(np.mean(F[i] ** 2))
        k += 1
        m2 = v if m2 is None else (m2 + (v - m2) / k if k <= warmup_n else (1 - a) * m2 + a * v)
    return out


def nearest_align(t_src, F_src, t_dst):
    """Nearest-neighbour row alignment (<0.25 s typical gap), same trick as ensemble.py's win2/win4 fusion."""
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t_dst) < np.abs(t_src[j] - t_dst), j - 1, j)
    return F_src[j]


class LinearEnsemble:
    """Average of LR, LDA and a Platt-calibrated linear SVM, all on the same features (fixed a priori:
    equal weights, no tuning)."""

    def __init__(self, C_reg=0.01):
        self.C_reg = C_reg

    def fit(self, X, y):
        self.lr = make_model("lr", self.C_reg).fit(X, y)
        self.lda = make_model("lda").fit(X, y)
        self.svm = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(LinearSVC(C=self.C_reg, class_weight="balanced", max_iter=5000, dual=False), cv=3),
        ).fit(X, y)
        return self

    def predict_proba(self, X):
        p = (self.lr.predict_proba(X)[:, 1] + self.lda.predict_proba(X)[:, 1] + self.svm.predict_proba(X)[:, 1]) / 3.0
        return np.c_[1 - p, p]


def lr_factory():
    return make_model("lr", C_REG)


def ens_factory():
    return LinearEnsemble(C_REG)


# ---------------------------------------------------------------- feature extraction (streaming: one recording at a time)
def compute_features(recs, verbose=True, sanity_check=True):
    """Returns (feats, meta4, meta2). feats[name][rid] is a (n_windows, d) float32 array. Streams one
    recording's cached windows at a time, computes each distinct feature "piece" (bands/channels/kind/tau)
    exactly once (PIECES), assembles the 8 base candidates from RECIPES, discards the covariance arrays
    immediately, and keeps everything in float32 -- all to keep peak RAM low on a shared machine."""
    feats = {n: {} for n in RECIPES}
    meta4, meta2 = {}, {}
    t0 = time.time()
    for i, r in enumerate(recs):
        W = {4.0: build_windows(r, win_s=4.0), 2.0: build_windows(r, win_s=2.0)}
        meta4[r.rec_id] = W[4.0]["meta"]
        meta2[r.rec_id] = W[2.0]["meta"]
        pieces = {}
        for name, spec in PIECES.items():
            pieces[name] = recording_features(W[spec["win_s"]], bands=spec["bands"], align="causal",
                                              tau_s=spec["tau_s"], kind=spec["kind"], channels=spec["channels"])
        if sanity_check and i == 0:
            ref = offline_features(W[4.0], FINAL_CFG)
            mine = np.concatenate([pieces["logbp_all_60"], pieces["ts_PO_60"]], axis=1)
            assert np.allclose(ref, mine, atol=1e-8), "fusion4 piece recipe does not match realtime.FINAL_CFG"
        for name, (grid, parts) in RECIPES.items():
            feats[name][r.rec_id] = np.concatenate([pieces[p] for p in parts], axis=1).astype(np.float32)
        del W, pieces
        if verbose and (i + 1) % 7 == 0:
            print(f"  features {i + 1}/{len(recs)} recordings ({time.time() - t0:.0f}s)", flush=True)

    # derived candidates (cheap post-processing on top of the base arrays above)
    feats["multiscale"] = {}
    for rid in feats["fusion4"]:
        t4 = meta4[rid]["t_end"].to_numpy()
        t2 = meta2[rid]["t_end"].to_numpy()
        f2a = nearest_align(t2, feats["fusion2"][rid], t4)
        feats["multiscale"][rid] = np.concatenate([feats["fusion4"][rid], f2a], axis=1).astype(np.float32)
    feats["stretch_scalar"] = {rid: causal_stretch(F, per_dim=False).astype(np.float32) for rid, F in feats["fusion4"].items()}
    feats["stretch_perdim"] = {rid: causal_stretch(F, per_dim=True).astype(np.float32) for rid, F in feats["fusion4"].items()}
    feats["featsmooth4s"] = {rid: ema_causal(F, 4.0).astype(np.float32) for rid, F in feats["fusion4"].items()}
    print(f"features ready in {time.time() - t0:.0f}s", flush=True)
    return feats, meta4, meta2


CANDIDATES = {name: (grid, name, lr_factory) for name, (grid, _parts) in RECIPES.items()}
CANDIDATES.update({
    # name                grid  feats key          model
    "multiscale":        (4, "multiscale", lr_factory),
    "stretch_scalar":    (4, "stretch_scalar", lr_factory),
    "stretch_perdim":    (4, "stretch_perdim", lr_factory),
    "featsmooth4s":      (4, "featsmooth4s", lr_factory),
    "linear_ensemble":   (4, "fusion4", ens_factory),
})
assert len(CANDIDATES) == 12, len(CANDIDATES)


def groups_for(B, subjects_test, subjects_train):
    return [([q for q, r in B.recs.items() if r.subject == s],
             [q for q, r in B.recs.items() if r.subject in subjects_train and r.subject != s])
            for s in subjects_test]


def eval_fold(B, groups, feats, model_factory, shifts=None, smooth_s=SMOOTH):
    """Like evaluation.eval_groups but takes an arbitrary model factory (not just a string name), so it
    also supports the LR+LDA+SVM ensemble candidate."""
    res = {}
    for test, train in groups:
        Xtr_l, ytr_l, _ = B.subset(train, "task_rest", shifts, feats)
        Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
        m = model_factory().fit(Xtr, ytr)
        for rid in test:
            y, ok, _, _ = B.labels(rid, "task_rest", 0.0 if shifts is None else shifts[rid])
            if len(np.unique(y[ok])) < 2:
                continue
            P = ema_causal(m.predict_proba(feats[rid]), smooth_s)[ok]
            pred = (P[:, 1] > 0.5).astype(int)
            res[rid] = dict(auc=float(roc_auc_score(y[ok], P[:, 1])),
                             bacc=float(balanced_accuracy_score(y[ok], pred)), n=int(ok.sum()))
    return res


def make_bundles(recs, feats, meta4, meta2):
    W4 = {rid: dict(meta=meta4[rid]) for rid in meta4}
    W2 = {rid: dict(meta=meta2[rid]) for rid in meta2}
    B4 = Bundle(recs, W4, feats["fusion4"])
    B2 = Bundle(recs, W2, feats["fusion2"])
    return B4, B2


def plain_table(B4, B2, feats, subjects):
    """Descriptive only: plain (non-nested) LOSO AUC for every candidate. NOT used for selection."""
    out = {}
    for name, (grid, fkey, model) in CANDIDATES.items():
        B = B4 if grid == 4 else B2
        res = eval_fold(B, loso_groups(B), feats[fkey], model)
        s = summarize(res, "auc")
        out[name] = s
        print(f"  plain LOSO  {name:<16} auc mean {s['mean']:.3f} median {s['median']:.3f} (n={s['n']})", flush=True)
    return out


def nested_run(B4, B2, feats, subjects):
    out = {}
    for s in subjects:
        inner = [x for x in subjects if x != s]
        scores = {}
        for name, (grid, fkey, model) in CANDIDATES.items():
            B = B4 if grid == 4 else B2
            scores[name] = summarize(eval_fold(B, groups_for(B, inner, inner), feats[fkey], model), "auc")["mean"]
        best = max(scores, key=scores.get)
        grid, fkey, model = CANDIDATES[best]
        B = B4 if grid == 4 else B2
        res = eval_fold(B, groups_for(B, [s], inner), feats[fkey], model)
        out[s] = dict(selected=best, inner=scores, test={k: v["auc"] for k, v in res.items()},
                      test_bacc={k: v["bacc"] for k, v in res.items()})
        print(f"{s}: selected {best:<16} (inner {scores[best]:.3f}) -> test AUC "
              + " ".join(f"{k}:{v['auc']:.3f}" for k, v in res.items()), flush=True)
    aucs = np.array([v for d in out.values() for v in d["test"].values()])
    baccs = np.array([v for d in out.values() for v in d["test_bacc"].values()])
    per_subj = np.array([np.mean(list(d["test"].values())) for d in out.values()])
    print(f"NESTED classical2 task_rest: recording AUC mean {aucs.mean():.3f} median {np.median(aucs):.3f}, "
          f"bacc mean {baccs.mean():.3f}; subject-level AUC mean {per_subj.mean():.3f}, "
          f"{(per_subj > 0.5).sum()}/{len(per_subj)} subjects above 0.5", flush=True)
    return out, dict(auc_mean=float(aucs.mean()), auc_median=float(np.median(aucs)), bacc_mean=float(baccs.mean()),
                      subject_auc_mean=float(per_subj.mean()), n_above_half=int((per_subj > 0.5).sum()), n_subj=len(per_subj))


def save_predictions(B4, feats, subjects):
    """Full-timeline (not just artifact-ok) LOSO P(active) of the established baseline (fusion4), for the
    decision-layer module (Part 2) to reuse without recomputing."""
    preds = {}
    for test, train in groups_for(B4, subjects, subjects):
        Xtr_l, ytr_l, _ = B4.subset(train, "task_rest", None, feats["fusion4"])
        Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
        m = lr_factory().fit(Xtr, ytr)
        for rid in test:
            p = m.predict_proba(feats["fusion4"][rid])[:, 1]
            t = B4.W[rid]["meta"]["t_end"].to_numpy()
            preds[rid] = dict(t=t, p=p)
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(preds, PRED_PATH)
    print(f"saved per-window LOSO P(active) for {len(preds)} recordings -> {PRED_PATH}", flush=True)
    return preds


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    recs = load_recordings(verbose=True)
    subjects = sorted({r.subject for r in recs})
    cache = RESULTS / "classical2_feats.joblib"
    if cache.exists():
        feats, meta4, meta2 = joblib.load(cache)
    else:
        feats, meta4, meta2 = compute_features(recs)
        joblib.dump((feats, meta4, meta2), cache)
    B4, B2 = make_bundles(recs, feats, meta4, meta2)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # sanity: candidate "fusion4" plain LOSO must reproduce the established 0.671 (ensemble.py "riemann")
    base = summarize(eval_fold(B4, loso_groups(B4), feats["fusion4"], lr_factory), "auc")
    print(f"BASELINE sanity (fusion4, plain LOSO, smooth {SMOOTH}s): auc mean {base['mean']:.3f} "
          f"median {base['median']:.3f} (expect ~0.671/0.683)", flush=True)

    if cmd in ("plain", "all"):
        plain = plain_table(B4, B2, feats, subjects)
        json.dump(plain, open(RESULTS / "classical2_plain.json", "w"), indent=1)

    nested_summary = None
    winner = "fusion4"
    if cmd in ("nested", "all"):
        nested_out, nested_summary = nested_run(B4, B2, feats, subjects)
        json.dump(dict(baseline_sanity=base, summary=nested_summary, per_subject=nested_out),
                  open(RESULTS / "classical2_nested.json", "w"), indent=1)
        winner = Counter(d["selected"] for d in nested_out.values()).most_common(1)[0][0]
        print(f"consensus winner (selected most often by the inner loop): {winner}", flush=True)

    if cmd in ("null", "all"):
        grid, fkey, model = CANDIDATES[winner]
        B = B4 if grid == 4 else B2
        obs = eval_fold(B, loso_groups(B), feats[fkey], model)
        obs_s = summarize(obs, "auc")
        null = null_distribution(lambda sh: eval_fold(B, loso_groups(B), feats[fkey], model, shifts=sh),
                                  B, n_perm=N_NULL, key="auc", n_jobs=1)
        p = (1 + np.sum(null >= obs_s["mean"])) / (1 + len(null))
        print(f"NULL CHECK winner={winner}: plain LOSO auc mean {obs_s['mean']:.3f} median {obs_s['median']:.3f} "
              f"| null {null.mean():.3f}+/-{null.std():.3f} (n={len(null)}) p={p:.3f} "
              f"| baseline fusion4 {base['mean']:.3f}", flush=True)
        json.dump(dict(winner=winner, obs=obs_s, null_mean=float(null.mean()), null_std=float(null.std()),
                        null_values=null.tolist(), p=float(p), baseline=base),
                  open(RESULTS / "classical2_null.json", "w"), indent=1)

    # always refresh the cached predictions for Part 2 (cheap: one extra LOSO pass of the baseline model)
    save_predictions(B4, feats, subjects)


if __name__ == "__main__":
    main()
