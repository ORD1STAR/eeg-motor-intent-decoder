"""Decoding ablations for the confound question.

Everything is the fusion pipeline of realtime.FINAL_CFG (4 s windows, causal re-centering tau 60 s,
logistic regression C=0.01, causal 8 s EMA on probabilities), leave-one-subject-out on the identical
10 folds, scheme task_rest, AUC per recording. Drift-null = the identical evaluation with every
recording's protocol circularly shifted (evaluation.random_shifts).

    python -m eegbci.confounds_decode run <ablation> [n_null] [seed]
    python -m eegbci.confounds_decode summary

Ablations
  base                 FINAL_CFG fusion (reference, must reproduce ~0.671)
  no_ocular4           log-power part without Fp1 Fp2 F7 F8 (removed before re-centering, so they cannot leak)
  no_frontal           log-power part without Fp1 Fp2 F7 F3 Fz F4 F8 FT9 FT10
  no_front_temp        no_frontal and also without T7 T8 (temporalis EMG)
  set_<name>           log-power (5 bands) + tangent space (bands 1-4) restricted to one channel set
  blink_score          base model, scored only on blink-free windows
  blink_gate           trained and scored on blink-free windows, EMA updated only on blink-free windows
  blink_gate_strict    blink_gate + covariance re-centering reference updated only on blink-free windows
  blink_gate_shuffled  blink_gate with each recording's blink mask circularly shifted (same amount of
                       gating, unrelated to blinks) - control for the loss of windows (5 shifts)
  blink_gate_strict_shuffled   the same control for blink_gate_strict (one shift per seed argument)
  bands_notheta/_alpha/_beta, po_beta, po_lowbeta   frequency-band ablations (see BANDSETS)
  closure_gate(_shuffled)      remove windows with possible eye closure (EC-level alpha without blinks)
  proxy_ocular/_muscle/_all   ONLY eye / muscle proxy features (no EEG)
  base_plus_proxy      fusion + all proxy features
  base_resid           fusion features with every proxy-explained (linear) component removed per recording
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .confounds import CH, blinks_cache, blink_windows, good, bads_of, blink_fp_channels, bp
from .data import load_recordings
from .evaluation import Bundle, ema_causal, loso_groups, make_model, random_shifts
from .features import CausalRecenter, expm, invsqrtm, logm, tangent
from .preprocessing import causal_bandpass
from .realtime import FINAL_CFG
from .windows import DECIM, build_windows

OUT = C.RESULTS_DIR / "confounds_decoding.jsonl"
PO = ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
SETS = {
    "frontal7": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "frontal5_noFp": ["F7", "F3", "Fz", "F4", "F8"],
    "central7": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "centpar7": ["CP5", "CP1", "CP2", "CP6", "P3", "Pz", "P4"],
    "occ3": ["O1", "Oz", "O2"],
    "occ5": ["O1", "Oz", "O2", "P7", "P8"],
    "temporal4": ["T7", "T8", "FT9", "FT10"],
    "po12": PO,
}
DROP = {
    "no_ocular4": ["Fp1", "Fp2", "F7", "F8"],
    "no_frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT9", "FT10"],
    "no_front_temp": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT9", "FT10", "T7", "T8"],
}
ALL = list(C.CHANNELS)
BANDSETS = {   # name: (log-power channels, tangent-space channels, log-power bands, tangent-space bands)
    "bands_notheta": (ALL, PO, (1, 2, 3, 4), (1, 2, 3, 4)),     # no 4-8 Hz (blink-dominated) band
    "bands_alpha": (ALL, PO, (1,), (1,)),                        # 8-12 Hz only
    "bands_beta": (ALL, PO, (2, 3, 4), (2, 3, 4)),               # 12-30 Hz only (eye closure is mostly alpha)
    "po_beta": (PO, PO, (2, 3, 4), (2, 3, 4)),                   # posterior channels, 12-30 Hz
    "po_lowbeta": (PO, PO, (2, 3), (2, 3)),                      # posterior, 12-24 Hz (below most EMG)
}
WIN = FINAL_CFG["win_s"]
SMOOTH = FINAL_CFG["smooth_s"]
CREG = FINAL_CFG["C_reg"]
TAU = 60.0


# ------------------------------------------------------------------------------------------ features
def recentered(covs, upd=None, tau_s=TAU):
    """Causal re-centering (features.CausalRecenter); with ``upd`` the reference mean is only updated
    on windows where upd is True (strict gating)."""
    rc = CausalRecenter(tau_s, C.STEP_S)
    if upd is None:
        return rc.run(covs.astype(np.float64))
    out = []
    for Cm, u in zip(covs.astype(np.float64), upd):
        L = logm(Cm)
        ref = L if rc.m is None else rc.m
        P = invsqrtm(expm(ref))
        if rc.m is None:
            rc.m, rc.k = L, 1
        elif u:
            rc.k += 1
            rc.m = rc.m + (L - rc.m) / rc.k if rc.k <= rc.n_warm else (1 - rc.a) * rc.m + rc.a * L
        out.append(P @ Cm @ P)
    return np.stack(out)


def eeg_features(covs, logbp_ch, ts_ch, logbp_bands=(0, 1, 2, 3, 4), ts_bands=(1, 2, 3, 4), upd=None):
    feats = []
    if logbp_ch:
        ix = [CH[c] for c in logbp_ch]
        for b in logbp_bands:
            A = recentered(covs[:, b][:, ix][:, :, ix], upd)
            feats.append(np.log(np.maximum(np.diagonal(A, axis1=1, axis2=2), 1e-12)))
    if ts_ch:
        ix = [CH[c] for c in ts_ch]
        for b in ts_bands:
            feats.append(tangent(recentered(covs[:, b][:, ix][:, :, ix], upd)))
    return np.concatenate(feats, axis=1)


def causal_center(F, tau_s=TAU, step_s=C.STEP_S, warmup_s=20.0):
    """Subtract a causal exponentially-weighted mean of PAST windows (same idea as the covariance re-centering)."""
    a = 1.0 - np.exp(-step_s / tau_s)
    n_warm = max(1, int(warmup_s / step_s))
    out = np.empty_like(F, dtype=np.float64)
    m, k = None, 0
    for i in range(len(F)):
        ref = F[i] if m is None else m
        out[i] = F[i] - ref
        k += 1
        if m is None:
            m = F[i].astype(np.float64)
        elif k <= n_warm:
            m = m + (F[i] - m) / k
        else:
            m = (1 - a) * m + a * F[i]
    return out


PROXY_NAMES = ["blinks", "veog_lf", "heog_f", "heog_ft", "theta_fp",
               "emg30_temp", "emg55_temp", "emg30_front", "emg55_front", "emg30_neck", "emg55_neck"]
OCULAR = PROXY_NAMES[:5]
MUSCLE = PROXY_NAMES[5:]


def proxy_window_features(rec, meta, blink_t):
    """Per-window eye/muscle proxies with CAUSAL filters over the window's own samples (log power), then
    causally centred. Blink count uses the blink detector (offline)."""
    bads = bads_of(rec)
    x = rec.data * 1e6
    ends = np.round(meta["t_end"].to_numpy() * C.SFREQ).astype(int)
    n = int(WIN * C.SFREQ)

    def wpow(sig):
        cs = np.concatenate([[0.0], np.cumsum(sig.astype(np.float64) ** 2)])
        e = np.clip(ends, n, len(sig))
        return np.log((cs[e] - cs[e - n]) / n + 1e-12)

    def group(chs, lo, hi):
        chs = good(chs, bads)
        if not chs:
            return np.zeros(len(ends))
        y = causal_bandpass(x[[CH[c] for c in chs]], lo, hi)
        return np.mean([wpow(s) for s in y], axis=0)

    fp_all = good(["Fp1", "Fp2"], bads)
    fp = blink_fp_channels(bp(x[[CH[c] for c in fp_all]], 0.5, 8.0), fp_all)
    veog = x[[CH[c] for c in fp]].mean(0)
    cols = {}
    ts, te = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    cols["blinks"] = (np.searchsorted(blink_t, te) - np.searchsorted(blink_t, ts)).astype(float)
    cols["veog_lf"] = wpow(causal_bandpass(veog[None], 0.5, 4.0)[0])
    cols["heog_f"] = (wpow(causal_bandpass((x[CH["F7"]] - x[CH["F8"]])[None], 0.5, 4.0)[0])
                      if not {"F7", "F8"} & set(bads) else np.zeros(len(ends)))
    cols["heog_ft"] = (wpow(causal_bandpass((x[CH["FT9"]] - x[CH["FT10"]])[None], 0.5, 4.0)[0])
                       if not {"FT9", "FT10"} & set(bads) else np.zeros(len(ends)))
    cols["theta_fp"] = group(["Fp1", "Fp2"], 4, 8)
    for name, chs in (("temp", ["T7", "T8", "FT9", "FT10"]), ("front", ["Fp1", "Fp2", "F7", "F8"]),
                      ("neck", ["O1", "Oz", "O2", "P7", "P8"])):
        cols[f"emg30_{name}"] = group(chs, 30, 45)
        cols[f"emg55_{name}"] = group(chs, 55, 70)
    rec.release()
    F = np.stack([cols[k] for k in PROXY_NAMES], axis=1)
    return causal_center(F)


def proxy_features_all(recs, W):
    path = C.RESULTS_DIR / f"confounds_proxyfeat_win{WIN:g}.npz"
    if path.exists():
        z = np.load(path)
        return {rid: z[rid] for rid in z.files}
    BL = blinks_cache()
    out = {}
    for r in recs:
        out[r.rec_id] = proxy_window_features(r, W[r.rec_id]["meta"], np.asarray(BL[r.rec_id]["t"])).astype(np.float32)
        print(f"  proxies {r.rec_id}", flush=True)
    np.savez(path, **out)
    return out


# ------------------------------------------------------------------------------------------ evaluation
def ema_gated(p, keep, tau_s=SMOOTH, step_s=C.STEP_S):
    """Causal EMA that ignores gated-out windows (holds the last value)."""
    a = 1.0 - np.exp(-step_s / tau_s)
    out = np.empty_like(p)
    m = None
    for i in range(len(p)):
        if keep[i]:
            m = p[i] if m is None else (1 - a) * m + a * p[i]
        out[i] = np.nan if m is None else m
    return out


def evaluate(B, shifts=None, gate=None, train_gate=False, score_gate=False, gated_ema=False):
    """LOSO, identical folds for every ablation. gate: {rid: bool (True = clean window)}."""
    sh = (lambda rid: 0.0) if shifts is None else (lambda rid: shifts[rid])
    res = {}
    for test, train in loso_groups(B):
        X, Y = [], []
        for rid in train:
            y, ok, _, _ = B.labels(rid, "task_rest", sh(rid))
            if train_gate:
                ok = ok & gate[rid]
            X.append(B.F[rid][ok]); Y.append(y[ok])
        m = make_model("lr", CREG).fit(np.concatenate(X), np.concatenate(Y))
        for rid in test:
            y, ok, _, _ = B.labels(rid, "task_rest", sh(rid))
            p = m.predict_proba(B.F[rid])[:, 1]
            p = ema_gated(p, gate[rid]) if gated_ema else ema_causal(p[:, None], SMOOTH)[:, 0]
            sc = ok & gate[rid] if score_gate else ok
            sc = sc & np.isfinite(p)
            if len(np.unique(y[sc])) < 2:
                continue
            res[rid] = dict(auc=float(roc_auc_score(y[sc], p[sc])), n=int(sc.sum()),
                            n_rest=int((y[sc] == 0).sum()), n_task=int((y[sc] == 1).sum()))
    return res


def summarize(res, recs):
    a = np.array([v["auc"] for v in res.values()])
    subj = sorted({recs[r].subject for r in res})
    sa = np.array([np.mean([res[r]["auc"] for r in res if recs[r].subject == s]) for s in subj])
    sc = {k: [res[r]["auc"] for r in res if recs[r].scenario == k] for k in (1, 2)}
    return dict(mean=float(a.mean()), median=float(np.median(a)), n=len(a), subj_mean=float(sa.mean()),
                subj_above=int((sa > 0.5).sum()), n_subj=len(sa),
                sc1_mean=float(np.mean(sc[1])) if sc[1] else np.nan, sc2_mean=float(np.mean(sc[2])) if sc[2] else np.nan)


# ------------------------------------------------------------------------------------------ ablations
class LazyCovs(dict):
    """{rid: dict(meta=..., covs=<loaded on access>)}: covariances are read from the cache one recording
    at a time (each is ~22 MB) instead of holding all 28 in memory."""

    def __init__(self, recs):
        super().__init__()
        self.recs = {r.rec_id: r for r in recs}
        for r in recs:
            self[r.rec_id] = _LazyW(r)


class _LazyW(dict):
    def __init__(self, rec):
        super().__init__()
        self.rec = rec
        self["meta"] = build_windows(rec, win_s=WIN)["meta"]

    def __getitem__(self, k):
        if k == "covs":
            return build_windows(self.rec, win_s=WIN)["covs"]
        return super().__getitem__(k)


def load_meta_covs(recs):
    return LazyCovs(recs)


def blink_gates(recs, W, shuffled=False, seed=0):
    """True = window free of blinks. shuffled: each recording's mask circularly shifted (control)."""
    BL = blinks_cache()
    rng = np.random.default_rng(seed)
    out = {}
    for r in recs:
        g = ~blink_windows(W[r.rec_id]["meta"], np.asarray(BL[r.rec_id]["t"]))
        if shuffled:
            g = np.roll(g, int(rng.integers(60, len(g) - 60)))
        out[r.rec_id] = g
    return out


def closure_gates(recs, W, shuffled=False, seed=0, D_half=4.6):
    """True = window without a 'possible eye-closure' frame: occipital alpha >= recording median + D_half dB
    (half the eyes-closed minus eyes-open difference measured in the valid scenario-1 baselines) with no
    blink within +-3 s. Label-free."""
    from .confounds import blink_mask_frames
    Z = np.load(C.RESULTS_DIR / "confounds_frames.npz")
    BL = blinks_cache()
    rng = np.random.default_rng(seed)
    out = {}
    for r in recs:
        t, a = Z[f"{r.rec_id}|t"], Z[f"{r.rec_id}|alpha_occ"]
        clo = (a >= np.nanmedian(a) + D_half) & ~blink_mask_frames(t, np.asarray(BL[r.rec_id]["t"]), half=3.0)
        tc = t[clo]
        meta = W[r.rec_id]["meta"]
        ts, te = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
        g = ~(np.searchsorted(tc, te - 0.5, side="right") > np.searchsorted(tc, ts + 0.5))
        if shuffled:
            g = np.roll(g, int(rng.integers(60, len(g) - 60)))
        out[r.rec_id] = g
    return out


def build(name, recs, W, seed=0):
    """Returns (features dict, eval kwargs)."""
    base_ch = list(C.CHANNELS)
    kw = {}
    if name in ("base", "blink_score", "blink_gate", "blink_gate_shuffled"):
        F = {rid: eeg_features(W[rid]["covs"], base_ch, PO) for rid in W}
    elif name in DROP:
        keep = [c for c in base_ch if c not in DROP[name]]
        F = {rid: eeg_features(W[rid]["covs"], keep, PO) for rid in W}
    elif name.startswith("set_"):
        chs = SETS[name[4:]]
        F = {rid: eeg_features(W[rid]["covs"], chs, chs) for rid in W}
    elif name in BANDSETS:
        lb_ch, ts_ch, lb_b, ts_b = BANDSETS[name]
        F = {rid: eeg_features(W[rid]["covs"], lb_ch, ts_ch, logbp_bands=lb_b, ts_bands=ts_b) for rid in W}
    elif name in ("closure_gate", "closure_gate_shuffled"):
        F = {rid: eeg_features(W[rid]["covs"], base_ch, PO) for rid in W}
    elif name in ("blink_gate_strict", "blink_gate_strict_shuffled"):
        G = blink_gates(recs, W, shuffled=name.endswith("shuffled"), seed=seed)
        F = {rid: eeg_features(W[rid]["covs"], base_ch, PO, upd=G[rid]) for rid in W}
    elif name.startswith("proxy_") or name in ("base_plus_proxy", "base_resid"):
        P = proxy_features_all(recs, W)
        if name == "proxy_ocular":
            F = {rid: P[rid][:, :len(OCULAR)].astype(np.float64) for rid in P}
        elif name == "proxy_muscle":
            F = {rid: P[rid][:, len(OCULAR):].astype(np.float64) for rid in P}
        elif name == "proxy_all":
            F = {rid: P[rid].astype(np.float64) for rid in P}
        else:
            E = {rid: eeg_features(W[rid]["covs"], base_ch, PO) for rid in W}
            if name == "base_plus_proxy":
                F = {rid: np.concatenate([E[rid], P[rid]], axis=1) for rid in W}
            else:
                F = {}
                for rid in W:
                    A = np.c_[np.ones(len(P[rid])), P[rid].astype(np.float64)]
                    coef, *_ = np.linalg.lstsq(A, E[rid], rcond=None)
                    F[rid] = E[rid] - A @ coef + A[:, :1] @ coef[:1]
    else:
        raise ValueError(name)
    if name == "blink_score":
        kw = dict(gate=blink_gates(recs, W), score_gate=True)
    elif name in ("blink_gate", "blink_gate_strict"):
        kw = dict(gate=blink_gates(recs, W), train_gate=True, score_gate=True, gated_ema=True)
    elif name in ("blink_gate_shuffled", "blink_gate_strict_shuffled"):
        kw = dict(gate=blink_gates(recs, W, shuffled=True, seed=seed), train_gate=True, score_gate=True,
                  gated_ema=True)
    elif name in ("closure_gate", "closure_gate_shuffled"):
        kw = dict(gate=closure_gates(recs, W, shuffled=name.endswith("shuffled"), seed=seed), train_gate=True,
                  score_gate=True, gated_ema=True)
    return F, kw


def run(name, n_null=0, seed=0):
    """name may end with '@2' to use 2 s windows instead of 4 s."""
    global WIN
    t0 = time.time()
    base_name, WIN = (name[:-2], 2.0) if name.endswith("@2") else (name, FINAL_CFG["win_s"])
    recs = load_recordings(verbose=False)
    W = load_meta_covs(recs)
    F, kw = build(base_name, recs, W, seed)
    B = Bundle(recs, {rid: dict(meta=W[rid]["meta"]) for rid in W}, F)
    if base_name == "blink_gate_shuffled":          # 5 different circular shifts of the blink masks
        vals, obs = [], None
        for sd in range(5):
            if sd:
                kw = dict(kw, gate=blink_gates(recs, W, shuffled=True, seed=sd))
            o = evaluate(B, **kw)
            vals.append(summarize(o, B.recs)["mean"])
            obs = obs or o
        print(f"{name:<22} shuffled-gate AUC per seed {np.round(vals, 3)} mean {np.mean(vals):.3f}", flush=True)
    del W
    obs = evaluate(B, **kw) if base_name != "blink_gate_shuffled" else obs
    s = summarize(obs, B.recs)
    if base_name == "blink_gate_shuffled":
        s["seed_means"] = [float(v) for v in vals]
    gate = kw.get("gate")
    if gate is not None:
        kr, kt = [], []
        for rid in gate:
            y, ok, _, _ = B.labels(rid, "task_rest")
            kr.append(gate[rid][ok & (y == 0)].mean()); kt.append(gate[rid][ok & (y == 1)].mean())
        s.update(kept_frac=float(np.mean([gate[r].mean() for r in gate])), kept_rest=float(np.mean(kr)),
                 kept_task=float(np.mean(kt)))
        print(f"{name:<22} windows kept: all {s['kept_frac']:.2f}, rest {s['kept_rest']:.2f}, task {s['kept_task']:.2f}")
    print(f"{name:<22} AUC mean {s['mean']:.3f} median {s['median']:.3f} (n={s['n']}) | scenario 1 {s['sc1_mean']:.3f}"
          f" scenario 2 {s['sc2_mean']:.3f} | subject-level "
          f"{s['subj_mean']:.3f} ({s['subj_above']}/{s['n_subj']} > 0.5) | dim {next(iter(F.values())).shape[1]} "
          f"| {time.time() - t0:.0f}s", flush=True)
    nulls = []
    if n_null:
        rng = np.random.default_rng(seed)
        for k in range(n_null):
            nulls.append(summarize(evaluate(B, shifts=random_shifts(B, rng), **kw), B.recs)["mean"])
        nv = np.array(nulls)
        s.update(null_mean=float(nv.mean()), null_std=float(nv.std()), n_null=len(nv),
                 p=float((1 + (nv >= s["mean"]).sum()) / (1 + len(nv))))
        print(f"{name:<22} null {nv.mean():.3f} +- {nv.std():.3f} (n={len(nv)}), p={s['p']:.3f} | {time.time() - t0:.0f}s",
              flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(name=name, summary=s, per_rec=obs, nulls=nulls, seed=seed,
                                time=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    return obs, s


def cmd_summary():
    rows = [json.loads(l) for l in open(OUT)]
    last = {}
    for r in rows:
        last[r["name"]] = r
    base = last.get("base")
    from scipy.stats import wilcoxon
    print(f"{'ablation':<22} {'AUC':>6} {'subj':>6} {'>0.5':>5} {'null':>13} {'p':>6} {'dAUC vs base':>13} {'p(paired)':>9}")
    for name, r in last.items():
        s = r["summary"]
        line = f"{name:<22} {s['mean']:6.3f} {s['subj_mean']:6.3f} {s['subj_above']:>2}/{s['n_subj']:<2}"
        line += (f" {s['null_mean']:.3f}+-{s['null_std']:.3f} {s['p']:6.3f}" if "null_mean" in s else " " * 20)
        if base and name != "base":
            common = sorted(set(r["per_rec"]) & set(base["per_rec"]))
            d = np.array([r["per_rec"][k]["auc"] - base["per_rec"][k]["auc"] for k in common])
            line += f" {d.mean():+13.3f} {wilcoxon(d).pvalue if len(d) > 5 else float('nan'):9.3f}"
        print(line)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for nm in sys.argv[2].split(","):
            run(nm, int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else:
        globals()[f"cmd_{sys.argv[1]}"]()
