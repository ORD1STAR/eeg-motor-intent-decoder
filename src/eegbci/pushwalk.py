"""Phase 2: is there a brain-only push != walk signature, once time / drift / eyes are controlled?

    python -m eegbci.pushwalk frames          # per-recording frame features (cached, results/pushwalk_frames.npz)
    python -m eegbci.pushwalk onsets          # test 1 (+ test 4): scenario-1 matched onsets
    python -m eegbci.pushwalk crossed         # test 2: scenario-2 crossed-order block test
    python -m eegbci.pushwalk decode [n_null] # test 3: decoding with time-only / eyes-only / drift-null controls
    python -m eegbci.pushwalk summary         # verdict table from the json outputs

Offline, descriptive analysis (zero-phase filters are fine: nothing here feeds an online decoder, and
every window keeps >= 1 s guard from block edges).

Brain-only features
  channels  Fp1 Fp2 F7 F8 excluded (ocular); bad channels (confounds_blinks.json) excluded from zone means
  zones     frontal F3 Fz F4 | c_left FC5 C3 CP5 | c_mid FC1 FC2 Cz CP1 CP2 | c_right FC6 C4 CP6 |
            parietal P3 Pz P4 P7 P8 | occipital O1 Oz O2 | temporal T7 T8 FT9 FT10
  bands     theta 4-8, alpha 8-13, beta1 13-20, beta2 20-30, gamma 30-45 Hz (dB, 1 s frames, 0.25 s step)
  blinks    frames within 0.25 s of a detected blink (confounds.blinks_cache) are masked (NaN)
  resid     label-free per recording: each feature regressed on the eye/EMG proxies (VEOG, HEOG x2, Fp theta,
            temporal/frontal EMG 30-45 and 55-70 Hz) over the analysed span; residual + mean kept
  strict    resid without the temporal zone and without gamma (6 zones x 4 bands)

A-priori motor-imagery feature (the one hypothesis tested without multiple-comparison correction):
  MI index = mean(c_left, c_right) - c_mid in alpha and in beta1. Hand imagery (push) -> lateral ERD, foot /
  walking imagery -> midline ERD, so MI index(push) - MI index(walk) < 0 if a motor signature exists.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from . import config as C
from .windows import label_windows

RES = C.RESULTS_DIR
FRAMES = RES / "pushwalk_frames.npz"
CH = {c: i for i, c in enumerate(C.CHANNELS)}
OCULAR = ["Fp1", "Fp2", "F7", "F8"]
ZONES = {
    "frontal": ["F3", "Fz", "F4"],
    "c_left": ["FC5", "C3", "CP5"],
    "c_mid": ["FC1", "FC2", "Cz", "CP1", "CP2"],
    "c_right": ["FC6", "C4", "CP6"],
    "parietal": ["P3", "Pz", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
    "temporal": ["T7", "T8", "FT9", "FT10"],
}
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta1": (13, 20), "beta2": (20, 30), "gamma": (30, 45)}
FEATS = [f"{z}|{b}" for z in ZONES for b in BANDS]
STRICT = [i for i, f in enumerate(FEATS) if not f.startswith("temporal") and not f.endswith("gamma")]
PROXY = ["veog_lf", "heog_f", "heog_ft", "theta_fp", "emg30_temp", "emg55_temp", "emg30_front", "emg55_front"]
BLINK_HALF = 0.25
SEED = 0


# ============================================================================================ frames
def _frames_one(r, bads, blink_t):
    from .confounds import blink_mask_frames, proxies
    from .neuro import band_envelope
    x = r.data
    xuv = x * 1e6
    keep = [c for c in C.CHANNELS if c not in OCULAR and c not in bads]
    ix = [CH[c] for c in keep]
    cols = []
    t = None
    for b, (lo, hi) in BANDS.items():
        e, t = band_envelope(x[ix], lo, hi)           # (n_ch, n_frames) dB
        z = []
        for zn, chs in ZONES.items():
            zi = [keep.index(c) for c in chs if c in keep]
            z.append(e[zi].mean(0))
        cols.append(np.stack(z, 1))                    # (n_frames, n_zones)
    F = np.stack(cols, 2).reshape(len(t), -1)          # (n, zones, bands) -> FEATS order (zone outer, band inner)
    tp, P = proxies(xuv, bads)
    assert np.allclose(tp, t)
    Pm = np.stack([P[k] for k in PROXY], 1)
    blink = blink_mask_frames(t, blink_t, half=BLINK_HALF)
    r.release()
    return r.rec_id, t.astype(np.float32), F.astype(np.float32), Pm.astype(np.float32), blink


def cmd_frames():
    from .confounds import blinks_cache
    from .data import load_recordings
    recs = load_recordings(verbose=True)
    BL = blinks_cache()
    t0 = time.time()
    out = Parallel(n_jobs=4)(delayed(_frames_one)(r, BL[r.rec_id]["bads"], np.asarray(BL[r.rec_id]["t"]))
                             for r in recs)
    d = {}
    for rid, t, F, P, bl in out:
        d[f"{rid}|t"], d[f"{rid}|F"], d[f"{rid}|P"], d[f"{rid}|blink"] = t, F, P, bl
        print(f"{rid}: {len(t)} frames, blink-masked {bl.mean():.2f}", flush=True)
    np.savez_compressed(FRAMES, **d)
    print(f"saved {FRAMES} in {time.time() - t0:.0f}s")


# ============================================================================================ loading
class Rec:
    def __init__(self, r, z, blink_t):
        self.rid, self.subject, self.scenario = r.rec_id, r.subject, r.scenario
        self.blocks, self.duration = r.blocks, r.duration
        self.t = z[f"{r.rec_id}|t"].astype(np.float64)
        self.raw = z[f"{r.rec_id}|F"].astype(np.float64)
        self.P = z[f"{r.rec_id}|P"].astype(np.float64)
        self.blink = z[f"{r.rec_id}|blink"]
        self.blink_t = blink_t
        self.lo = 120.0 if r.scenario == 1 else 0.0     # analysed span starts after the eyes-closed baseline
        self.raw[self.blink] = np.nan
        self.resid = self._residualize()

    def _residualize(self):
        """Label-free: regress every feature on the proxies over blink-free frames of the analysed span."""
        cols = np.isfinite(self.P).mean(0) > 0.9                     # drop proxies of bad channels
        ok = (~self.blink) & (self.t >= self.lo) & np.all(np.isfinite(self.P[:, cols]), 1)
        P = self.P[:, cols]
        Pc = np.nan_to_num(P - np.nanmean(P[ok], 0))
        good_cols = np.nanstd(P[ok], 0) > 1e-9
        A = np.c_[np.ones(len(self.t)), Pc[:, good_cols]]
        coef, *_ = np.linalg.lstsq(A[ok], self.raw[ok], rcond=None)
        out = self.raw - A[:, 1:] @ coef[1:]
        out[self.blink] = np.nan
        return out

    def feats(self, kind):
        if kind == "raw":
            return self.raw
        if kind == "resid":
            return self.resid
        if kind == "strict":
            return self.resid[:, STRICT]
        raise ValueError(kind)


def load(kinds=None):
    from .confounds import blinks_cache
    from .data import load_recordings
    recs = load_recordings(verbose=False)
    BL = blinks_cache()
    z = np.load(FRAMES)
    return [Rec(r, z, np.asarray(BL[r.rec_id]["t"])) for r in recs]


def feat_names(kind):
    return [FEATS[i] for i in STRICT] if kind == "strict" else list(FEATS)


def mi_index(X, names):
    """A-priori MI index per band: mean(c_left, c_right) - c_mid (alpha, beta1)."""
    out = {}
    for b in ("alpha", "beta1"):
        if f"c_left|{b}" in names:
            out[f"MI|{b}"] = 0.5 * (X[..., names.index(f"c_left|{b}")] + X[..., names.index(f"c_right|{b}")]) \
                - X[..., names.index(f"c_mid|{b}")]
    return out


def seg_mean(rec, X, a, b):
    m = (rec.t - 0.5 >= a) & (rec.t + 0.5 <= b)
    return np.nanmean(X[m], 0) if m.any() else np.full(X.shape[1], np.nan)


def subj_mean(values, subjects):
    """values (n_rec, ...) -> (n_subj, ...) mean over each subject's recordings."""
    subs = sorted(set(subjects))
    s = np.asarray(subjects)
    return np.stack([np.nanmean(values[s == q], 0) for q in subs]), subs


def tstat(v):
    v = np.asarray(v, float)
    n = np.sum(np.isfinite(v), 0)
    m, sd = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)
    return m / (sd / np.sqrt(n) + 1e-12)


def signflip_maxt(V, n_perm=5000, seed=SEED):
    """V (n_units, n_feat). Global test: max |t| over features vs sign-flip distribution (exact if 2^n small).
    Returns (t per feature, FWER-corrected p per feature, uncorrected sign-flip p per feature)."""
    V = np.asarray(V, float)
    n = V.shape[0]
    t_obs = tstat(V)
    if 2 ** n <= n_perm:
        flips = np.array(list(itertools.product([1, -1], repeat=n)), float)
    else:
        flips = np.random.default_rng(seed).choice([1.0, -1.0], size=(n_perm, n))
    T = np.stack([tstat(V * f[:, None]) for f in flips])            # (n_flips, n_feat)
    mx = np.nanmax(np.abs(T), 1)
    p_fwer = np.array([(np.sum(mx >= abs(t)) ) / len(mx) for t in t_obs])
    p_unc = np.array([(np.sum(np.abs(T[:, j]) >= abs(t_obs[j]))) / len(T) for j in range(V.shape[1])])
    return t_obs, p_fwer, p_unc


def mde(sd, n, alpha=0.05, power=0.8, n_tests=1):
    """Minimal detectable mean for a two-sided one-sample t-test."""
    a = alpha / n_tests
    df = max(n - 1, 1)
    return (stats.t.ppf(1 - a / 2, df) + stats.t.ppf(power, df)) * sd / np.sqrt(n)


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    json.dump(obj, open(RES / name, "w"), indent=1, default=conv)


# ============================================================================================ test 1 + 4
ONSETS = {"P1": 240.0, "P2": 290.0, "W1": 340.0, "W2": 390.0}
PRE = (-10.0, -1.0)
POSTS = {"early": (1.0, 10.0), "late": (10.0, 29.0), "full": (1.0, 29.0)}


def onset_responses(rec, X, post):
    out = {}
    for k, t0 in ONSETS.items():
        out[k] = seg_mean(rec, X, t0 + post[0], t0 + post[1]) - seg_mean(rec, X, t0 + PRE[0], t0 + PRE[1])
    return out


def cmd_onsets():
    recs = [r for r in load() if r.scenario == 1]
    subjects = [r.subject for r in recs]
    res = dict(n_rec=len(recs), subjects=sorted(set(subjects)), analyses={})
    tt = np.array([0.0, 1.0, 2.0, 3.0])                               # onset order (50 s steps)
    D = np.c_[np.ones(4), tt, [0, 0, 1, 1]]
    pinv = np.linalg.pinv(D)                                          # row 2 = walk - push, linear time removed
    for kind in ("raw", "resid", "strict"):
        names = feat_names(kind)
        for pname, post in POSTS.items():
            R = {k: [] for k in ONSETS}
            for r in recs:
                o = onset_responses(r, r.feats(kind), post)
                for k in ONSETS:
                    R[k].append(o[k])
            R = {k: np.array(v) for k, v in R.items()}                 # (n_rec, n_feat)
            # add a-priori MI index as extra columns
            mi_names = []
            for k in R:
                mi = mi_index(R[k], names)
                mi_names = list(mi)
                R[k] = np.c_[R[k], np.stack([mi[m] for m in mi_names], 1)] if mi_names else R[k]
            allnames = names + mi_names
            contr = {
                "W-P naive (gap 100s)": 0.5 * (R["W1"] + R["W2"]) - 0.5 * (R["P1"] + R["P2"]),
                "W1-P2 (gap 50s, different class)": R["W1"] - R["P2"],
                "P2-P1 (gap 50s, same class)": R["P2"] - R["P1"],
                "W2-W1 (gap 50s, same class)": R["W2"] - R["W1"],
                "W-P linear-time-controlled": np.einsum("k,kij->ij", pinv[2], np.stack([R[k] for k in ONSETS])),
            }
            block = {}
            nf = len(names)
            for cname, V in contr.items():
                Vs, subs = subj_mean(V, subjects)
                t_r, pf_r, pu_r = signflip_maxt(V[:, :nf])
                t_s, pf_s, pu_s = signflip_maxt(Vs[:, :nf])
                # a-priori MI index (no correction), recording and subject level
                mi = {}
                for j, m in enumerate(mi_names):
                    v, vs = V[:, nf + j], Vs[:, nf + j]
                    mi[m] = dict(mean_db=float(np.nanmean(v)), t_rec=float(tstat(v)),
                                 p_rec=float(stats.ttest_1samp(v, 0, nan_policy="omit").pvalue),
                                 t_subj=float(tstat(vs)), p_subj=float(stats.ttest_1samp(vs, 0, nan_policy="omit").pvalue),
                                 subj_neg=int(np.sum(vs < 0)), n_subj=len(vs))
                best = int(np.nanargmax(np.abs(t_s)))
                block[cname] = dict(
                    rms_t_rec=float(np.sqrt(np.nanmean(t_r ** 2))), rms_t_subj=float(np.sqrt(np.nanmean(t_s ** 2))),
                    mean_abs_db=float(np.nanmean(np.abs(np.nanmean(V[:, :nf], 0)))),
                    n_sig_unc_rec=int(np.sum(pu_r < 0.05)), n_sig_fwer_rec=int(np.sum(pf_r < 0.05)),
                    n_sig_fwer_subj=int(np.sum(pf_s < 0.05)), min_pfwer_subj=float(np.min(pf_s)),
                    min_pfwer_rec=float(np.min(pf_r)),
                    mde_subj_db_median=float(np.nanmedian(mde(np.nanstd(Vs[:, :nf], 0, ddof=1), Vs.shape[0]))),
                    mde_rec_db_median=float(np.nanmedian(mde(np.nanstd(V[:, :nf], 0, ddof=1), V.shape[0]))),
                    best_subj=dict(feat=names[best], t=float(t_s[best]), mean_db=float(np.nanmean(Vs[:, best])),
                                   p_fwer=float(pf_s[best]), same_sign=int(np.sum(np.sign(Vs[:, best]) == np.sign(t_s[best])))),
                    per_feat=dict(names=names, mean_db=np.nanmean(V[:, :nf], 0), t_rec=t_r, p_fwer_rec=pf_r,
                                  t_subj=t_s, p_fwer_subj=pf_s),
                    MI=mi)
            # rms-t comparison: is the different-class 50 s contrast larger than the same-class 50 s contrasts?
            # permutation: per recording, randomly swap which onset of the pair is which (sign flip) - per contrast
            # -> compare the observed rms-t to the same-class ones directly.
            res["analyses"][f"{kind}|{pname}"] = block
            print(f"\n=== onsets {kind} {pname} (n_rec={len(recs)}, n_subj={len(set(subjects))})")
            for cname, b in block.items():
                mi_s = " ".join(f"{m}: {d['mean_db']:+.2f}dB t_s={d['t_subj']:+.2f} p_s={d['p_subj']:.3f}"
                                for m, d in b["MI"].items())
                print(f"  {cname:<36} |mean| {b['mean_abs_db']:.2f} dB  rms-t rec {b['rms_t_rec']:.2f} subj "
                      f"{b['rms_t_subj']:.2f} MDE subj {b['mde_subj_db_median']:.2f} dB rec {b['mde_rec_db_median']:.2f}  #p<.05unc(rec) {b['n_sig_unc_rec']:2d}/{nf}  FWER rec "
                      f"{b['n_sig_fwer_rec']} subj {b['n_sig_fwer_subj']} (min p {b['min_pfwer_subj']:.3f}) "
                      f"best {b['best_subj']['feat']} {b['best_subj']['mean_db']:+.2f}dB | {mi_s}")
    # ---------------- positive control: the rest->task response itself (known posterior beta increase)
    pc = {}
    for kind in ("strict",):
        names = feat_names(kind)
        V = np.array([np.nanmean(np.stack(list(onset_responses(r, r.feats(kind), POSTS["full"]).values())), 0)
                      for r in recs])
        Vs, _ = subj_mean(V, subjects)
        t_s, pf_s, _ = signflip_maxt(Vs)
        t_r, pf_r, _ = signflip_maxt(V)
        pc[kind] = dict(names=names, mean_db=np.nanmean(V, 0), t_rec=t_r, p_fwer_rec=pf_r, t_subj=t_s, p_fwer_subj=pf_s)
        top = np.argsort(-np.abs(t_r))[:5]
        print("  POSITIVE CONTROL rest->task onset response (" + kind + "): " +
              ", ".join(f"{names[i]} {pc[kind]['mean_db'][i]:+.2f}dB t_rec={t_r[i]:+.2f} pFWER_rec={pf_r[i]:.3f} "
                        f"pFWER_subj={pf_s[i]:.3f}" for i in top))
    res["positive_control"] = pc
    # ---------------- test 4: multivariate pair discrimination of onset responses (LOSO, early window)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pairs = {"P1 vs P2 (same class, 50s)": ("P1", "P2"), "W1 vs W2 (same class, 50s)": ("W1", "W2"),
             "P2 vs W1 (diff class, 50s)": ("P2", "W1"), "P1 vs W1 (diff class, 100s)": ("P1", "W1"),
             "P2 vs W2 (diff class, 100s)": ("P2", "W2")}
    t4 = {}
    for kind in ("resid", "strict"):
        for pname in ("early", "full"):
            Rr = [onset_responses(r, r.feats(kind), POSTS[pname]) for r in recs]
            for lab, (a, b) in pairs.items():
                Xa = np.array([o[a] for o in Rr]); Xb = np.array([o[b] for o in Rr])
                ok = np.all(np.isfinite(Xa), 1) & np.all(np.isfinite(Xb), 1)
                subs = np.array(subjects)
                correct = []
                for s in sorted(set(subs[ok])):
                    tr, te = ok & (subs != s), ok & (subs == s)
                    X = np.r_[Xa[tr], Xb[tr]]; y = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, y)
                    pa, pb = m.predict_proba(Xa[te])[:, 1], m.predict_proba(Xb[te])[:, 1]
                    correct += list((pb > pa).astype(float))
                acc = float(np.mean(correct)); n = len(correct)
                t4[f"{kind}|{pname}|{lab}"] = dict(pair_acc=acc, n=n,
                                                   p_binom=float(stats.binomtest(int(round(acc * n)), n, 0.5).pvalue))
                print(f"  T4 {kind:<6} {pname:<5} {lab:<30} pairwise acc {acc:.2f} (n={n}) "
                      f"p={t4[f'{kind}|{pname}|{lab}']['p_binom']:.3f}")
    res["test4_pair_decoding"] = t4
    # ---------------- time-resolved curves (strict, MI index + c_mid beta1), mean over recordings
    curves = {}
    rel = np.arange(-10, 29.01, 1.0)
    for kind in ("strict",):
        names = feat_names(kind)
        for k, t0 in ONSETS.items():
            arr = []
            for r in recs:
                X = r.feats(kind)
                base = seg_mean(r, X, t0 + PRE[0], t0 + PRE[1])
                arr.append([seg_mean(r, X, t0 + u - 0.5, t0 + u + 1.5) - base for u in rel])
            arr = np.array(arr)                                         # (n_rec, n_rel, n_feat)
            mi = mi_index(arr, names)
            curves[k] = {f: np.nanmean(arr[:, :, names.index(f)], 0) for f in
                         ("c_mid|beta1", "c_left|alpha", "c_right|alpha", "occipital|alpha", "parietal|beta1")}
            curves[k].update({m: np.nanmean(v, 0) for m, v in mi.items()})
    res["curves"] = dict(rel=rel, data=curves)
    # MDE for the linear-time-controlled contrast (strict, full) at subject level
    jdump(res, "pushwalk_onsets.json")


# ============================================================================================ test 2
S2_BLOCKS = {"p1": (0, 80), "w1": (80, 160), "p2": (180, 260), "w2": (260, 340), "p3": (360, 430), "w3": (430, 500)}
PF = [("p1", "w1"), ("p2", "w2"), ("p3", "w3")]
WF = [("p2", "w1"), ("p3", "w2")]           # (push, walk) with walk EARLIER
GUARD2 = 2.0


def block_values(rec, X, blocks):
    return {k: seg_mean(rec, X, a + GUARD2, min(b, rec.duration) - GUARD2) for k, (a, b) in blocks.items()}


def crossed_stats(Bv, order):
    """Bv: dict block -> (n_rec, n_feat); order: dict mapping real block names -> labels (perm).
    Returns d_PF, d_WF (push - walk), D_lin (OLS push coefficient with linear time)."""
    dPF = np.mean([Bv[p] - Bv[w] for p, w in PF], 0)
    dWF = np.mean([Bv[p] - Bv[w] for p, w in WF], 0)
    return dPF, dWF


def lin_delta(Bv, push_set, t_c):
    names = list(S2_BLOCKS)
    Y = np.stack([Bv[k] for k in names])                       # (6, n_rec, n_feat)
    x = np.array([1.0 if k in push_set else 0.0 for k in names])
    D = np.c_[np.ones(6), t_c, x]
    return np.einsum("k,kij->ij", np.linalg.pinv(D)[2], Y)     # push - walk


def cmd_crossed():
    recs = [r for r in load() if r.scenario == 2]
    subjects = np.array([r.subject for r in recs])
    t_c = np.array([(a + b) / 2 for a, b in S2_BLOCKS.values()]) / 100.0
    res = dict(n_rec=len(recs), subjects=sorted(set(subjects)), analyses={})
    rng = np.random.default_rng(SEED)
    combos = [set(c) for c in itertools.combinations(list(S2_BLOCKS), 3)]
    for kind in ("raw", "resid", "strict"):
        names = feat_names(kind)
        Bv = {}
        for r in recs:
            bv = block_values(r, r.feats(kind), S2_BLOCKS)
            for k, v in bv.items():
                mi = mi_index(v, names)
                Bv.setdefault(k, []).append(np.r_[v, [mi[m] for m in mi]])
        Bv = {k: np.array(v) for k, v in Bv.items()}
        mi_names = list(mi_index(np.zeros(len(names)), names))
        allnames = names + mi_names
        nf = len(names)
        dPF, dWF = crossed_stats(Bv, None)
        S, O = 0.5 * (dPF + dWF), 0.5 * (dPF - dWF)                  # order-invariant / order effect
        dL = lin_delta(Bv, {"p1", "p2", "p3"}, t_c)
        # sign consistency on BLOCK-DISJOINT pair sets (the full dPF and dWF share blocks: under pure block noise
        # corr(dPF, dWF) = +0.82, which would fake consistency). Sets: {(p1,w1) vs (w2,p3)}, {(p3,w3) vs (w1,p2)}
        DJ = [(("p1", "w1"), ("p3", "w2")), (("p3", "w3"), ("p2", "w1"))]
        djPF = np.stack([Bv[a[0]] - Bv[a[1]] for a, b in DJ]); djWF = np.stack([Bv[b[0]] - Bv[b[1]] for a, b in DJ])
        cons_rec_feat = np.nanmean(np.sign(djPF) * np.sign(djWF), 0)   # (n_rec, n_feat): +1 same sign, -1 flips
        cons = np.nanmean(cons_rec_feat, 0)
        cons_shared = np.nanmean(np.sign(dPF) * np.sign(dWF), 0)      # biased version, reported for reference
        r_dj = float(np.nanmean([stats.spearmanr(djPF[c][:, j], djWF[c][:, j])[0] for c in range(2) for j in range(nf)]))
        Ss, subs = subj_mean(S, subjects); Os, _ = subj_mean(O, subjects); dLs, _ = subj_mean(dL, subjects)
        dPFs, _ = subj_mean(dPF, subjects); dWFs, _ = subj_mean(dWF, subjects)
        djPFs = np.stack([subj_mean(v, subjects)[0] for v in djPF]); djWFs = np.stack([subj_mean(v, subjects)[0] for v in djWF])
        cons_s = np.nanmean(np.sign(djPFs) * np.sign(djWFs), (0, 1))
        # (a) linear-time D: permutation over block labels (independent random 3-of-6 'push' set per recording)
        # (b) order-invariant S: per-recording sign flips (= swapping push/walk labels of a whole recording)
        # (c) sign consistency: per-recording sign flips of dWF only (independence of the two orders); drift
        #     predicts consistency < 0, noise 0, a real order-invariant signature > 0
        n_perm = 2000
        Y = np.stack([Bv[k] for k in S2_BLOCKS])                       # (6, n_rec, n_feat_all)
        keys = list(S2_BLOCKS)
        perm_max_L, perm_mi_L = [], []
        for _ in range(n_perm):
            choice = [combos[i] for i in rng.integers(0, len(combos), len(recs))]
            Lp = []
            for ri, ps in enumerate(choice):
                x = np.array([1.0 if k in ps else 0.0 for k in keys])
                Lp.append(np.linalg.pinv(np.c_[np.ones(6), t_c, x])[2] @ Y[:, ri, :])
            Lps, _ = subj_mean(np.array(Lp), subjects)
            tl = tstat(Lps)
            perm_max_L.append(np.nanmax(np.abs(tl[:nf]))); perm_mi_L.append(tl[nf:])
        perm_max_L, perm_mi_L = np.array(perm_max_L), np.array(perm_mi_L)
        flips = rng.choice([1.0, -1.0], size=(n_perm, len(recs)))
        perm_max_S, perm_mi_S, perm_cons = [], [], []
        for f in flips:
            Sps, _ = subj_mean(S * f[:, None], subjects)
            ts_ = tstat(Sps)
            perm_max_S.append(np.nanmax(np.abs(ts_[:nf]))); perm_mi_S.append(ts_[nf:])
            perm_cons.append(np.nanmean((cons_rec_feat * f[:, None])[:, :nf]))
        perm_max_S, perm_mi_S, perm_cons = np.array(perm_max_S), np.array(perm_mi_S), np.array(perm_cons)
        perm_mi = np.c_[perm_mi_S, perm_mi_L]
        tS, tL, tO = tstat(Ss), tstat(dLs), tstat(Os)
        pS = np.array([np.mean(perm_max_S >= abs(v)) for v in tS[:nf]])
        pL = np.array([np.mean(perm_max_L >= abs(v)) for v in tL[:nf]])
        c_obs = float(np.nanmean(cons[:nf]))
        p_cons = float(np.mean(perm_cons >= c_obs))
        # drift signature: order effect O dominates S
        _, pO_fwer, _ = signflip_maxt(Os[:, :nf])
        _, pS_sf, _ = signflip_maxt(Ss[:, :nf])
        # MI index
        mi = {}
        for j, m in enumerate(mi_names):
            k = nf + j
            pm = perm_mi[:, [j, len(mi_names) + j]]
            mi[m] = dict(dPF_mean=float(np.nanmean(dPF[:, k])), dWF_mean=float(np.nanmean(dWF[:, k])),
                         S_mean=float(np.nanmean(S[:, k])), lin_mean=float(np.nanmean(dL[:, k])),
                         t_S_subj=float(tS[k]), t_lin_subj=float(tL[k]),
                         p_S_perm=float(np.mean(np.abs(pm[:, 0]) >= abs(tS[k]))),
                         p_lin_perm=float(np.mean(np.abs(pm[:, 1]) >= abs(tL[k]))),
                         subj_S=Ss[:, k], subj_dPF=dPFs[:, k], subj_dWF=dWFs[:, k],
                         cons_rec=float(cons[k]))
        # MDE for S (order-invariant push-walk) per feature at subject and recording level
        sd_s = np.nanstd(Ss[:, :nf], 0, ddof=1); sd_r = np.nanstd(S[:, :nf], 0, ddof=1)
        mde_s = mde(sd_s, len(subs)); mde_r = mde(sd_r, len(recs))
        mde_r_bonf = mde(sd_r, len(recs), n_tests=nf)
        out = dict(names=names, dPF=np.nanmean(dPF[:, :nf], 0), dWF=np.nanmean(dWF[:, :nf], 0),
                   S=np.nanmean(S[:, :nf], 0), O=np.nanmean(O[:, :nf], 0), lin=np.nanmean(dL[:, :nf], 0),
                   t_S_subj=tS[:nf], t_O_subj=tO[:nf], t_lin_subj=tL[:nf], p_S_permFWER=pS, p_lin_permFWER=pL,
                   p_O_signflipFWER=pO_fwer, p_S_signflipFWER=pS_sf,
                   cons_rec=cons[:nf], cons_subj=cons_s[:nf], cons_mean=c_obs, cons_shared_biased=float(np.nanmean(cons_shared[:nf])),
                   mean_spearman_disjoint_PF_WF_across_recs=r_dj, cons_null_mean=float(np.nanmean(perm_cons)),
                   cons_null_sd=float(np.nanstd(perm_cons)), p_cons_perm=p_cons,
                   frac_flip_rec=float(np.mean(cons[:nf] < 0)),
                   corr_dPF_dWF_across_feats=float(stats.spearmanr(np.nanmean(dPF[:, :nf], 0), np.nanmean(dWF[:, :nf], 0))[0]),
                   mde_S_subj_db_median=float(np.nanmedian(mde_s)), mde_S_rec_db_median=float(np.nanmedian(mde_r)),
                   mde_S_rec_bonf_db_median=float(np.nanmedian(mde_r_bonf)),
                   sd_S_rec_median=float(np.nanmedian(sd_r)), MI=mi, n_perm=n_perm)
        res["analyses"][kind] = out
        print(f"\n=== crossed-order {kind}: n_rec={len(recs)}, n_subj={len(subs)}")
        print(f"  mean |push-walk| push-first {np.nanmean(np.abs(out['dPF'])):.3f} dB, walk-first "
              f"{np.nanmean(np.abs(out['dWF'])):.3f} dB; spearman(dPF, dWF) across features "
              f"{out['corr_dPF_dWF_across_feats']:+.2f}")
        print(f"  [shared-block consistency (biased by construction) {out['cons_shared_biased']:+.3f}]; disjoint-pair "
              f"PF-vs-WF across-recording spearman (mean over feats) {r_dj:+.3f}; subject-level consistency "
              f"{np.nanmean(cons_s[:nf]):+.3f}")
        print(f"  DISJOINT sign consistency (rec) mean {c_obs:+.3f} (perm null {out['cons_null_mean']:+.3f} +- "
              f"{out['cons_null_sd']:.3f}, p={p_cons:.3f}); features flipping sign: {out['frac_flip_rec']:.2f}")
        print(f"  order-invariant S: max|t_subj| {np.nanmax(np.abs(tS[:nf])):.2f}, min p_FWER(perm) {pS.min():.3f}"
              f" | linear-time D: max|t| {np.nanmax(np.abs(tL[:nf])):.2f}, min p_FWER {pL.min():.3f}"
              f" | order effect O: max|t| {np.nanmax(np.abs(tO[:nf])):.2f}, min p_FWER {pO_fwer.min():.3f}")
        top = np.argsort(-np.abs(tS[:nf]))[:5]
        print("  top S features: " + ", ".join(f"{names[i]} S={out['S'][i]:+.2f}dB (PF {out['dPF'][i]:+.2f}, "
                                              f"WF {out['dWF'][i]:+.2f}) t={tS[i]:+.2f} p={pS[i]:.3f}" for i in top))
        topO = np.argsort(-np.abs(tO[:nf]))[:3]
        print("  top order-effect features: " + ", ".join(f"{names[i]} PF {out['dPF'][i]:+.2f} WF {out['dWF'][i]:+.2f}"
                                                         f" t_O={tO[i]:+.2f}" for i in topO))
        for m, d in mi.items():
            print(f"  {m}: PF {d['dPF_mean']:+.3f} WF {d['dWF_mean']:+.3f} S {d['S_mean']:+.3f} dB t_S={d['t_S_subj']:+.2f}"
                  f" p_perm={d['p_S_perm']:.3f} | lin {d['lin_mean']:+.3f} p={d['p_lin_perm']:.3f} | subj S "
                  f"{np.round(d['subj_S'], 2)}")
        print(f"  MDE (80% power, a=.05) of S: subject-level median {out['mde_S_subj_db_median']:.2f} dB, "
              f"recording-level {out['mde_S_rec_db_median']:.2f} dB (Bonferroni {out['mde_S_rec_bonf_db_median']:.2f}), "
              f"median SD {out['sd_S_rec_median']:.2f} dB")
    jdump(res, "pushwalk_crossed.json")


# ============================================================================================ test 3
DEC_WIN, DEC_STEP = 2.0, 1.0


def window_table(rec, kind):
    """2 s windows every 1 s: nan-mean of the blink-free 1 s frames inside; label-free per-recording centring
    (median over the analysed span). Also returns eye proxies (same windows) and blink count in the 10 s before."""
    X = rec.feats(kind)
    ends = np.arange(rec.lo + DEC_WIN, rec.duration + 1e-9, DEC_STEP)
    F, P, n_ok = [], [], []
    for te in ends:
        m = (rec.t - 0.5 >= te - DEC_WIN - 1e-9) & (rec.t + 0.5 <= te + 1e-9)
        good = m & ~rec.blink
        n_ok.append(good.sum())
        F.append(np.nanmean(X[good], 0) if good.sum() >= 2 else np.full(X.shape[1], np.nan))
        P.append(np.nanmean(rec.P[m], 0))
    F, P = np.array(F), np.array(P)
    F = F - np.nanmedian(F, 0)
    P = np.nan_to_num(P - np.nanmedian(P, 0))
    br = (np.searchsorted(rec.blink_t, ends) - np.searchsorted(rec.blink_t, ends - 10.0)) * 6.0   # /min
    E = np.c_[P, br - np.median(br)]
    meta = pd.DataFrame(dict(t_start=ends - DEC_WIN, t_end=ends))
    return F, E, meta, np.array(n_ok)


class DecData:
    def __init__(self, recs, kind):
        self.recs = recs
        self.F, self.E, self.meta, self.ok = {}, {}, {}, {}
        for r in recs:
            F, E, meta, n_ok = window_table(r, kind)
            self.F[r.rid], self.E[r.rid], self.meta[r.rid] = F, E, meta
            self.ok[r.rid] = np.all(np.isfinite(F), 1)

    def labels(self, r, shift=0.0):
        lab, blk, cyc, tib = label_windows(self.meta[r.rid], r.blocks, r.duration, shift, guard=1.0)
        return lab, blk, tib

    def X(self, r, model, shift=0.0):
        lab, blk, tib = self.labels(r, shift)
        if model == "eeg":
            return self.F[r.rid]
        if model == "eyes":
            return self.E[r.rid]
        if model == "eeg+eyes":
            return np.c_[self.F[r.rid], self.E[r.rid]]
        if model == "time":
            t = self.meta[r.rid]["t_end"].to_numpy() / 100.0
            return np.c_[t, t ** 2, np.nan_to_num(tib) / 10.0]
        raise ValueError(model)


def _fit(X, y, C_reg=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_reg, class_weight="balanced", max_iter=3000)).fit(X, y)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def _bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (p > 0.5).astype(int))) if len(np.unique(y)) == 2 else np.nan


def score_rec(r, y, p, blk, shift):
    """AUC (windows), balanced accuracy, block-level AUC, and crossed-pair AUCs (scenario 2, true protocol only)."""
    out = dict(auc=_auc(y, p), bacc=_bacc(y, p))
    bl = np.unique(blk)
    bm = np.array([p[blk == b].mean() for b in bl]); by = np.array([y[blk == b][0] for b in bl])
    out["auc_block"] = _auc(by, bm)
    if r.scenario == 2 and shift == 0.0:
        names = [b[3] for b in r.blocks]
        idx = {n.replace("#", "").replace("push", "p").replace("walk", "w"): i for i, n in enumerate(names)}
        def pair_auc(pairs):
            v = []
            for a, b in pairs:
                m = (blk == idx[a]) | (blk == idx[b])
                v.append(_auc(y[m], p[m]))
            return float(np.nanmean(v))
        out["auc_PF"] = pair_auc(PF)
        out["auc_WF"] = pair_auc(WF)
    return out


def eval_loso(D, model, shifts=None, train_filter=None, C_reg=0.1):
    """Leave-one-subject-out push vs walk. shifts: {rid: s}. train_filter(rec) -> bool restricts training recs."""
    sh = lambda r: 0.0 if shifts is None else shifts[r.rid]
    cache = {}
    for r in D.recs:
        lab, blk, tib = D.labels(r, sh(r))
        X = D.X(r, model, sh(r))
        m = ((lab == C.PUSH) | (lab == C.WALK)) & D.ok[r.rid]
        cache[r.rid] = (X[m], (lab[m] == C.WALK).astype(int), blk[m])
    res = {}
    for s in sorted({r.subject for r in D.recs}):
        tr = [r for r in D.recs if r.subject != s and (train_filter is None or train_filter(r))]
        te = [r for r in D.recs if r.subject == s]
        mdl = _fit(np.concatenate([cache[r.rid][0] for r in tr]), np.concatenate([cache[r.rid][1] for r in tr]), C_reg)
        for r in te:
            X, y, blk = cache[r.rid]
            if len(np.unique(y)) < 2:
                continue
            res[r.rid] = score_rec(r, y, mdl.predict_proba(X)[:, 1], blk, sh(r))
    return res


def eval_lobo(D, model, shifts=None, C_reg=0.1):
    """Within-subject leave-one-block-out: train on every other push/walk block of the same subject (all its
    recordings), predict the held-out block. Scored per recording."""
    sh = lambda r: 0.0 if shifts is None else shifts[r.rid]
    cache = {}
    for r in D.recs:
        lab, blk, tib = D.labels(r, sh(r))
        X = D.X(r, model, sh(r))
        m = ((lab == C.PUSH) | (lab == C.WALK)) & D.ok[r.rid]
        cache[r.rid] = (X[m], (lab[m] == C.WALK).astype(int), blk[m])
    res = {}
    for r in D.recs:
        X, y, blk = cache[r.rid]
        if len(np.unique(y)) < 2:
            continue
        others = [q for q in D.recs if q.subject == r.subject and q.rid != r.rid]
        Xo = [cache[q.rid][0] for q in others]; yo = [cache[q.rid][1] for q in others]
        p = np.full(len(y), np.nan)
        for b in np.unique(blk):
            tr = blk != b
            Xt = np.concatenate([X[tr]] + Xo); yt = np.concatenate([y[tr]] + yo)
            if len(np.unique(yt)) < 2:
                continue
            p[blk == b] = _fit(Xt, yt, C_reg).predict_proba(X[blk == b])[:, 1]
        ok = np.isfinite(p)
        res[r.rid] = score_rec(r, y[ok], p[ok], blk[ok], sh(r))
    return res


def shifts_for(recs, rng, min_s=60.0):
    return {r.rid: float(rng.uniform(min_s, (r.duration - r.lo) - min_s)) for r in recs}


def summ(res, recs, key="auc", scen=None):
    rmap = {r.rid: r for r in recs}
    v = {k: d[key] for k, d in res.items() if key in d and np.isfinite(d[key]) and (scen is None or rmap[k].scenario == scen)}
    if not v:
        return dict(mean=np.nan, n=0)
    subs = sorted({rmap[k].subject for k in v})
    sv = np.array([np.mean([v[k] for k in v if rmap[k].subject == s]) for s in subs])
    rng = np.random.default_rng(1)
    boot = [np.mean(sv[rng.integers(0, len(sv), len(sv))]) for _ in range(5000)]
    return dict(mean=float(np.mean(list(v.values()))), subj_mean=float(sv.mean()),
                ci95_subj=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                subj_above=int(np.sum(sv > 0.5)), n_subj=len(sv), n=len(v), subj=dict(zip(subs, sv.tolist())))


def _null_one(D, model, k, evaluator, n_seed):
    rng = np.random.default_rng(1000 + k + 100000 * n_seed)
    sh = shifts_for(D.recs, rng)
    res = evaluator(D, model, shifts=sh)
    return {sc: summ(res, D.recs, "auc", sc)["mean"] for sc in (None, 1, 2)}


def cmd_decode(n_null=30):
    recs = load()
    out = dict(n_null=n_null, results={})
    t0 = time.time()
    for kind in ("strict", "resid", "raw"):
        D = DecData(recs, kind)
        models = ["eeg", "time", "eyes"] if kind == "strict" else ["eeg"]
        if kind == "strict":
            models.append("eeg+eyes")
        for model in models:
            for ev_name, evaluator in (("loso", eval_loso), ("lobo", eval_lobo)):
                key = f"{kind}|{model}|{ev_name}"
                obs = evaluator(D, model)
                s = {sc: summ(obs, recs, "auc", sc) for sc in (None, 1, 2)}
                sb = summ(obs, recs, "bacc")
                sblk = summ(obs, recs, "auc_block")
                spf, swf = summ(obs, recs, "auc_PF", 2), summ(obs, recs, "auc_WF", 2)
                nulls = []
                if n_null and (model != "eeg+eyes"):
                    nulls = Parallel(n_jobs=4)(delayed(_null_one)(D, model, k, evaluator, 0) for k in range(n_null))
                entry = dict(all=s[None], sc1=s[1], sc2=s[2], bacc=sb, block=sblk, PF=spf, WF=swf, per_rec=obs)
                if nulls:
                    for sc, nm in ((None, "all"), (1, "sc1"), (2, "sc2")):
                        nv = np.array([n[sc] for n in nulls])
                        entry[f"null_{nm}"] = dict(mean=float(np.nanmean(nv)), sd=float(np.nanstd(nv)),
                                                   p=float((1 + np.sum(nv >= s[sc]["mean"])) / (1 + len(nv))),
                                                   mde95=float(np.nanmean(nv) + 2.49 * np.nanstd(nv)))
                out["results"][key] = entry
                nl = entry.get("null_all", {})
                print(f"{key:<26} AUC {s[None]['mean']:.3f} subj {s[None]['subj_mean']:.3f} CI "
                      f"[{s[None]['ci95_subj'][0]:.3f},{s[None]['ci95_subj'][1]:.3f}] ({s[None]['subj_above']}/{s[None]['n_subj']}>.5)"
                      f" | sc1 {s[1]['mean']:.3f} sc2 {s[2]['mean']:.3f} | bacc {sb['mean']:.3f} | block {sblk['mean']:.3f}"
                      f" | sc2 pairs push-first {spf['mean']:.3f} walk-first {swf['mean']:.3f}"
                      + (f" | null {nl['mean']:.3f}+-{nl['sd']:.3f} p={nl['p']:.3f}" if nl else "")
                      + (f" (sc2 null {entry['null_sc2']['mean']:.3f}+-{entry['null_sc2']['sd']:.3f} p={entry['null_sc2']['p']:.3f})"
                         if "null_sc2" in entry else "") + f" | {time.time() - t0:.0f}s", flush=True)
        if kind == "strict":
            # train on scenario 1 only (all push before walk = maximally time-confounded), test on scenario-2
            # walk-first pairs of held-out subjects: drift predicts AUC < 0.5, a real signature > 0.5
            for model in ("eeg", "time", "eyes"):
                obs = eval_loso(D, model, train_filter=lambda r: r.scenario == 1)
                spf, swf = summ(obs, recs, "auc_PF", 2), summ(obs, recs, "auc_WF", 2)
                s2 = summ(obs, recs, "auc", 2)
                out["results"][f"strict|{model}|train_sc1_test_sc2"] = dict(sc2=s2, PF=spf, WF=swf, per_rec=obs)
                print(f"strict|{model}|train sc1 -> sc2: AUC {s2['mean']:.3f} | push-first pairs {spf['mean']:.3f} "
                      f"CI {spf.get('ci95_subj')} | walk-first pairs {swf['mean']:.3f} CI {swf.get('ci95_subj')}", flush=True)
            for model in ("eeg", "time", "eyes"):
                obs = eval_loso(D, model, train_filter=lambda r: r.scenario == 2)
                spf, swf = summ(obs, recs, "auc_PF", 2), summ(obs, recs, "auc_WF", 2)
                s1 = summ(obs, recs, "auc", 1)
                out["results"][f"strict|{model}|train_sc2"] = dict(sc1=s1, PF=spf, WF=swf, per_rec=obs)
                print(f"strict|{model}|train sc2 (LOSO): sc1 AUC {s1['mean']:.3f} | sc2 push-first {spf['mean']:.3f} "
                      f"walk-first {swf['mean']:.3f} CI {swf.get('ci95_subj')}", flush=True)
        jdump(out, "pushwalk_decode.json")
    jdump(out, "pushwalk_decode.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decode":
        cmd_decode(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        globals()[f"cmd_{cmd}"]()
