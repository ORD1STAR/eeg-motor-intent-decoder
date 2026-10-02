"""Is the brain switch (rest vs active) neural, ocular or muscular?

Descriptive, offline analyses (zero-phase filters are fine here; nothing feeds a decoder):

    python -m eegbci.confounds blinks      # detect blinks, validate (EC baseline, blink-locked averages, example traces)
    python -m eegbci.confounds physio      # proxies per recording: block rates, transition-locked changes, drift-null
    python -m eegbci.confounds spectra     # 1-70 Hz spectral profile of the switch (EMG check) + eyes-closed reference
    python -m eegbci.confounds coupling    # does the posterior effect follow the eye/muscle changes across recordings?

Proxies (no EOG/EMG channels exist, so they come from the EEG montage, reference "TP"):
    blinks        positive peaks of mean(Fp1, Fp2), 0.5-15 Hz, bilateral, frontal-dominant
    VEOG          0.5-4 Hz power of mean(Fp1, Fp2)          (blinks + vertical eye movements)
    HEOG          0.5-4 Hz power of F7-F8 and FT9-FT10        (horizontal eye movements)
    EMG           30-45 Hz and 55-70 Hz power: temporal (T7 T8 FT9 FT10), frontal (Fp1 Fp2 F7 F8),
                  posterior/neck (O1 Oz O2 P7 P8)
    eye closure   occipital alpha relative to each recording's own eyes-closed baseline (scenario 1)

Decoding ablations live in confounds_decode.py.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import spearmanr, wilcoxon

from . import config as C
from .data import load_recordings
from .neuro import band_envelope, transitions
from .windows import _params_key

FS = C.SFREQ
CH = {c: i for i, c in enumerate(C.CHANNELS)}
RES = C.RESULTS_DIR
PO = ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
BETA_TOP = ["P3", "P7", "P8", "O1", "CP5"]           # strongest beta channels in ersp.json
OCC = ["O1", "Oz", "O2"]
TEMPORAL = ["T7", "T8", "FT9", "FT10"]
FRONTAL_EMG = ["Fp1", "Fp2", "F7", "F8"]
NECK = ["O1", "Oz", "O2", "P7", "P8"]
PRE, POST = (-12.0, -2.0), (3.0, 15.0)              # same windows as neuro.py
STEP = 0.25                                          # envelope frame step (neuro.band_envelope)

# palette (dataviz reference instance)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"


# ----------------------------------------------------------------------------------------------- helpers
def bads_of(rec):
    """Bad channels found by the cache builder (interpolated there; excluded from proxies here)."""
    p = dict(reref="none", interp=True, win=4.0, step=C.STEP_S, bands=C.BANDS, raw_band=(1, 40), v=4)
    path = C.CACHE_DIR / f"{rec.rec_id}_{_params_key(p)}.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as z:
        return [str(b) for b in z["bads"]]


def good(chs, bads):
    return [c for c in chs if c not in bads]


def bp(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def block_label_at(rec, t, guard=C.GUARD_S, shift=0.0):
    """Protocol label at times t (frame centres or event times). -1 = unusable/edge, -2 = EC baseline,
    0 rest, 1 push, 2 walk. ``shift`` circularly shifts the protocol (drift-null)."""
    tt = (np.asarray(t, float) - shift) % rec.duration
    lab = np.full(len(tt), -1)
    for t0, t1, l, name, c in rec.blocks:
        m = (tt >= t0 + guard) & (tt < t1 - guard)
        lab[m] = -2 if l is None else l
    return lab


def all_transitions(rec):
    """neuro.transitions + the eyes-closed -> eyes-open change at 120 s in scenario 1 (positive control)."""
    out = list(transitions(rec))
    if rec.scenario == 1:
        out.append((120.0, "EC->EO", 0))
    return out


def wil(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 6 or np.allclose(v, 0):
        return float("nan")
    return float(wilcoxon(v).pvalue)


# ----------------------------------------------------------------------------------------------- blinks
BLINK_BAND = (0.5, 8.0)            # below alpha: 0.5-15 Hz let frontal alpha peaks through (S06)
K_SD, MIN_UV, FWHM_MS = 6.0, 30.0, (60.0, 500.0)
CENTRAL = ["C3", "Cz", "C4"]


def blink_fp_channels(F_fp, names):
    """Fp channels that actually carry the blinks: in some recordings one Fp electrode barely sees them
    (S06: Fp2 ~3x weaker than Fp1 on every blink). Keep channels with >= 50 % of the strongest
    channel's positive-tail amplitude (99.5th percentile minus median)."""
    a = np.percentile(F_fp, 99.5, axis=1) - np.median(F_fp, axis=1)
    return [n for n, ai in zip(names, a) if ai >= 0.5 * a.max()]


def detect_blinks(xuv, bads=(), k_sd=K_SD, min_uv=MIN_UV, fwhm_ms=FWHM_MS, return_signals=False):
    """Blinks = positive peaks of the VEOG proxy (mean of the blink-carrying Fp channels, 0.5-8 Hz) with
       * prominence >= max(k_sd * robust SD of the VEOG trace, min_uv)
       * full width at half prominence within fwhm_ms
       * bilateral when both Fp channels carry blinks: each has its own local prominence >= 35 % of the VEOG one
       * frontal-dominant (rejects global/electrode transients): deflection at C3/Cz/C4 < 50 % and at
         O1/Oz/O2 < 40 % of the Fp prominence
    xuv: (30, n) microvolts. Returns dict(t, amp, fwhm, sd, thr, fp)."""
    fp_all, cen, occ = good(["Fp1", "Fp2"], bads), good(CENTRAL, bads), good(OCC, bads)
    F_fp = bp(xuv[[CH[c] for c in fp_all]], *BLINK_BAND)
    fp = blink_fp_channels(F_fp, fp_all)
    F_fp = F_fp[[fp_all.index(c) for c in fp]]
    F_ref = bp(xuv[[CH[c] for c in cen + occ]], *BLINK_BAND)
    v = F_fp.mean(0)
    sd = 1.4826 * np.median(np.abs(v - np.median(v)))
    thr = max(k_sd * sd, min_uv)
    pk, pr = find_peaks(v, prominence=thr, distance=int(0.25 * FS), wlen=int(2 * FS), rel_height=0.5,
                        width=(fwhm_ms[0] / 1e3 * FS, fwhm_ms[1] / 1e3 * FS))
    prom = pr["prominences"]
    keep = np.ones(len(pk), bool)
    w = int(0.4 * FS)
    for j, p in enumerate(pk):
        a, b = max(0, p - w), min(len(v), p + w)
        if len(fp) > 1:
            for s in F_fp:
                if s[p] - max(s[a:p + 1].min(), s[p:b].min()) < 0.35 * prom[j]:
                    keep[j] = False
        dev = np.abs(F_ref[:, p] - np.median(F_ref[:, a:b], axis=1))
        if (len(cen) and dev[:len(cen)].max() > 0.5 * prom[j]) or (len(occ) and dev[len(cen):].max() > 0.4 * prom[j]):
            keep[j] = False
    out = dict(t=pk[keep] / FS, amp=prom[keep], fwhm=pr["widths"][keep] / FS * 1e3, sd=float(sd), thr=float(thr),
               n_rejected=int((~keep).sum()), fp=fp)
    if return_signals:
        out["veog"] = v
    return out


def blink_mask_frames(t_frames, blink_t, half=0.5, win=1.0):
    """True for envelope frames (centre t, length win) overlapping any blink (+-half s around its peak)."""
    if len(blink_t) == 0:
        return np.zeros(len(t_frames), bool)
    lo, hi = t_frames - win / 2 - half, t_frames + win / 2 + half
    i = np.searchsorted(blink_t, lo)
    j = np.searchsorted(blink_t, hi)
    return j > i


def blink_windows(meta, blink_t, half=0.25):
    """True for decoding windows [t_start, t_end] that contain (part of) a blink."""
    if len(blink_t) == 0:
        return np.zeros(len(meta), bool)
    ts, te = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    i = np.searchsorted(blink_t, ts - half)
    j = np.searchsorted(blink_t, te + half)
    return j > i


def blinks_cache(recs=None, force=False):
    """Detect blinks in every recording once; cached in results/confounds_blinks.json."""
    path = RES / "confounds_blinks.json"
    if path.exists() and not force:
        return {k: dict(v, t=np.array(v["t"]), amp=np.array(v["amp"]), fwhm=np.array(v["fwhm"]))
                for k, v in json.load(open(path)).items()}
    recs = recs or load_recordings(verbose=False)
    out = {}
    for r in recs:
        bads = bads_of(r)
        d = detect_blinks(r.data * 1e6, bads)
        out[r.rec_id] = dict(t=d["t"].tolist(), amp=d["amp"].tolist(), fwhm=d["fwhm"].tolist(), sd=d["sd"],
                             thr=d["thr"], n_rejected=d["n_rejected"], bads=bads, fp=d["fp"], scenario=r.scenario,
                             duration=r.duration)
        r.release()
    json.dump(out, open(path, "w"))
    return blinks_cache(force=False)


# ----------------------------------------------------------------------------------------------- proxies
def proxies(xuv, bads):
    """1-s frames every 0.25 s (neuro.band_envelope). Returns t and dict name -> (n_frames,) dB series."""
    def env(sig, lo, hi):
        e, t = band_envelope(np.atleast_2d(sig), lo, hi)
        return e, t

    def chan_mean(chs, lo, hi):
        chs = good(chs, bads)
        e, t = env(xuv[[CH[c] for c in chs]], lo, hi)
        return e.mean(0), t

    out = {}
    fp = good(["Fp1", "Fp2"], bads)
    veog = xuv[[CH[c] for c in fp]].mean(0)
    e, t = env(veog, 0.5, 4.0); out["veog_lf"] = e[0]
    if not ({"F7", "F8"} & set(bads)):
        out["heog_f"] = env(xuv[CH["F7"]] - xuv[CH["F8"]], 0.5, 4.0)[0][0]
    else:
        out["heog_f"] = np.full(len(t), np.nan)
    if not ({"FT9", "FT10"} & set(bads)):
        out["heog_ft"] = env(xuv[CH["FT9"]] - xuv[CH["FT10"]], 0.5, 4.0)[0][0]
    else:
        out["heog_ft"] = np.full(len(t), np.nan)
    out["theta_fp"] = chan_mean(["Fp1", "Fp2"], 4, 8)[0]
    for name, chs in (("temp", TEMPORAL), ("front", FRONTAL_EMG), ("neck", NECK)):
        chs_g = good(chs, bads)
        if chs_g:
            out[f"emg30_{name}"] = chan_mean(chs, 30, 45)[0]
            out[f"emg55_{name}"] = chan_mean(chs, 55, 70)[0]
        else:
            out[f"emg30_{name}"] = out[f"emg55_{name}"] = np.full(len(t), np.nan)
    out["alpha_po"] = chan_mean(PO, 8, 13)[0]
    out["beta_po"] = chan_mean(PO, 13, 30)[0]
    out["beta_top"] = chan_mean(BETA_TOP, 13, 30)[0]
    out["alpha_occ"] = chan_mean(OCC, 8, 13)[0]
    return t, out


def blink_rate_series(blink_t, t_frames, win=1.0):
    """Blinks per minute in each frame (count of peaks inside the frame / frame length)."""
    lo, hi = t_frames - win / 2, t_frames + win / 2
    return (np.searchsorted(blink_t, hi) - np.searchsorted(blink_t, lo)) / win * 60.0


def transition_deltas(t, series, trans, stat="mean"):
    """{kind: (n_events, ) post-minus-pre values} for one series (frames centred at t)."""
    out = {}
    f = np.nanmean if stat == "mean" else np.nanmedian
    for t0, kind, _ in trans:
        pre = (t >= t0 + PRE[0]) & (t < t0 + PRE[1])
        post = (t >= t0 + POST[0]) & (t <= t0 + POST[1])
        if pre.sum() < 10 or post.sum() < 10:
            continue
        out.setdefault(kind, []).append(float(f(series[post]) - f(series[pre])))
    return out


def blink_rate_delta(blink_t, trans):
    out = {}
    for t0, kind, _ in trans:
        pre = ((blink_t >= t0 + PRE[0]) & (blink_t < t0 + PRE[1])).sum() / (PRE[1] - PRE[0]) * 60
        post = ((blink_t >= t0 + POST[0]) & (blink_t <= t0 + POST[1])).sum() / (POST[1] - POST[0]) * 60
        out.setdefault(kind, []).append(float(post - pre))
    return out


# ----------------------------------------------------------------------------------------------- plots
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def cmd_blinks():
    """Detect blinks, then validation plots: example traces, blink-locked average per recording,
    and the eyes-closed baseline rate (blinks cannot happen with eyes closed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = load_recordings(verbose=False)
    B = blinks_cache(recs, force=True)
    rows = []
    # blink-locked averages + example traces
    fig_avg, axes = plt.subplots(4, 7, figsize=(14, 8), sharex=True)
    ex_t0 = {"S03-1": 108.0, "S06-1": 250.0, "S05-3": 40.0, "S07-4": 170.0, "S02-3": 170.0, "S09-2": 110.0}
    ex_ids = list(ex_t0)
    fig_ex, axe = plt.subplots(len(ex_ids), 1, figsize=(13, 2.6 * len(ex_ids)))
    for k, r in enumerate(recs):
        b = B[r.rec_id]
        x = r.data * 1e6
        F = bp(x, 0.5, 15.0)
        bt = b["t"]
        idx = (bt * FS).astype(int)
        idx = idx[(idx > FS) & (idx < x.shape[1] - FS)]
        lag = np.arange(-int(0.6 * FS), int(0.6 * FS))
        ax = axes.flat[k]
        for c, col in (("Fp1", S1), ("Fp2", S3), ("F8", S2), ("Cz", S4), ("O1", MUTED)):
            if len(idx):
                ax.plot(lag / FS, F[CH[c]][idx[:, None] + lag].mean(0), color=col, lw=1.2, label=c)
        ax.set_title(f"{r.rec_id}  n={len(bt)}", fontsize=8, color=INK)
        _style(ax)
        ec = (bt < 118) if r.scenario == 1 else np.zeros(len(bt), bool)
        t0_eo = 122 if r.scenario == 1 else 0
        lab = block_label_at(r, bt)
        rows.append(dict(rec=r.rec_id, scenario=r.scenario, n=len(bt), thr=b["thr"], sd=b["sd"],
                         rate_ec=float(ec.sum() / 116 * 60) if r.scenario == 1 else np.nan,
                         rate_eo_all=float(((bt >= t0_eo)).sum() / (r.duration - t0_eo) * 60),
                         amp_med=float(np.median(b["amp"])) if len(bt) else np.nan,
                         fwhm_med=float(np.median(b["fwhm"])) if len(bt) else np.nan))
        if r.rec_id in ex_ids:
            a = axe[ex_ids.index(r.rec_id)]
            t0 = ex_t0[r.rec_id]
            sl = slice(int(t0 * FS), int((t0 + 25) * FS))
            tt = np.arange(sl.start, sl.stop) / FS
            off = 0
            for c, col in (("Fp1", S1), ("Fp2", S3), ("F7", S2), ("F8", S4), ("O1", MUTED)):
                a.plot(tt, F[CH[c], sl] - off, color=col, lw=0.8, label=c)
                off += 120
            m = (bt >= t0) & (bt < t0 + 25)
            for tb in bt[m]:
                a.axvline(tb, color=INK, lw=0.6, ls=":")
            lab_seg = {-2: "eyes-closed baseline", 0: "rest", 1: "push", 2: "walk", -1: "edge"}
            segs = sorted({lab_seg[int(l)] for l in block_label_at(r, np.arange(t0, t0 + 25, 0.5))})
            a.set_title(f"{r.rec_id} (scenario {r.scenario}), {t0:.0f}-{t0 + 25:.0f} s [{', '.join(segs)}], "
                        f"0.5-15 Hz, 120 uV between traces, dotted = detected blink (VEOG from {'+'.join(b['fp'])})",
                        fontsize=9, color=INK)
            a.set_yticks([]); _style(a)
            if ex_ids.index(r.rec_id) == 0:
                a.legend(ncol=5, fontsize=7, frameon=False, loc="upper right")
        r.release()
    for ax in axes.flat[len(recs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig_avg.suptitle("Blink-locked average (0.5-15 Hz, microvolts) per recording", fontsize=10, color=INK)
    fig_avg.tight_layout()
    fig_avg.savefig(RES / "confounds_blink_avg.png", dpi=110, facecolor="white")
    fig_ex.tight_layout()
    fig_ex.savefig(RES / "confounds_blink_examples.png", dpi=110, facecolor="white")
    for row in rows:
        print(f"{row['rec']} sc{row['scenario']} n={row['n']:4d} thr {row['thr']:5.1f} uV  rate EO {row['rate_eo_all']:5.1f}/min"
              f"  EC {row['rate_ec']:5.1f}/min  amp {row['amp_med']:5.0f} uV  fwhm {row['fwhm_med']:4.0f} ms"
              f"  VEOG from {'+'.join(B[row['rec']]['fp'])}")
    sc1 = [r for r in rows if r["scenario"] == 1]
    print(f"scenario 1 (n={len(sc1)}): EC median {np.median([r['rate_ec'] for r in sc1]):.1f}/min vs EO "
          f"{np.median([r['rate_eo_all'] for r in sc1]):.1f}/min, Wilcoxon p={wil([r['rate_eo_all'] - r['rate_ec'] for r in sc1]):.4f}")
    json.dump(rows, open(RES / "confounds_blink_validation.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------- physio
SERIES = ["blink_rate", "veog_lf", "heog_f", "heog_ft", "theta_fp", "emg30_temp", "emg55_temp", "emg30_front",
          "emg55_front", "emg30_neck", "emg55_neck", "alpha_po", "beta_po", "beta_top", "alpha_occ"]
UNITS = {"blink_rate": "/min"}


def interior_duration(rec, label_set, guard=C.GUARD_S, shift=0.0):
    """Seconds of block interior with a label in label_set (labels as in block_label_at)."""
    tt = np.arange(0, rec.duration, 0.05)
    return float(np.isin(block_label_at(rec, tt, guard, shift), list(label_set)).sum() * 0.05)


def physio_one(r, BL):
    """Per-recording frame series + labels."""
    bads = bads_of(r)
    x = r.data * 1e6
    t, P = proxies(x, bads)
    r.release()
    bt = np.asarray(BL[r.rec_id]["t"])
    P["blink_rate"] = blink_rate_series(bt, t)
    P["blinkfree"] = ~blink_mask_frames(t, bt)
    return t, P, bt


def block_stats(r, t, P, bt, shift=0.0):
    """Mean of each series in rest / task / EC frames, and event-based blink rates."""
    lab = block_label_at(r, t, shift=shift)
    blab = block_label_at(r, bt, shift=shift)
    out = {}
    for k in SERIES:
        if k == "blink_rate":
            for name, ls in (("rest", (0,)), ("task", (1, 2)), ("ec", (-2,))):
                d = interior_duration(r, ls, shift=shift)
                out[f"{k}|{name}"] = float(np.isin(blab, ls).sum() / d * 60) if d > 0 else np.nan
            continue
        s = P[k]
        for name, m in (("rest", lab == 0), ("task", (lab == 1) | (lab == 2)), ("ec", lab == -2)):
            out[f"{k}|{name}"] = float(np.nanmean(s[m])) if m.any() else np.nan
    return out


def eye_closure_stats(r, t, P, bt, D_half):
    """Sparse eye-closure check. Frames whose occipital alpha exceeds the recording's rest median by
    D_half dB (half the typical eyes-closed minus eyes-open difference); 'possible closure' = such a
    frame with no blink within +-3 s. Long blink-free stretches = frames inside an inter-blink interval > 10 s."""
    lab = block_label_at(r, t)
    rest, task = lab == 0, (lab == 1) | (lab == 2)
    a = P["alpha_occ"]
    thr = np.nanmedian(a[rest]) + D_half
    high = a > thr
    near = blink_mask_frames(t, bt, half=3.0)
    closure = high & ~near
    # inter-blink interval containing each frame
    ev = np.r_[0.0, bt, r.duration]
    k = np.searchsorted(ev, t)
    gap = ev[np.clip(k, 1, len(ev) - 1)] - ev[np.clip(k - 1, 0, len(ev) - 2)]
    long_ibi = gap > 10.0
    q = {}
    for qq in (10, 25, 50, 75, 90):
        q[f"q{qq}"] = float(np.nanpercentile(a[task], qq) - np.nanpercentile(a[rest], qq))
    return {"alpha_high|rest": float(high[rest].mean()), "alpha_high|task": float(high[task].mean()),
            "closure|rest": float(closure[rest].mean()), "closure|task": float(closure[task].mean()),
            "long_ibi|rest": float(long_ibi[rest].mean()), "long_ibi|task": float(long_ibi[task].mean()), **q}


def cmd_physio(n_null=1000):
    """Block-level (with drift-null) and transition-locked changes of every proxy, eye-closure checks."""
    recs = load_recordings(verbose=False)
    BL = blinks_cache()
    data = {}
    for r in recs:
        t, P, bt = physio_one(r, BL)
        data[r.rec_id] = (t, P, bt)
        print(f"  {r.rec_id}", flush=True)
    np.savez_compressed(RES / "confounds_frames.npz", **{f"{rid}|{k}": v for rid, (t, P, bt) in data.items()
                                                       for k, v in dict(P, t=t).items()})
    R = {r.rec_id: r for r in recs}
    out = dict(per_rec={}, block={}, transition={}, eye_closure={}, ec_calibration={})

    # ---- eyes-closed calibration (scenario 1): is the 0-120 s baseline really eyes closed?
    D = []
    for rid, (t, P, bt) in data.items():
        r = R[rid]
        if r.scenario != 1:
            continue
        a = P["alpha_occ"]
        ec_a = float(np.nanmedian(a[(t > 5) & (t < 115)])); eo_a = float(np.nanmedian(a[(t > 125) & (t < 235)]))
        ec_rate = float(((bt > 5) & (bt < 115)).sum() / 110 * 60); eo_rate = float(((bt > 125) & (bt < 235)).sum() / 110 * 60)
        valid = (ec_a - eo_a >= 3.0) and (ec_rate <= 0.5 * eo_rate + 1)
        out["ec_calibration"][rid] = dict(alpha_ec_minus_eo=ec_a - eo_a, blink_ec=ec_rate, blink_eo=eo_rate, valid=valid)
        if valid:
            D.append(ec_a - eo_a)
    D_half = float(np.median(D) / 2)
    out["ec_calibration"]["_D_median_dB"] = float(np.median(D))
    print(f"eyes-closed baseline valid in {len(D)}/12 scenario-1 recordings; occipital alpha EC-EO median "
          f"{np.median(D):+.1f} dB", flush=True)

    # ---- block level
    for rid, (t, P, bt) in data.items():
        out["per_rec"][rid] = dict(subject=R[rid].subject, scenario=R[rid].scenario,
                                   **block_stats(R[rid], t, P, bt), **eye_closure_stats(R[rid], t, P, bt, D_half))
    rng = np.random.default_rng(0)
    null = {k: [] for k in SERIES}
    for _ in range(n_null):
        st = {rid: block_stats(R[rid], t, P, bt, shift=float(rng.uniform(60, R[rid].duration - 60)))
              for rid, (t, P, bt) in data.items()}
        for k in SERIES:
            null[k].append(np.nanmean([s[f"{k}|task"] - s[f"{k}|rest"] for s in st.values()]))
        if _ == 199 and n_null > 200:
            print("  null 200 done", flush=True)
    print(f"\nBLOCK LEVEL task - rest (n=28 recordings; Wilcoxon; drift-null = circular protocol shift x{n_null})")
    for k in SERIES:
        d = np.array([v[f"{k}|task"] - v[f"{k}|rest"] for v in out["per_rec"].values()])
        rest = np.array([v[f"{k}|rest"] for v in out["per_rec"].values()])
        task = np.array([v[f"{k}|task"] for v in out["per_rec"].values()])
        nv = np.array(null[k])
        p_null = float((1 + (np.abs(nv - nv.mean()) >= abs(np.nanmean(d) - nv.mean())).sum()) / (1 + len(nv)))
        out["block"][k] = dict(rest_median=float(np.nanmedian(rest)), task_median=float(np.nanmedian(task)),
                               diff_mean=float(np.nanmean(d)), diff_median=float(np.nanmedian(d)), p_wilcoxon=wil(d),
                               null_mean=float(nv.mean()), null_std=float(nv.std()), p_null=p_null,
                               n_pos=int((d > 0).sum()), n=int(np.isfinite(d).sum()))
        b = out["block"][k]
        print(f"  {k:<12} rest {b['rest_median']:7.2f} task {b['task_median']:7.2f} {UNITS.get(k, 'dB')} | diff mean "
              f"{b['diff_mean']:+.3f} median {b['diff_median']:+.3f} ({b['n_pos']}/{b['n']} >0) p={b['p_wilcoxon']:.4f}"
              f" | null {b['null_mean']:+.3f}+-{b['null_std']:.3f} p_null={b['p_null']:.4f}")
    for sc in (1, 2):
        for k in ("blink_rate",):
            v = [x for x in out["per_rec"].values() if x["scenario"] == sc]
            d = np.array([x[f"{k}|task"] - x[f"{k}|rest"] for x in v])
            print(f"  scenario {sc} blink rate: rest {np.median([x[f'{k}|rest'] for x in v]):.1f} task "
                  f"{np.median([x[f'{k}|task'] for x in v]):.1f}/min, diff median {np.median(d):+.1f} "
                  f"({(d > 0).sum()}/{len(d)} >0) p={wil(d):.4f}")
            out["block"][f"{k}|scenario{sc}"] = dict(rest_median=float(np.median([x[f'{k}|rest'] for x in v])),
                                                    task_median=float(np.median([x[f'{k}|task'] for x in v])),
                                                    diff_median=float(np.median(d)), p_wilcoxon=wil(d), n=len(d))

    # ---- transition-locked (local baseline; drift cancels in the onset/offset contrast)
    kinds = ("rest->task", "task->rest", "push->walk", "EC->EO")
    per = {rid: {} for rid in data}
    for rid, (t, P, bt) in data.items():
        tr = all_transitions(R[rid])
        for k in SERIES:
            d = blink_rate_delta(bt, tr) if k == "blink_rate" else transition_deltas(t, P[k], tr)
            for kind, v in d.items():
                per[rid][f"{k}|{kind}"] = float(np.mean(v))
        for k in ("alpha_po", "beta_po", "beta_top", "alpha_occ"):
            dm = transition_deltas(t, P[k], tr, stat="median")
            clean = np.where(P["blinkfree"], P[k], np.nan)
            dc = transition_deltas(t, clean, tr)
            for kind in dm:
                per[rid][f"{k}_median|{kind}"] = float(np.mean(dm[kind]))
                per[rid][f"{k}_blinkfree|{kind}"] = float(np.mean(dc[kind]))
        for k in list(per[rid]):
            pass
    out["transition_per_rec"] = per
    print("\nTRANSITION-LOCKED post(3-15 s) - pre(-12..-2 s); switch = (on - off)/2 (drift cancels)")
    keys = SERIES + [f"{k}_{v}" for k in ("alpha_po", "beta_po", "beta_top", "alpha_occ") for v in ("median", "blinkfree")]
    for k in keys:
        row = {}
        for kind in kinds:
            v = np.array([per[rid].get(f"{k}|{kind}", np.nan) for rid in data])
            v = v[np.isfinite(v)]
            row[kind] = dict(median=float(np.median(v)) if len(v) else np.nan, mean=float(np.mean(v)) if len(v) else np.nan,
                             p=wil(v), n=len(v))
        sw = np.array([(per[rid][f"{k}|rest->task"] - per[rid][f"{k}|task->rest"]) / 2 for rid in data])
        subj = sorted({R[r].subject for r in data})
        sws = np.array([np.mean([s for r_, s in zip(data, sw) if R[r_].subject == s_]) for s_ in subj])
        row["switch"] = dict(median=float(np.median(sw)), mean=float(np.mean(sw)), p=wil(sw), n=len(sw),
                             p_subject=wil(sws), n_pos=int((sw > 0).sum()))
        out["transition"][k] = row
        print(f"  {k:<20} on {row['rest->task']['median']:+6.2f} (p={row['rest->task']['p']:.3f})  off "
              f"{row['task->rest']['median']:+6.2f} (p={row['task->rest']['p']:.3f})  switch {row['switch']['median']:+6.2f} "
              f"(p={row['switch']['p']:.4f}, subj p={row['switch']['p_subject']:.3f}, {row['switch']['n_pos']}/28 >0)"
              f"  | EC->EO {row['EC->EO']['median']:+6.2f} (n={row['EC->EO']['n']})")

    # ---- eye closure summary
    print(f"\nEYE CLOSURE (alpha threshold = rest median + {D_half:.1f} dB)")
    for k in ("alpha_high", "closure", "long_ibi"):
        rr = np.array([v[f"{k}|rest"] for v in out["per_rec"].values()])
        tt_ = np.array([v[f"{k}|task"] for v in out["per_rec"].values()])
        out["eye_closure"][k] = dict(rest_median=float(np.median(rr)), task_median=float(np.median(tt_)),
                                     p=wil(tt_ - rr), n_pos=int((tt_ > rr).sum()))
        print(f"  {k:<10} fraction of frames: rest {np.median(rr):.3f} task {np.median(tt_):.3f} "
              f"({(tt_ > rr).sum()}/28 task>rest) p={wil(tt_ - rr):.4f}")
    qd = {q: np.array([v[q] for v in out["per_rec"].values()]) for q in ("q10", "q25", "q50", "q75", "q90")}
    out["eye_closure"]["alpha_occ_quantile_shift"] = {q: dict(median=float(np.median(v)), p=wil(v)) for q, v in qd.items()}
    print("  occipital alpha task-rest shift by quantile (dB): " +
          "  ".join(f"{q} {np.median(v):+.2f} (p={wil(v):.3f})" for q, v in qd.items()))
    json.dump(out, open(RES / "confounds_physio.json", "w"), indent=1)
    print("saved confounds_physio.json, confounds_frames.npz")


# ----------------------------------------------------------------------------------------------- spectra
def cmd_spectra():
    """Spectral profile (1-70 Hz, 1 Hz bins) of the transition effect on every channel. EMG would show an
    effect that grows with frequency and is largest above 30 Hz; a neural alpha/beta effect peaks below 30 Hz."""
    from scipy.signal import spectrogram
    recs = load_recordings(verbose=False)
    kinds = ("rest->task", "task->rest", "EC->EO")
    acc = {k: [] for k in kinds}
    rids = {k: [] for k in kinds}
    f = None
    for r in recs:
        x = r.data * 1e6
        f, tt, S = spectrogram(x, fs=FS, window="hann", nperseg=int(FS), noverlap=int(FS / 2), detrend="constant",
                               scaling="density", mode="psd")
        r.release()
        keep = (f >= 1) & (f <= 70)
        f = f[keep]
        L = 10 * np.log10(S[:, keep, :] + 1e-12)                    # (ch, f, frames) dB
        del S
        by = {}
        for t0, kind, _ in all_transitions(r):
            if kind not in kinds:
                continue
            pre = (tt >= t0 + PRE[0]) & (tt < t0 + PRE[1])
            post = (tt >= t0 + POST[0]) & (tt <= t0 + POST[1])
            if pre.sum() < 5 or post.sum() < 5:
                continue
            by.setdefault(kind, []).append(L[:, :, post].mean(2) - L[:, :, pre].mean(2))
        for kind, v in by.items():
            acc[kind].append(np.mean(v, axis=0)); rids[kind].append(r.rec_id)
        print(f"  {r.rec_id}", flush=True)
    np.savez_compressed(RES / "confounds_spectra.npz", f=f, **{k.replace("->", "_to_"): np.array(v) for k, v in acc.items()},
                        **{f"rids_{k.replace('->', '_to_')}": np.array(v) for k, v in rids.items()})
    spectra_report()


def spectra_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(RES / "confounds_spectra.npz")
    f = z["f"]
    on, off = z["rest_to_task"], z["task_to_rest"]
    sw = (on - off) / 2                                             # (n_rec, ch, f)
    ec = z["EC_to_EO"]
    rid_ec = list(z["rids_EC_to_EO"])
    cal = json.load(open(RES / "confounds_physio.json"))["ec_calibration"] if (RES / "confounds_physio.json").exists() else {}
    valid = [i for i, rid in enumerate(rid_ec) if cal.get(rid, {}).get("valid", True)]
    ec = -ec[valid]                                                  # eyes closing = - (EC->EO)
    groups = {"posterior beta channels (P3 P7 P8 O1 CP5)": BETA_TOP, "occipital (O1 Oz O2)": OCC,
              "temporal (T7 T8 FT9 FT10)": TEMPORAL, "frontopolar (Fp1 Fp2)": ["Fp1", "Fp2"]}
    bands = {"theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30), "gamma 30-45": (30, 45),
             "high 55-70": (55, 70)}
    res = {}
    print("SPECTRAL PROFILE of the switch effect (dB, median over 28 recordings; Wilcoxon p)")
    for gname, chs in groups.items():
        ix = [CH[c] for c in chs]
        g = sw[:, ix].mean(1)                                         # (n_rec, f)
        res[gname] = {}
        line = f"  {gname:<44}"
        for bname, (lo, hi) in bands.items():
            m = (f >= lo) & (f <= hi) & ~((f > 45) & (f < 55))
            v = g[:, m].mean(1)
            res[gname][bname] = dict(median=float(np.median(v)), p=wil(v))
            line += f" {bname} {np.median(v):+.2f} (p={wil(v):.3f}) |"
        print(line)
    # profile similarity with eyes closing
    ix = [CH[c] for c in BETA_TOP]
    m = (f >= 2) & (f <= 40)
    prof_sw = np.median(sw[:, ix].mean(1), axis=0)[m]
    prof_ec = np.median(ec[:, ix].mean(1), axis=0)[m]
    r_prof = float(np.corrcoef(prof_sw, prof_ec)[0, 1])
    a_m, b_m = (f >= 8) & (f <= 13), (f >= 13) & (f <= 30)
    ratio_sw = float(np.median(sw[:, ix].mean(1)[:, b_m].mean(1)) / np.median(sw[:, ix].mean(1)[:, a_m].mean(1)))
    ratio_ec = float(np.median(ec[:, ix].mean(1)[:, b_m].mean(1)) / np.median(ec[:, ix].mean(1)[:, a_m].mean(1)))
    res["profile_vs_eyes_closing"] = dict(r=r_prof, beta_over_alpha_switch=ratio_sw, beta_over_alpha_eyes_closing=ratio_ec,
                                          n_ec=len(valid))
    print(f"  posterior profile 2-40 Hz: correlation with eyes-closing profile r={r_prof:.2f}; beta/alpha effect ratio "
          f"switch {ratio_sw:.2f} vs eyes closing {ratio_ec:.2f} (n_EC={len(valid)})")
    json.dump(res, open(RES / "confounds_spectra.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    cols = [S1, S2, S3, S4]
    for ax, (title, arr, scale) in zip(axes, [("switch effect (rest->task minus task->rest)/2", sw, 1.0),
                                              ("rest -> task", on, 1.0),
                                              ("eyes closing (scenario 1 baseline, x0.1)", ec, 0.1)]):
        for (gname, chs), col in zip(groups.items(), cols):
            g = arr[:, [CH[c] for c in chs]].mean(1) * scale
            med = np.median(g, axis=0)
            q1, q3 = np.percentile(g, 25, axis=0), np.percentile(g, 75, axis=0)
            mm = (f < 45) | (f > 55)
            ff = np.where(mm, f, np.nan)
            ax.plot(ff, np.where(mm, med, np.nan), color=col, lw=1.6, label=gname)
            ax.fill_between(f, np.where(mm, q1, np.nan), np.where(mm, q3, np.nan), color=col, alpha=0.12, lw=0)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("frequency (Hz)", fontsize=9, color=INK2)
        _style(ax)
    axes[0].set_ylabel("change in power (dB), median and IQR over recordings", fontsize=9, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Muscle check: a muscle (EMG) effect would grow above 30 Hz; the switch effect is confined to alpha/beta",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "confounds_spectra.png", dpi=120, facecolor="white")
    print("saved confounds_spectra.png / .json")


# ----------------------------------------------------------------------------------------------- coupling
TARGETS = ["beta_top", "beta_po", "alpha_po", "alpha_occ"]
PROX = ["blink_rate", "veog_lf", "heog_f", "theta_fp", "emg30_temp", "emg55_temp", "emg30_neck", "emg55_neck",
        "emg30_front"]


def _dct_basis(t, period_s=240.0):
    T = t[-1] - t[0]
    K = int(np.floor(2 * T / period_s))
    u = (t - t[0]) / T
    return np.stack([np.cos(np.pi * k * u) for k in range(1, K + 1)], axis=1) if K else np.zeros((len(t), 0))


def cmd_coupling(n_boot=2000):
    """(1) across recordings: does the size of the posterior switch effect follow the change in blinks /
    EOG / EMG? Spearman + regression whose intercept = effect expected with NO proxy change (subject
    bootstrap CI). (2) within recordings: GLM of the posterior power time course on task + slow drift, with
    and without the eye/muscle proxies as covariates; does the task coefficient survive?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J = json.load(open(RES / "confounds_physio.json"))
    per = J["transition_per_rec"]
    rids = sorted(per)
    recs = {r.rec_id: r for r in load_recordings(verbose=False)}
    subj = np.array([recs[r].subject for r in rids])
    sw = lambda k: np.array([(per[r].get(f"{k}|rest->task", np.nan) - per[r].get(f"{k}|task->rest", np.nan)) / 2
                             for r in rids])
    on = lambda k: np.array([per[r].get(f"{k}|rest->task", np.nan) for r in rids])
    out = dict(spearman={}, intercept={}, glm={})
    print("ACROSS RECORDINGS (n=28): Spearman rho between the switch effect on posterior power and on each proxy")
    for tg in TARGETS:
        line = f"  {tg:<10}"
        for px in PROX:
            a, b = sw(tg), sw(px)
            m = np.isfinite(a) & np.isfinite(b)
            rho, p = spearmanr(a[m], b[m])
            out["spearman"][f"{tg}~{px}"] = dict(rho=float(rho), p=float(p), n=int(m.sum()))
            line += f" {px} {rho:+.2f}{'*' if p < 0.05 else ' '}"
        print(line)
    # intercept regression: effect at zero proxy change
    preds = ["blink_rate", "veog_lf", "heog_f", "emg30_temp", "emg55_temp", "emg55_neck"]
    rng = np.random.default_rng(0)
    usub = np.unique(subj)
    for tg in TARGETS:
        for mode, fn in (("switch", sw), ("onset", on)):
            y = fn(tg)
            X = np.stack([fn(p) for p in preds], axis=1)
            m = np.isfinite(y) & np.isfinite(X).all(1)
            y, X, s_ = y[m], X[m], subj[m]
            sdx = X.std(0) + 1e-12
            A = np.c_[np.ones(len(y)), X / sdx]
            b0 = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            boots = []
            for _ in range(n_boot):
                pick = np.concatenate([np.flatnonzero(s_ == u) for u in rng.choice(usub, len(usub))])
                if len(pick) < len(preds) + 3:
                    continue
                boots.append(np.linalg.lstsq(A[pick], y[pick], rcond=None)[0][0])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["intercept"][f"{tg}|{mode}"] = dict(raw_median=float(np.median(y)), b0=b0, ci=[float(lo), float(hi)],
                                                     p_boot=float(2 * min(np.mean(np.array(boots) <= 0), np.mean(np.array(boots) >= 0))))
            print(f"  {tg:<10} {mode:<7} raw median {np.median(y):+.2f} dB -> effect at zero proxy change b0 {b0:+.2f} dB "
                  f"[95% subject-bootstrap CI {lo:+.2f}, {hi:+.2f}]")
    # within-recording GLM
    Z = np.load(RES / "confounds_frames.npz")
    cov_sets = {"eyes": ["veog_lf", "heog_f", "heog_ft", "theta_fp"],
                "muscle": ["emg30_temp", "emg55_temp", "emg30_front", "emg55_front", "emg55_neck"],
                "all": ["veog_lf", "heog_f", "heog_ft", "theta_fp", "emg30_temp", "emg55_temp", "emg30_front",
                        "emg55_front", "emg55_neck"]}
    print("\nWITHIN RECORDINGS: GLM power(t) ~ task + slow drift (DCT, periods >= 240 s) [+ proxies]; task beta in dB")
    for tg in TARGETS:
        res = {k: [] for k in ("none", "eyes", "muscle", "all")}
        for rid in rids:
            t = Z[f"{rid}|t"]
            lab = block_label_at(recs[rid], t)
            m = lab >= 0
            y = Z[f"{rid}|{tg}"][m]
            base = np.c_[np.ones(m.sum()), (lab[m] > 0).astype(float), _dct_basis(t[m])]
            for name in res:
                cols = [] if name == "none" else [Z[f"{rid}|{c}"][m] for c in cov_sets[name]]
                cols = [c for c in cols if np.isfinite(c).all()]
                A = np.c_[base, np.stack(cols, 1)] if cols else base
                if cols:
                    A[:, base.shape[1]:] = (A[:, base.shape[1]:] - A[:, base.shape[1]:].mean(0)) / (A[:, base.shape[1]:].std(0) + 1e-12)
                ok = np.isfinite(y)
                res[name].append(float(np.linalg.lstsq(A[ok], y[ok], rcond=None)[0][1]))
        out["glm"][tg] = {k: dict(median=float(np.median(v)), mean=float(np.mean(v)), p=wil(v),
                                  n_pos=int((np.array(v) > 0).sum())) for k, v in res.items()}
        g = out["glm"][tg]
        print(f"  {tg:<10} " + " | ".join(f"{k}: {g[k]['median']:+.3f} (p={g[k]['p']:.4f}, {g[k]['n_pos']}/28>0)" for k in res)
              + f" | retained with all proxies {g['all']['median'] / g['none']['median']:.0%}")
    json.dump(out, open(RES / "confounds_coupling.json", "w"), indent=1)

    # figure: switch effect on posterior beta vs proxies
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, px, lab_ in zip(axes, ["blink_rate", "veog_lf", "emg55_neck", "emg30_temp"],
                            ["blink rate (/min)", "VEOG 0.5-4 Hz (dB)", "neck EMG 55-70 Hz (dB)", "temporal EMG 30-45 Hz (dB)"]):
        a, b = sw("beta_top"), sw(px)
        ax.scatter(b, a, s=22, color=S1, edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color="#c3c2b7", lw=0.8)
        r_ = out["spearman"][f"beta_top~{px}"]
        ax.set_title(f"rho = {r_['rho']:+.2f} (p = {r_['p']:.2f})", fontsize=9, color=INK)
        ax.set_xlabel(f"switch effect on {lab_}", fontsize=8, color=INK2)
        _style(ax)
    axes[0].set_ylabel("switch effect, posterior beta (dB)", fontsize=8, color=INK2)
    fig.suptitle("Across 28 recordings, the posterior beta switch does not scale with eye or muscle changes "
                 "(one dot = one recording)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "confounds_coupling.png", dpi=120, facecolor="white")
    print("saved confounds_coupling.json / .png")


def cmd_figures():
    """Blink-rate time course around transitions (all recordings)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = load_recordings(verbose=False)
    BL = blinks_cache()
    edges = np.arange(-12, 21, 2.0)
    curves = {k: [] for k in ("rest->task", "task->rest")}
    for r in recs:
        bt = np.asarray(BL[r.rec_id]["t"])
        by = {k: [] for k in curves}
        for t0, kind, _ in transitions(r):
            if kind in by and t0 + edges[0] >= 0 and t0 + edges[-1] <= r.duration:
                by[kind].append(np.histogram(bt - t0, bins=edges)[0] / np.diff(edges) * 60)
        for k, v in by.items():
            if v:
                curves[k].append(np.mean(v, 0))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    mid = (edges[:-1] + edges[1:]) / 2
    for (k, v), col in zip(curves.items(), (S1, S2)):
        v = np.array(v)
        m, se = v.mean(0), v.std(0) / np.sqrt(len(v))
        ax.plot(mid, m, color=col, lw=2, label=f"{k} (n={len(v)} recordings)")
        ax.fill_between(mid, m - se, m + se, color=col, alpha=0.15, lw=0)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("time from block change (s)", fontsize=9, color=INK2)
    ax.set_ylabel("blinks per minute (mean +- SEM)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax.set_title("Blink rate around block changes", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "confounds_blink_psth.png", dpi=120, facecolor="white")
    json.dump({k: dict(t=mid.tolist(), mean=np.mean(v, 0).tolist(), n=len(v)) for k, v in curves.items()},
              open(RES / "confounds_blink_psth.json", "w"))
    print("saved confounds_blink_psth.png")


if __name__ == "__main__":
    globals()[f"cmd_{sys.argv[1]}"]()
