"""Phase 2: onset-locked channel x frequency x time maps of brain power per action change.

    python -m eegbci.maps compute    # per recording TF power (raw / blink-masked / clean), onset-locked, local baseline
    python -m eegbci.maps stats      # cluster permutation tests (channels x freqs x time) + band summaries + eye control
    python -m eegbci.maps figures    # PNG figures
    python -m eegbci.maps all

Cleaning: eegbci.brainonly (channel exclusion, blink masks, label-free proxy regression).
Onset types (eegbci.neuro.transitions):
    rest_push  rest -> push   scenario 1 at 240, 290; scenario 2 at 180, 360
    rest_walk  rest -> walk   scenario 1 at 340, 390
    push_walk  push -> walk   scenario 2 at 80, 260, 430
    push_rest  push -> rest   scenario 1 at 270, 320               (offset control)
    walk_rest  walk -> rest   scenario 1 at 370, 420; scenario 2 at 160, 340   (offset control)
Each onset: values at relative times -10..+25 s (0.5 s grid; each value = mean of the 1-s multitaper frames
centred within +-0.5 s, i.e. data from tau-1 to tau+1 s), minus the mean of -10..-2 s per channel x frequency
(LOCAL baseline). Onsets are averaged within a recording, recordings within a subject (primary level).
Offsets are only tested up to +18 s (the following rest lasts 20 s).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from functools import partial

import mne
import numpy as np
from scipy import stats as st

from . import brainonly as B
from . import config as C
from .confounds import blinks_cache
from .data import load_recordings
from .neuro import transitions

RES = C.RESULTS_DIR
REL = np.round(np.arange(-10.0, 25.0 + 1e-9, 0.5), 2)
BASE = (-10.0, -2.0)
TYPES = ["rest_push", "rest_walk", "push_walk", "push_rest", "walk_rest"]
TYPE_LABEL = {"rest_push": "rest -> push", "rest_walk": "rest -> walk", "push_walk": "push -> walk",
              "push_rest": "push -> rest (offset)", "walk_rest": "walk -> rest (offset)",
              "rest_push_s1": "rest -> push (scen. 1)", "rest_push_s2": "rest -> push (scen. 2)",
              "rp_minus_rw_s1": "rest->push minus rest->walk (scen. 1)",
              "pr_minus_wr_s1": "push->rest minus walk->rest (scen. 1)"}
TMAX = {"rest_push": 25.0, "rest_walk": 25.0, "push_walk": 25.0, "push_rest": 18.0, "walk_rest": 18.0}
VERSIONS = ["raw", "masked", "clean"]
PROX = ["blink_rate", "veog", "heog_f", "heog_ft", "emg", "emg_neck"]
PROX_LABEL = {"blink_rate": "blink rate (/min)", "veog": "VEOG 0.5-4 Hz (dB)", "heog_f": "HEOG F7-F8 (dB)",
              "heog_ft": "HEOG FT9-FT10 (dB)", "emg": "EMG 30-45 Hz temporal (dB)",
              "emg_neck": "neck EMG 55-70 Hz (dB)"}
BANDS = {"theta": (4, 8), "alpha_mu": (8, 13), "low_beta": (13, 20), "high_beta": (20, 30), "gamma": (30, 45)}
TBINS = [(0.0, 5.0), (5.0, 15.0), (15.0, 25.0)]
GROUPS = {"frontal": ["Fz", "F3", "F4", "FC1", "FC2", "FC5", "FC6"], "central": ["C3", "Cz", "C4", "CP1", "CP2"],
          "parietal": ["P3", "Pz", "P4", "CP5", "CP6"], "occipital": ["O1", "Oz", "O2", "P7", "P8"],
          "temporal": ["T7", "T8", "FT9", "FT10"]}
NCH, NF, NT = len(B.MAP_CHANNELS), len(B.FREQS), len(REL)
N_JOBS = 4
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"


def onset_type(kind, lab):
    if kind == "rest->task":
        return "rest_push" if lab == C.PUSH else "rest_walk"
    if kind == "task->rest":
        return "push_rest" if lab == C.PUSH else "walk_rest"
    if kind == "push->walk":
        return "push_walk"
    return None


# ------------------------------------------------------------------------------------------------ compute
def _epoch(t, X, t0):
    """X (..., n_frames) -> (..., NT) mean of frames within +-0.5 s of t0 + REL; NaN where none / outside."""
    out = np.full(X.shape[:-1] + (NT,), np.nan, np.float32)
    for k, tau in enumerate(REL):
        sel = np.abs(t - (t0 + tau)) <= 0.5 + 1e-9
        if sel.sum() < 3:                                    # outside the recording
            continue
        with np.errstate(all="ignore"), _quiet():
            out[..., k] = np.nanmean(X[..., sel], axis=-1)
    return out


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _baseline(seg):
    b = (REL >= BASE[0]) & (REL <= BASE[1])
    with _quiet():
        return seg - np.nanmean(seg[..., b], axis=-1, keepdims=True)


def _fill_gaps(A, valid_t):
    """Linear interpolation along time of NaNs left by blink masking (only inside valid_t)."""
    flat = A.reshape(-1, NT)
    idx = np.flatnonzero(valid_t)
    n_filled = 0
    for row in flat:
        bad = ~np.isfinite(row[idx])
        if bad.any() and (~bad).sum() >= 2:
            n_filled += int(bad.sum())
            row[idx[bad]] = np.interp(idx[bad], idx[~bad], row[idx[~bad]])
    return A, n_filled


def cmd_compute():
    recs = load_recordings()
    blinks = blinks_cache()
    nR = len(recs)
    E = {v: np.full((nR, len(TYPES), C.CHANNELS.__len__(), NF, NT), np.nan, np.float32) for v in VERSIONS}
    PX = np.full((nR, len(TYPES), len(PROX), NT), np.nan, np.float32)
    n_on = np.zeros((nR, len(TYPES)), int)
    cov = np.full((nR, len(TYPES), NT), np.nan, np.float32)          # fraction of blink-free frames used
    r2 = np.zeros((nR, len(C.CHANNELS), NF), np.float32)
    filled = np.zeros(nR, int)
    blink_frac = np.zeros(nR)
    for i, r in enumerate(recs):
        d = B.brain_power(r, blinks)
        r.release()
        t = d["t"]
        r2[i] = d["reg_info"]["partial_r2"]
        blink_frac[i] = d["blink_frames"].mean()
        P = np.stack([d["proxies"][k] for k in PROX])
        segs = {}
        for t0, kind, lab in transitions(r):
            ty = onset_type(kind, lab)
            if ty is None:
                continue
            segs.setdefault(ty, []).append(t0)
        for j, ty in enumerate(TYPES):
            if ty not in segs:
                continue
            n_on[i, j] = len(segs[ty])
            for v in VERSIONS:
                ep = np.stack([_baseline(_epoch(t, d[v], t0)) for t0 in segs[ty]])
                with _quiet():
                    m = np.nanmean(ep, 0)
                if v == "raw":
                    valid_t = np.isfinite(m).all(axis=(0, 1))
                else:
                    m, nf = _fill_gaps(m, valid_t)
                    filled[i] += nf if v == "clean" else 0
                E[v][i, j] = m
            ok = (~d["blink_frames"]).astype(float)
            cov[i, j] = np.nanmean(np.stack([_epoch(t, ok[None], t0)[0] for t0 in segs[ty]]), 0)
            PX[i, j] = np.nanmean(np.stack([_baseline(_epoch(t, P, t0)) for t0 in segs[ty]]), 0)
        print(f"{r.rec_id} sc{r.scenario} onsets {dict(zip(TYPES, n_on[i]))} blink-masked {blink_frac[i]:.0%} "
              f"bads {d['bads']} proxies {d['reg_info']['proxies']} median partial R2 {np.median(r2[i]):.3f}", flush=True)
    np.savez_compressed(RES / "maps_recordings.npz", **{f"E_{v}": E[v] for v in VERSIONS}, PX=PX, n_onsets=n_on,
                        coverage=cov, partial_r2=r2, filled=filled, blink_frac=blink_frac, rel=REL, freqs=B.FREQS,
                        channels=np.array(C.CHANNELS), map_channels=np.array(B.MAP_CHANNELS), types=np.array(TYPES),
                        prox=np.array(PROX), rec_ids=np.array([r.rec_id for r in recs]),
                        subjects=np.array([r.subject for r in recs]), scenarios=np.array([r.scenario for r in recs]))
    print("saved maps_recordings.npz")


# ------------------------------------------------------------------------------------------------ data access
def load_rec():
    z = np.load(RES / "maps_recordings.npz")
    return {k: z[k] for k in z.files}


def select(Z, test, version="clean", level="subject"):
    """Return (X (n, ch26, f, t), PX (n, prox, t), unit ids, tmax) for a named test.
    test: one of TYPES, rest_push_s1, rest_push_s2, rp_minus_rw_s1, pr_minus_wr_s1."""
    E = Z[f"E_{version}"][:, :, B.MAP_IDX]
    PXa = Z["PX"]
    sc, subj, n_on = Z["scenarios"], Z["subjects"], Z["n_onsets"]
    ti = {t: k for k, t in enumerate(TYPES)}

    def pick(ty, scen=None):
        m = n_on[:, ti[ty]] > 0
        if scen:
            m &= sc == scen
        return m

    if test in ("rp_minus_rw_s1", "pr_minus_wr_s1"):
        a, b = ("rest_push", "rest_walk") if test == "rp_minus_rw_s1" else ("push_rest", "walk_rest")
        m = pick(a, 1) & pick(b, 1)
        X = E[m, ti[a]] - E[m, ti[b]]
        P = PXa[m, ti[a]] - PXa[m, ti[b]]
        tmax = min(TMAX[a], TMAX[b])
    else:
        ty, scen = (test[:-3], int(test[-1])) if test.endswith(("_s1", "_s2")) else (test, None)
        m = pick(ty, scen)
        X, P, tmax = E[m, ti[ty]], PXa[m, ti[ty]], TMAX[ty]
    ids = Z["rec_ids"][m]
    if level == "subject":
        s = subj[m]
        us = np.unique(s)
        with _quiet():
            X = np.stack([np.nanmean(X[s == u], 0) for u in us])
            P = np.stack([np.nanmean(P[s == u], 0) for u in us])
        ids = us
    return X, P, ids, tmax


# ------------------------------------------------------------------------------------------------ stats
def ch_adjacency():
    info = mne.create_info(B.MAP_CHANNELS, C.SFREQ, "eeg")
    info.set_montage("standard_1005")
    adj, names = mne.channels.find_ch_adjacency(info, "eeg")
    adj = adj.tolil()
    for a, b in (("FT9", "P7"), ("FT10", "P8")):            # Delaunay hull edges across the missing T/TP sites
        i, j = names.index(a), names.index(b)
        adj[i, j] = adj[j, i] = 0
    return adj.tocsr()


def cluster_test(X, tmax, n_perm=2048, p_form=0.05, seed=0):
    """X (n, ch, f, t). Tests 0..tmax s. Returns T map, list of clusters (dicts), cluster label map."""
    tsel = (REL >= 0) & (REL <= tmax + 1e-9)
    Y = np.transpose(X[..., tsel], (0, 3, 2, 1)).astype(float)     # (n, time, freq, ch)
    n = len(Y)
    adj = mne.stats.combine_adjacency(int(tsel.sum()), NF, ch_adjacency())
    thr = st.t.ppf(1 - p_form / 2, n - 1)
    stat_fun = partial(mne.stats.ttest_1samp_no_p, sigma=1e-3)
    T, clusters, pv, H0 = mne.stats.permutation_cluster_1samp_test(
        Y, threshold=thr, n_permutations=n_perm, tail=0, adjacency=adj, stat_fun=stat_fun, seed=seed,
        n_jobs=N_JOBS, out_type="mask", verbose="ERROR")
    mean = Y.mean(0)
    out, lab = [], np.zeros(T.shape, np.int16)
    order = np.argsort(pv)
    tr = REL[tsel]
    for rank, ci in enumerate(order):
        mk = clusters[ci]
        lab[mk & (lab == 0)] = rank + 1
        if rank >= 6 and pv[ci] > 0.2:
            continue
        ti, fi, chi = np.nonzero(mk)
        vals = mean[mk]
        pk = np.argmax(np.abs(vals))
        sign = float(np.sign(T[mk].sum()))
        chs = [B.MAP_CHANNELS[c] for c in np.unique(chi)]
        chw = np.bincount(chi, minlength=NCH)
        out.append(dict(rank=rank + 1, p=float(pv[ci]), sign="+" if sign > 0 else "-", n_cells=int(mk.sum()),
                        mass=float(T[mk].sum()), channels=chs,
                        top_channels=[B.MAP_CHANNELS[c] for c in np.argsort(-chw)[:6] if chw[c] > 0],
                        freq_range=[int(B.FREQS[fi.min()]), int(B.FREQS[fi.max()])],
                        time_range=[float(tr[ti.min()]), float(tr[ti.max()])],
                        peak_db=float(vals[pk]), peak_at=dict(ch=B.MAP_CHANNELS[chi[pk]], f=int(B.FREQS[fi[pk]]),
                                                             t=float(tr[ti[pk]])),
                        mean_db=float(vals.mean())))
    min_p = float(2.0 ** -(n - 1)) if 2 ** n <= n_perm else float(1.0 / (n_perm + 1))
    return T, out, lab, dict(n=n, t_threshold=float(thr), min_attainable_p=min_p, n_time=int(tsel.sum()))


def prox_cluster_1d(P, tmax, n_perm=2048):
    """1-D (time) cluster test per proxy. P (n, prox, t)."""
    tsel = (REL >= 0) & (REL <= tmax + 1e-9)
    res = {}
    for k, name in enumerate(PROX):
        Y = P[:, k, tsel].astype(float)
        ok = np.isfinite(Y).all(1)
        Y = Y[ok]
        if len(Y) < 4:
            continue
        thr = st.t.ppf(0.975, len(Y) - 1)
        T, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(
            Y, threshold=thr, n_permutations=n_perm, tail=0, seed=0, out_type="mask", verbose="ERROR",
            stat_fun=partial(mne.stats.ttest_1samp_no_p, sigma=1e-3))
        tr = REL[tsel]
        out = []
        for c, p in sorted(zip(cl, pv), key=lambda x: x[1])[:3]:
            mk = np.zeros(len(tr), bool)
            mk[c] = True
            mk = mk.reshape(-1)
            out.append(dict(p=float(p), time_range=[float(tr[mk].min()), float(tr[mk].max())],
                            mean=float(Y[:, mk].mean())))
        res[name] = out
    return res


def ci95(v, axis=0):
    v = np.asarray(v, float)
    n = np.isfinite(v).sum(axis)
    m = np.nanmean(v, axis)
    se = np.nanstd(v, axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    h = st.t.ppf(0.975, np.maximum(n - 1, 1)) * se
    return m, m - h, m + h


def band_summary(X, tmax):
    """X (n, ch, f, t) -> {band: {bin: {roi/all: [mean, lo, hi], channels: [...]}}}"""
    out = {}
    for bn, (lo, hi) in BANDS.items():
        fm = (B.FREQS >= lo) & (B.FREQS < hi) if hi < 45 else (B.FREQS >= lo) & (B.FREQS <= hi)
        out[bn] = {}
        for a, b in TBINS:
            b = min(b, tmax)
            tm = (REL >= a) & (REL <= b)
            V = np.nanmean(X[:, :, fm][..., tm], axis=(2, 3))            # (n, ch)
            d = {}
            for g, chs in GROUPS.items():
                m, l, h = ci95(V[:, [B.MAP_CHANNELS.index(c) for c in chs]].mean(1))
                d[g] = [round(float(m), 3), round(float(l), 3), round(float(h), 3)]
            m, l, h = ci95(V.mean(1))
            d["all"] = [round(float(m), 3), round(float(l), 3), round(float(h), 3)]
            d["channels"] = {c: round(float(x), 3) for c, x in zip(B.MAP_CHANNELS, np.nanmean(V, 0))}
            out[bn][f"{a:g}-{b:g}"] = d
    return out


def prox_summary(P, tmax):
    out = {}
    for k, name in enumerate(PROX):
        out[name] = {}
        for a, b in TBINS:
            b = min(b, tmax)
            tm = (REL >= a) & (REL <= b)
            m, l, h = ci95(np.nanmean(P[:, k, tm], 1))
            out[name][f"{a:g}-{b:g}"] = [round(float(m), 3), round(float(l), 3), round(float(h), 3)]
    return out


TESTS = ["rest_push", "rest_push_s1", "rest_push_s2", "rest_walk", "push_walk", "push_rest", "walk_rest",
         "rp_minus_rw_s1", "pr_minus_wr_s1"]


def cmd_stats():
    Z = load_rec()
    summary = dict(method=dict(
        tf="multitaper DPSS 1 s window NW=1.5 (2 tapers), 0.25 s hop, 1-Hz bins 4-45 Hz, dB; values on a 0.5 s grid "
           "pool frames within +-0.5 s (support tau-1..tau+1 s)",
        baseline="local, mean of -10..-2 s before each onset, per channel x frequency",
        cleaning="eegbci.brainonly: Fp1/Fp2/F7/F8 excluded; frames overlapping blinks +-0.5 s excluded; label-free "
                 "regression of VEOG, HEOG (F7-F8, FT9-FT10), temporal EMG 30-45 Hz, neck EMG 55-70 Hz out of every channel x freq",
        stats="mne permutation_cluster_1samp_test over time x freq x 26 channels (Delaunay adjacency), t with hat "
              "sigma=1e-3, cluster-forming p<0.05 two-tailed, 2048 permutations (exact sign-flip enumeration when "
              "2^n <= 2048), tested window 0..25 s (offsets 0..18 s)",
        primary_level="subject (recordings averaged within subject)", bands=BANDS, tbins=TBINS, groups=GROUPS),
        tests={}, regression=dict(), data=dict())
    nf = Z["filled"]
    summary["data"] = dict(blink_masked_fraction={r: round(float(f), 3) for r, f in zip(Z["rec_ids"], Z["blink_frac"])},
                           n_onsets={r: dict(zip(TYPES, map(int, n))) for r, n in zip(Z["rec_ids"], Z["n_onsets"])})
    pr2 = Z["partial_r2"]
    summary["regression"] = dict(
        median_partial_r2=float(np.median(pr2)),
        by_band={bn: {g: round(float(np.median(pr2[:, [C.CHANNELS.index(c) for c in chs]][:, :, (B.FREQS >= lo) & (B.FREQS < max(hi, 46 if hi == 45 else hi))])), 4)
                      for g, chs in dict(GROUPS, excluded=B.EXCLUDE).items()} for bn, (lo, hi) in BANDS.items()})
    masks = {}
    for test in TESTS:
        for version, level in (("clean", "subject"), ("clean", "recording"), ("raw", "subject"), ("masked", "subject")):
            X, P, ids, tmax = select(Z, test, version, level)
            key = f"{test}|{version}|{level}"
            if len(X) < 4:
                summary["tests"][key] = dict(n=len(X), skipped="n < 4")
                continue
            import time
            t_start = time.time()
            T, cl, lab, info = cluster_test(X, tmax)
            info["seconds"] = round(time.time() - t_start, 1)
            entry = dict(info, units=[str(u) for u in ids], clusters=cl,
                         n_significant=int(sum(c["p"] < 0.05 for c in cl)),
                         band_summary=band_summary(X, tmax))
            if version == "clean":
                entry["eyes_control"] = dict(summary=prox_summary(P, tmax), clusters_1d=prox_cluster_1d(P, tmax))
            summary["tests"][key] = entry
            if level == "subject" and version in ("clean", "raw"):
                masks[f"T|{key}"] = T.astype(np.float32)
                sig = np.zeros_like(lab)
                for c in cl:
                    if c["p"] < 0.05:
                        sig[lab == c["rank"]] = c["rank"]
                masks[f"sig|{key}"] = sig
            msg = "; ".join(f"p={c['p']:.3f} {c['sign']} {c['freq_range'][0]}-{c['freq_range'][1]}Hz "
                            f"{c['time_range'][0]:g}-{c['time_range'][1]:g}s peak {c['peak_db']:+.2f}dB "
                            f"[{','.join(c['top_channels'][:4])}]" for c in cl[:3])
            print(f"{key:<38} n={info['n']:<3} minp={info['min_attainable_p']:.3f} | {msg}", flush=True)
            json.dump(summary, open(RES / "maps_summary.json", "w"), indent=1)
    np.savez_compressed(RES / "maps_stats.npz", **masks, rel=REL, freqs=B.FREQS, channels=np.array(B.MAP_CHANNELS))
    json.dump(summary, open(RES / "maps_summary.json", "w"), indent=1)
    print("saved maps_summary.json / maps_stats.npz")


# ------------------------------------------------------------------------------------------------ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7)


def _vlim(*arrs, q=98):
    v = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrs])
    return max(float(np.percentile(v, q)), 0.1)


def fig_panel(Z, S, test, plt):
    Xc, Pc, ids, tmax = select(Z, test, "clean", "subject")
    Xr, _, _, _ = select(Z, test, "raw", "subject")
    mc, mr = np.nanmean(Xc, 0), np.nanmean(Xr, 0)
    sig = S.get(f"sig|{test}|clean|subject")
    tsel = (REL >= 0) & (REL <= tmax + 1e-9)
    vl = _vlim(mc, mr)
    fig = plt.figure(figsize=(13, 14.5), facecolor="white")
    gs = fig.add_gridspec(len(GROUPS) + 2, 3, height_ratios=[1] * len(GROUPS) + [0.12, 1.0], hspace=0.45, wspace=0.18,
                          top=0.93, bottom=0.04)
    ext = [REL[0] - 0.25, REL[-1] + 0.25, B.FREQS[0] - 0.5, B.FREQS[-1] + 0.5]
    for gi, (g, chs) in enumerate(GROUPS.items()):
        idx = [B.MAP_CHANNELS.index(c) for c in chs]
        for col, (img, title) in enumerate(((mc[idx].mean(0), "brain-only (clean)"), (mr[idx].mean(0), "uncleaned (all frames, no regression)"),
                                            (mc[idx].mean(0) - mr[idx].mean(0), "clean minus uncleaned"))):
            ax = fig.add_subplot(gs[gi, col])
            im = ax.imshow(img, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-vl, vmax=vl,
                           interpolation="nearest")
            ax.axvline(0, color=INK, lw=0.8); ax.axvspan(BASE[0], BASE[1], ymin=0, ymax=0.03, color=MUTED)
            if tmax < 25:
                ax.axvline(tmax, color=MUTED, lw=0.8, ls=":")
            if col == 0 and sig is not None:
                full = np.zeros((NF, NT), bool)
                full[:, tsel] = (sig[:, :, idx] > 0).any(2).T
                if full.any():
                    ax.contour(REL, B.FREQS, full.astype(float), levels=[0.5], colors=INK, linewidths=1.0)
            if gi == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"{g}\n({', '.join(chs)})\nHz", fontsize=7.5, color=INK2)
            if gi == len(GROUPS) - 1:
                ax.set_xlabel("time from change (s)", fontsize=8, color=INK2)
            _style(ax)
    cax = fig.add_axes([0.92, 0.42, 0.01, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.set_label("dB vs -10..-2 s", fontsize=8, color=INK2); cb.ax.tick_params(labelsize=7)
    # eyes-only control
    sub = gs[len(GROUPS) + 1, :].subgridspec(1, len(PROX), wspace=0.35)
    for k, name in enumerate(PROX):
        ax = fig.add_subplot(sub[0, k])
        m, lo, hi = ci95(Pc[:, k])
        ax.fill_between(REL, lo, hi, color=S2, alpha=0.18, lw=0)
        ax.plot(REL, m, color=S2, lw=1.6)
        ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(PROX_LABEL[name], fontsize=8, color=INK)
        ax.set_xlabel("time (s)", fontsize=7.5, color=INK2)
        _style(ax)
    y_eye = fig.axes[-1].get_position().y1 + 0.022
    fig.text(0.5, y_eye, "Eyes-only / muscle control at the same moments (subject mean, 95% CI)", ha="center",
             fontsize=9.5, color=INK)
    n_rec = len(select(Z, test, "clean", "recording")[2])
    fig.suptitle(f"{TYPE_LABEL.get(test, test)}: channel group x frequency x time, subject mean (n = {len(ids)} subjects, "
                 f"{n_rec} recordings)\nblack contour = significant cluster (p < 0.05, subject level, clean)",
                 fontsize=11, color=INK, y=0.985)
    fig.savefig(RES / f"maps_panel_{test}.png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_topos(Z, S, plt, version="clean"):
    info = mne.create_info(B.MAP_CHANNELS, C.SFREQ, "eeg")
    info.set_montage("standard_1005")
    tests = ["rest_push", "rest_walk", "push_walk", "push_rest", "walk_rest", "rp_minus_rw_s1"]
    data = {t: select(Z, t, version, "subject") for t in tests}
    clean = {t: select(Z, t, "clean", "subject") for t in tests} if version != "clean" else data
    for bn, (lo, hi) in BANDS.items():
        fm = (B.FREQS >= lo) & ((B.FREQS < hi) if hi < 45 else (B.FREQS <= hi))
        vals, vals_c = {}, {}
        for t in tests:
            for dd, store in ((data, vals), (clean, vals_c)):
                X, _, _, tmax = dd[t]
                for a, b in TBINS:
                    tm = (REL >= a) & (REL <= min(b, tmax))
                    store[(t, a)] = np.nanmean(X[:, :, fm][..., tm], axis=(0, 2, 3))
        vl = _vlim(*vals_c.values(), q=100)                      # common scale per band (clean + raw share it)
        if version != "clean":
            vl = max(vl, _vlim(*vals.values(), q=100))
        fig, axes = plt.subplots(len(tests), len(TBINS), figsize=(7.5, 2.2 * len(tests)), facecolor="white")
        for i, t in enumerate(tests):
            sig = S.get(f"sig|{t}|{version}|subject")
            tmax = data[t][3]
            tsel = REL[(REL >= 0) & (REL <= tmax + 1e-9)]
            for j, (a, b) in enumerate(TBINS):
                ax = axes[i, j]
                mask = None
                if sig is not None:
                    tm = (tsel >= a) & (tsel <= min(b, tmax))
                    mask = (sig[tm][:, fm] > 0).any(axis=(0, 1))
                im, _ = mne.viz.plot_topomap(vals[(t, a)], info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vl, vl),
                                             mask=mask, contours=0, sensors=True,
                                             mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor=INK,
                                                              markersize=5))
                if i == 0:
                    ax.set_title(f"{a:g}-{min(b, 25):g} s", fontsize=9, color=INK)
                if j == 0:
                    ax.text(-0.25, 0.5, TYPE_LABEL[t].replace(" (", "\n("), transform=ax.transAxes, ha="right",
                            va="center", fontsize=8, color=INK2)
        cax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
        cb = fig.colorbar(im, cax=cax); cb.set_label("dB vs -10..-2 s", fontsize=8); cb.ax.tick_params(labelsize=7)
        vtxt = "brain-only (clean)" if version == "clean" else "UNCLEANED (all channels' frames, no regression)"
        fig.suptitle(f"{bn.replace('_', '/')} {lo}-{hi} Hz, {vtxt}, subject mean\nwhite dots = channel in a significant "
                     f"cluster in this band/time bin; offsets: last bin 15-18 s", fontsize=9.5, color=INK)
        fig.subplots_adjust(left=0.2, right=0.9, top=0.92, hspace=0.15, wspace=0.05)
        fig.savefig(RES / f"maps_topo_{bn}{'' if version == 'clean' else '_' + version}.png", dpi=110,
                    facecolor="white")
        plt.close(fig)


def fig_eyes(Z, plt):
    tests = ["rest_push", "rest_walk", "push_walk", "push_rest", "walk_rest"]
    fig, axes = plt.subplots(len(PROX), len(tests), figsize=(16, 11), facecolor="white", sharex=True)
    for j, t in enumerate(tests):
        _, P, ids, _ = select(Z, t, "clean", "subject")
        _, Pr, rids, _ = select(Z, t, "clean", "recording")
        for k, name in enumerate(PROX):
            ax = axes[k, j]
            for p_ in Pr[:, k]:
                ax.plot(REL, p_, color=GRID, lw=0.6)
            m, lo, hi = ci95(P[:, k])
            ax.fill_between(REL, lo, hi, color=S2, alpha=0.2, lw=0)
            ax.plot(REL, m, color=S2, lw=1.8)
            ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color=INK, lw=0.8)
            if k == 0:
                ax.set_title(f"{TYPE_LABEL[t]}\n(n = {len(ids)} subj, {len(rids)} rec)", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(PROX_LABEL[name], fontsize=8, color=INK2)
            if k == len(PROX) - 1:
                ax.set_xlabel("time from change (s)", fontsize=8, color=INK2)
            _style(ax)
    fig.suptitle("Eyes-only / muscle control: proxies time-locked to the same changes, local baseline -10..-2 s "
                 "(orange = subject mean, 95% CI; grey = recordings)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "maps_eyes_control.png", dpi=110, facecolor="white")
    plt.close(fig)


def fig_cleaning(Z, plt):
    """Band power (mean over the 26 map channels, and posterior) raw vs blink-masked vs clean."""
    tests = ["rest_push", "rest_walk", "push_walk", "push_rest", "walk_rest"]
    post = [B.MAP_CHANNELS.index(c) for c in GROUPS["parietal"] + GROUPS["occipital"]]
    fig, axes = plt.subplots(len(BANDS), len(tests), figsize=(16, 12), facecolor="white", sharex=True)
    cols = dict(raw=MUTED, masked=S1, clean=S2)
    for j, t in enumerate(tests):
        for v in VERSIONS:
            X, _, ids, _ = select(Z, t, v, "subject")
            for i, (bn, (lo, hi)) in enumerate(BANDS.items()):
                fm = (B.FREQS >= lo) & ((B.FREQS < hi) if hi < 45 else (B.FREQS <= hi))
                y = np.nanmean(X[:, post][:, :, fm], axis=(1, 2))
                ax = axes[i, j]
                m, l, h = ci95(y)
                ax.plot(REL, m, color=cols[v], lw=1.6 if v == "clean" else 1.1, label=v)
                if v == "clean":
                    ax.fill_between(REL, l, h, color=cols[v], alpha=0.15, lw=0)
                ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color=INK, lw=0.8)
                if i == 0:
                    ax.set_title(TYPE_LABEL[t], fontsize=9, color=INK)
                if j == 0:
                    ax.set_ylabel(f"{bn} {lo}-{hi} Hz\nparieto-occipital (dB)", fontsize=8, color=INK2)
                if i == len(BANDS) - 1:
                    ax.set_xlabel("time from change (s)", fontsize=8, color=INK2)
                _style(ax)
    axes[0, 0].legend(fontsize=7.5, frameon=False, title="version", title_fontsize=7.5)
    fig.suptitle("What cleaning changed: parieto-occipital band power, uncleaned (raw) vs blink-masked vs clean "
                 "(masked + proxy regression), subject mean; band = clean 95% CI", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "maps_cleaning_effect.png", dpi=110, facecolor="white")
    plt.close(fig)
    # regression diagnostics
    pr2 = np.median(Z["partial_r2"], 0)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    im = ax.imshow(pr2.T, aspect="auto", origin="lower", cmap="Blues", extent=[-0.5, 29.5, 3.5, 45.5])
    ax.set_xticks(range(30)); ax.set_xticklabels(C.CHANNELS, rotation=90, fontsize=7)
    ax.set_ylabel("Hz", fontsize=8); _style(ax)
    fig.colorbar(im, ax=ax, label="partial R2 of the proxies (median over 28 recordings)")
    ax.set_title("How much of each channel x frequency power fluctuation the eye/EMG proxies explain (removed)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RES / "maps_regression_r2.png", dpi=110, facecolor="white")
    plt.close(fig)


def fig_clusters(Z, plt):
    """Per test: up to 3 best clusters (subject level, clean): topomap of cluster channel weight + f x t extent."""
    J = json.load(open(RES / "maps_summary.json"))
    S = dict(np.load(RES / "maps_stats.npz"))
    info = mne.create_info(B.MAP_CHANNELS, C.SFREQ, "eeg")
    info.set_montage("standard_1005")
    tests = [t for t in TESTS if f"T|{t}|clean|subject" in S]
    fig, axes = plt.subplots(len(tests), 6, figsize=(16, 2.4 * len(tests)), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1.4] * 3))
    for i, t in enumerate(tests):
        e = J["tests"][f"{t}|clean|subject"]
        X, _, _, tmax = select(Z, t, "clean", "subject")
        tsel = (REL >= 0) & (REL <= tmax + 1e-9)
        Y = np.transpose(X[..., tsel], (0, 3, 2, 1)).mean(0)
        T = S[f"T|{t}|clean|subject"]
        # recompute cluster labels: threshold map identical; use summary ranks via sig mask for p<.05,
        # otherwise show the T-threshold extent around the reported peak
        for k in range(3):
            ax_t, ax_m = axes[i, 2 * k], axes[i, 2 * k + 1]
            if k >= len(e["clusters"]):
                ax_t.axis("off"); ax_m.axis("off"); continue
            c = e["clusters"][k]
            f0, f1 = c["freq_range"]; t0, t1 = c["time_range"]
            fm = (B.FREQS >= f0) & (B.FREQS <= f1)
            tm = (REL[tsel] >= t0) & (REL[tsel] <= t1)
            sup = np.abs(T) > e["t_threshold"]
            sup &= np.sign(T) == (1 if c["sign"] == "+" else -1)
            box = np.zeros_like(sup); box[np.ix_(tm, fm, np.arange(NCH))] = True
            sel = sup & box
            vals = np.array([Y[:, :, ch][sel[:, :, ch]].mean() if sel[:, :, ch].any() else 0 for ch in range(NCH)])
            mask = np.array([ch in c["channels"] for ch in B.MAP_CHANNELS])
            vl = max(np.abs(vals).max(), 0.1)
            mne.viz.plot_topomap(vals, info, axes=ax_t, show=False, cmap="RdBu_r", vlim=(-vl, vl), mask=mask,
                                 contours=0, mask_params=dict(marker="o", markerfacecolor="white",
                                                              markeredgecolor=INK, markersize=4))
            ext = sel.sum(2).T
            ax_m.imshow(ext, aspect="auto", origin="lower", cmap="Greys",
                        extent=[REL[tsel][0] - 0.25, REL[tsel][-1] + 0.25, 3.5, 45.5])
            ax_m.set_title(f"p = {c['p']:.3f} ({c['sign']}), {f0}-{f1} Hz, {t0:g}-{t1:g} s\npeak {c['peak_db']:+.2f} dB "
                           f"at {c['peak_at']['ch']} {c['peak_at']['f']} Hz", fontsize=7.5,
                           color=INK if c["p"] < 0.05 else MUTED)
            ax_m.set_ylabel("Hz", fontsize=7); _style(ax_m)
        axes[i, 0].text(-0.35, 0.5, f"{TYPE_LABEL.get(t, t)}\nn = {e['n']} subj\nmin p = {e['min_attainable_p']:.3f}",
                        transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=8, color=INK2)
    fig.suptitle("Three best clusters per test (subject level, clean). Topomap = mean dB over supra-threshold cells; "
                 "grey image = number of channels supra-threshold per freq x time (approximate extent).\n"
                 "Black title = p < 0.05, grey = not significant", fontsize=10, color=INK)
    fig.tight_layout(rect=(0.06, 0, 1, 0.96))
    fig.savefig(RES / "maps_clusters.png", dpi=100, facecolor="white")
    plt.close(fig)


def cmd_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = load_rec()
    S = dict(np.load(RES / "maps_stats.npz"))
    for t in ["rest_push", "rest_walk", "push_walk", "push_rest", "walk_rest", "rp_minus_rw_s1"]:
        fig_panel(Z, S, t, plt)
    fig_topos(Z, S, plt, "clean")
    fig_topos(Z, S, plt, "raw")
    fig_eyes(Z, plt)
    fig_cleaning(Z, plt)
    fig_clusters(Z, plt)
    print("figures saved")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("compute", "all"):
        cmd_compute()
    if cmd in ("stats", "all"):
        cmd_stats()
    if cmd in ("figures", "all"):
        cmd_figures()
