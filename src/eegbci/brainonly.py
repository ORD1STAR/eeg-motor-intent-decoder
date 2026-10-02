"""Brain-only spectral power: what is removed, how, and what cannot be removed.

Reusable by any module that wants channel x frequency x time log-power "as free of eyes and muscle as this
montage allows". There are no EOG / EMG channels, so every artefact estimate comes from the EEG itself.

Pipeline (``brain_power(rec)``), per recording, label-free (the protocol is never used to fit anything):

0. Bad-channel repair.  Bad channels found by the cache builder (``confounds.bads_of``, or the copy stored in
   results/confounds_blinks.json when the cache file is absent) are replaced by
   spherical-spline interpolation (mne, standard_1005 montage).  Native "TP" reference is kept: an average
   reference would spread the frontal ocular field onto every channel.

1. Time-frequency power.  Multitaper (DPSS, 1 s window, NW = 1.5 -> 2 tapers, half-bandwidth 1.5 Hz),
   frames every 0.25 s, PSD in uV^2/Hz averaged into 1-Hz bins 4..45 Hz (bin f = [f-0.5, f+0.5]), log
   power in dB.  Adjacent bins are correlated (1.5 Hz spectral smoothing).

2. (a) CHANNEL EXCLUSION.  ``EXCLUDE = Fp1, Fp2, F7, F8`` are dropped from the brain maps (they are the
   VEOG/HEOG pick-ups and the frontalis-EMG sites).  They are still used to build the proxies.

   (b) BLINK MASK.  Blinks are detected by ``confounds.detect_blinks`` (cached in
   results/confounds_blinks.json): positive, bilateral, frontal-dominant peaks of mean(Fp1, Fp2) at
   0.5-8 Hz.  Samples within +-0.5 s of each blink peak are masked (``blink_mask``); any 1-s frame that
   overlaps a masked sample is set to NaN and never enters a power average (``frame_mask``).

   (c) PROXY REGRESSION.  Proxy time-courses on the same frames (dB):
         veog    0.5-4 Hz power of mean(Fp1, Fp2)         vertical eye movements / slow eye drift
         heog_f  0.5-4 Hz power of F7 - F8                horizontal eye movements
         heog_ft 0.5-4 Hz power of FT9 - FT10             horizontal eye movements (lower, more lateral)
         emg     30-45 Hz power, mean of T7 T8 FT9 FT10   temporalis / jaw muscle
         emg_neck 55-70 Hz power, mean of O1 Oz O2 P7 P8  neck / occipital muscle (above the 4-45 Hz maps, so
                  regressing it does not remove posterior gamma by construction; it removes the part of every
                  channel x frequency that co-varies with posterior high-frequency muscle activity)
       For every channel x frequency, an OLS model  P(t) = a + D(t) b + Z(t) g + e  is fitted on all
       blink-free frames of the recording (scenario 1: the eyes-closed baseline 0-120 s is left out of the
       fit, it is not analysed).  Z = z-scored proxies, D = slow-drift DCT basis (periods >= 240 s), which
       is a nuisance term so that slow co-drift of proxy and power is not attributed to the proxy.  Only the
       proxy part Z g is subtracted; drift and intercept stay in the data (the maps remove drift with a
       local pre-onset baseline).

What the cleaning CANNOT remove (read before interpreting):
  * Ocular activity below the proxy detection floor: micro-saccades, small saccades with no F7-F8 signature,
    eyelid tone, sustained partial eye closure (the alpha rise it causes is a real brain-state change that
    is not linearly predicted by VEOG power, so it survives regression).
  * Indirect effects of the eyes on the brain: visual input changes when the eyes close / fixate (occipital
    alpha reactivity) are neural and are kept on purpose; they are *behaviour-linked*, not *command-linked*.
  * Muscle not captured by the temporal / neck proxies (frontalis above Fp, swallowing, isolated motor
    units), and neck EMG whose 30-45 Hz part does not scale with its 55-70 Hz part; and any
    EMG at frequencies < 30 Hz (EMG is broadband; its 13-30 Hz part is only removed if it is linearly
    proportional to the 30-45 Hz proxy).
  * Non-linear or channel-varying artefact transfer beyond a single linear gain per channel x frequency.
  * Over-correction: a genuine task effect that co-varies with the proxies is partly removed (conservative).
    Gamma at T7/T8/FT9/FT10 is, by construction, almost fully removed (it *is* the EMG proxy).
  * Blink masking makes time points near blinks NaN; the remaining frames are a non-random, blink-free
    subset (eyes open, not blinking).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss

from . import config as C
from .confounds import bads_of, blinks_cache

FS = C.SFREQ
CH = {c: i for i, c in enumerate(C.CHANNELS)}
EXCLUDE = ["Fp1", "Fp2", "F7", "F8"]
MAP_CHANNELS = [c for c in C.CHANNELS if c not in EXCLUDE]
MAP_IDX = [CH[c] for c in MAP_CHANNELS]
FREQS = np.arange(4, 46)                    # 1-Hz bins 4..45 Hz
WIN_S, HOP_S, NW = 1.0, 0.25, 1.5
BLINK_HALF = 0.5
EMG_CH = ["T7", "T8", "FT9", "FT10"]
NECK_CH = ["O1", "Oz", "O2", "P7", "P8"]
PROXY_NAMES = ["veog", "heog_f", "heog_ft", "emg", "emg_neck"]
DRIFT_PERIOD = 240.0


# ------------------------------------------------------------------------------------------------ helpers
def repair_bads(xuv, bads):
    """Spherical-spline interpolation of bad channels (in place on a copy). xuv: (30, n) uV."""
    if not bads:
        return xuv
    info = mne.create_info(C.CHANNELS, FS, "eeg")
    raw = mne.io.RawArray(xuv * 1e-6, info, verbose="ERROR")
    raw.set_montage("standard_1005")
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw.get_data() * 1e6


def frame_times(n_samples):
    """Centres (s) of 1-s frames every 0.25 s; frame k covers [k*hop, k*hop + win)."""
    w, h = int(WIN_S * FS), int(HOP_S * FS)
    n = (n_samples - w) // h + 1
    return (np.arange(n) * h + w / 2) / FS


def multitaper_tf(xuv, freqs=FREQS):
    """(n_ch, n_freqs, n_frames) log power (dB re 1 uV^2/Hz), 1-Hz bins, plus frame centres."""
    w, h = int(WIN_S * FS), int(HOP_S * FS)
    n = (xuv.shape[1] - w) // h + 1
    K = max(1, int(2 * NW - 1))
    tap = dpss(w, NW, K)                                     # (K, w)
    nfft = 2 * w                                             # 0.5-Hz grid
    fgrid = np.fft.rfftfreq(nfft, 1 / FS)
    keep = (fgrid >= freqs[0] - 0.5) & (fgrid <= freqs[-1] + 0.5)
    fk = fgrid[keep]
    # 1-Hz bins: triangular weights over the 0.5-Hz grid (f-0.5: .25, f: .5, f+0.5: .25)
    Wb = np.zeros((len(freqs), len(fk)))
    for i, f in enumerate(freqs):
        for df, wt in ((-0.5, .25), (0.0, .5), (0.5, .25)):
            Wb[i, np.argmin(np.abs(fk - (f + df)))] += wt
    idx = np.arange(w)[None, :] + (np.arange(n) * h)[:, None]   # (n, w)
    out = np.empty((xuv.shape[0], len(freqs), n), np.float32)
    for c in range(xuv.shape[0]):
        seg = xuv[c][idx]
        seg = seg - seg.mean(1, keepdims=True)
        spec = np.zeros((n, keep.sum()))
        for k in range(K):
            spec += np.abs(np.fft.rfft(seg * tap[k], nfft, axis=1)[:, keep]) ** 2
        psd = spec / K * 2 / FS                               # one-sided, uV^2/Hz (taper energy 1)
        out[c] = (10 * np.log10(psd @ Wb.T + 1e-20)).T
    return out, frame_times(xuv.shape[1])


def _bp(x, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos"), x, axis=-1)


def _frame_mean(p, n_frames):
    w, h = int(WIN_S * FS), int(HOP_S * FS)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    s = np.arange(n_frames) * h
    return (cs[s + w] - cs[s]) / w


def proxy_series(xuv, bads, n_frames):
    """dict name -> (n_frames,) dB, on the multitaper frame grid. NaN when the needed channels are bad."""
    out = {}
    fp = [c for c in ("Fp1", "Fp2") if c not in bads] or ["Fp1", "Fp2"]
    db = lambda p: 10 * np.log10(_frame_mean(p, n_frames) + 1e-20)
    out["veog"] = db(_bp(xuv[[CH[c] for c in fp]].mean(0), 0.5, 4.0) ** 2)
    for name, (a, b) in (("heog_f", ("F7", "F8")), ("heog_ft", ("FT9", "FT10"))):
        out[name] = (np.full(n_frames, np.nan) if {a, b} & set(bads)
                     else db(_bp(xuv[CH[a]] - xuv[CH[b]], 0.5, 4.0) ** 2))
    emg = [c for c in EMG_CH if c not in bads]
    out["emg"] = (db(np.mean([_bp(xuv[CH[c]], 30, 45) ** 2 for c in emg], axis=0)) if emg
                  else np.full(n_frames, np.nan))
    neck = [c for c in NECK_CH if c not in bads]
    out["emg_neck"] = db(np.mean([_bp(xuv[CH[c]], 55, 70) ** 2 for c in neck], axis=0))
    return out


def blink_mask(n_samples, blink_t, half=BLINK_HALF):
    """(n_samples,) bool, True within +-half s of a blink peak."""
    m = np.zeros(n_samples, bool)
    for b in blink_t:
        m[max(0, int((b - half) * FS)):min(n_samples, int((b + half) * FS) + 1)] = True
    return m


def frame_mask(sample_mask, n_frames):
    """True for frames whose 1-s window contains any masked sample."""
    return _frame_mean(sample_mask.astype(float), n_frames) > 0


def blink_rate_frames(blink_t, t_frames, win=2.0):
    """Blinks/min in a centred window around each frame (eyes-only control)."""
    lo, hi = t_frames - win / 2, t_frames + win / 2
    bt = np.asarray(blink_t)
    return (np.searchsorted(bt, hi) - np.searchsorted(bt, lo)) / win * 60.0


def drift_basis(t, period_s=DRIFT_PERIOD):
    T = t[-1] - t[0]
    K = int(np.floor(2 * T / period_s))
    u = (t - t[0]) / T
    return np.stack([np.cos(np.pi * k * u) for k in range(1, K + 1)], 1) if K else np.zeros((len(t), 0))


def regress_proxies(P, prox, fit_mask, t):
    """Remove the linear proxy contribution from P (n_ch, n_f, n_t) in dB.
    Fit on frames in fit_mask with intercept + slow DCT drift (nuisance, kept) + z-scored proxies (removed).
    Returns (P_clean float32, info dict with per-proxy mean |gain| and variance explained)."""
    names = [k for k in PROXY_NAMES if np.isfinite(prox[k]).all()]
    Z = np.stack([prox[k] for k in names], 1)
    mu, sd = Z[fit_mask].mean(0), Z[fit_mask].std(0) + 1e-12
    Z = (Z - mu) / sd
    A = np.c_[np.ones(len(t)), drift_basis(t), Z]
    nz = Z.shape[1]
    Y = P.reshape(-1, P.shape[-1]).T.astype(float)            # (n_t, ch*f)
    m = fit_mask & np.isfinite(Y).all(1)
    beta = np.linalg.lstsq(A[m], Y[m], rcond=None)[0]
    G = beta[-nz:]                                            # (nz, ch*f)
    remove = Z @ G
    resid_full = Y[m] - A[m] @ beta
    resid_noprox = Y[m] - A[m, :-nz] @ np.linalg.lstsq(A[m, :-nz], Y[m], rcond=None)[0]
    r2 = 1 - (resid_full ** 2).sum(0) / ((resid_noprox ** 2).sum(0) + 1e-20)   # partial R2 of the proxies
    Pc = (Y - remove).T.reshape(P.shape).astype(np.float32)
    info = dict(proxies=names, partial_r2=r2.reshape(P.shape[:2]).astype(np.float32),
                gain=G.reshape((nz,) + P.shape[:2]).astype(np.float32))
    return Pc, info


# ------------------------------------------------------------------------------------------------ main API
def brain_power(rec, blinks=None):
    """All versions of the per-recording TF power + proxies.

    Returns dict:
      t            (n_frames,) frame centres (s)
      raw          (30, 42, n_frames) dB, bad channels interpolated, no cleaning
      masked       raw with blink-overlapping frames set to NaN
      clean        masked + proxy regression (use ``MAP_IDX`` to drop the excluded channels)
      proxies      dict veog/heog_f/heog_ft/emg/emg_neck (dB) + blink_rate (/min), unmasked
      blink_frames (n_frames,) bool
      reg_info     regression diagnostics
    The caller should ``rec.release()`` afterwards."""
    blinks = blinks if blinks is not None else blinks_cache()
    # bad channels: cache builder's list, else the list stored with the cached blinks (same detector)
    bads = sorted(set(bads_of(rec)) | set(blinks.get(rec.rec_id, {}).get("bads", [])))
    xuv = rec.data * 1e6
    bt = np.asarray(blinks[rec.rec_id]["t"]) if rec.rec_id in blinks else np.array([])
    prox_bads = list(bads)
    xrep = repair_bads(xuv, bads)
    P, t = multitaper_tf(xrep)
    prox = proxy_series(xuv, prox_bads, len(t))
    del xrep
    bf = frame_mask(blink_mask(xuv.shape[1], bt), len(t))
    prox["blink_rate"] = blink_rate_frames(bt, t)
    fit = ~bf
    if rec.scenario == 1:
        fit &= t >= 120.0 + WIN_S / 2                       # eyes-closed baseline not used in the fit
    Pm = P.copy()
    Pm[:, :, bf] = np.nan
    Pc, info = regress_proxies(P, prox, fit, t)
    Pc[:, :, bf] = np.nan
    return dict(t=t, raw=P, masked=Pm, clean=Pc, proxies=prox, blink_frames=bf, reg_info=info, bads=bads)
