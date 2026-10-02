"""Turn recordings into sliding windows: filter-bank covariances, log band powers, short raw
epochs for deep models, and a metadata table. Cached per recording on disk.

A window ending at time t only uses samples <= t (causal), exactly like the online decoder.
Labels are attached separately (``label_windows``) so the same windows can be relabelled for
null distributions (circularly shifted protocols).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pyriemann.estimation import Covariances

from . import config as C
from .preprocessing import causal_bandpass, common_average, find_bad_channels, interpolation_matrix

BAD_PREFIX_S = 60.0

DECIM = 4                     # 500 Hz -> 125 Hz after band-limiting (content < 45 Hz)
FS_D = C.SFREQ / DECIM


def _params_key(p):
    return hashlib.md5(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]


def build_windows(rec, reref="none", interp_bads=True, win_s=C.WIN_S, step_s=C.STEP_S, bands=C.BANDS,
                  raw_band=(1, 40), cache=True):
    params = dict(reref=reref, interp=interp_bads, win=win_s, step=step_s, bands=bands, raw_band=raw_band, v=5)
    path = C.CACHE_DIR / f"{rec.rec_id}_{_params_key(params)}.npz"
    raw_path = path.with_name(path.stem + "_raw.npy")
    if cache and path.exists() and raw_path.exists():
        z = np.load(path, allow_pickle=True)
        meta = pd.DataFrame(z["meta"].tolist())
        # raw epochs are memory-mapped: only read when a deep model actually needs them
        return dict(covs=z["covs"], logbp=z["logbp"], raw=np.load(raw_path, mmap_mode="r"), meta=meta,
                    bads=list(z["bads"]))

    x = rec.data
    # bad channels are detected on a 60 s calibration prefix only (causal), then frozen; the same
    # spatial interpolation matrix is used by the online decoder
    bads = find_bad_channels(x[:, :int(BAD_PREFIX_S * C.SFREQ)], rec.ch_names) if interp_bads else []
    x = interpolation_matrix(rec.ch_names, bads) @ x
    if reref == "car":
        x = common_average(x)

    win, step = int(win_s * FS_D), int(step_s * FS_D)
    ends = np.arange(win, x.shape[1] // DECIM + 1, step)             # window end index (decimated samples)

    def windows_of(sig):
        d = sig[:, ::DECIM]
        v = sliding_window_view(d, win, axis=1)                       # (ch, n_all, win)
        return np.ascontiguousarray(v[:, ends - win, :].transpose(1, 0, 2))

    cov_est = Covariances("oas")
    covs = []
    for lo, hi in list(bands) + [C.MI_BAND]:
        w = windows_of(causal_bandpass(x, lo, hi))
        covs.append(cov_est.transform(w.astype(np.float64)))
    covs = np.stack(covs, axis=1).astype(np.float32)                 # (n, n_bands+1, ch, ch)

    broad = causal_bandpass(x, *raw_band)
    raw_w = windows_of(broad)                                         # (n, ch, win) volts
    ptp = np.ptp(raw_w, axis=2) * 1e6
    logbp = np.log(np.stack([np.diagonal(covs[:, b], axis1=1, axis2=2) for b in range(covs.shape[1])], axis=1)
                   ).reshape(len(ends), -1).astype(np.float32)
    fp = [rec.ch_names.index(c) for c in C.FRONTOPOLAR]
    nonfp = [i for i in range(len(rec.ch_names)) if i not in fp]
    t_end = ends / FS_D
    meta = pd.DataFrame(dict(rec_id=rec.rec_id, subject=rec.subject, session=rec.session, scenario=rec.scenario,
                             t_end=t_end, t_start=t_end - win_s, ptp_max=ptp[:, nonfp].max(1), ptp_fp=ptp[:, fp].max(1)))
    raw_w = (raw_w * 1e6).astype(np.float32)                           # µV for deep models
    if cache:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, raw_w)
        np.savez(path, covs=covs, logbp=logbp, meta=np.array(meta.to_dict("records"), dtype=object),
                 bads=np.array(bads, dtype=object))
        raw_w = np.load(raw_path, mmap_mode="r")
    return dict(covs=covs, logbp=logbp, raw=raw_w, meta=meta, bads=bads)


def fill_cache(rec, **kw):
    """Build (or reuse) the cached windows without sending arrays back to the parent process."""
    build_windows(rec, **kw)
    return rec.rec_id


def label_windows(meta, blocks, duration, shift=0.0, guard=C.GUARD_S):
    """Assign protocol labels to windows. ``shift`` circularly shifts the protocol timeline
    (for null distributions). Returns arrays label (-1 = unusable), block index, cycle, time-in-block."""
    # The shift is restricted to the analysed span; segments with label None (eyes-closed baseline,
    # a property of the DATA) stay excluded in real time under any shift (QA finding H2).
    n = len(meta)
    lab = np.full(n, -1); blk = np.full(n, -1); cyc = np.full(n, -1); tib = np.full(n, np.nan)
    lo = min((b[0] for b in blocks if b[2] is not None), default=0.0)
    span = duration - lo
    t_s, t_e = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    ts = lo + (t_s - lo - shift) % span
    te = ts + (t_e - t_s)
    analysed = t_s >= lo
    for i, (t0, t1, l, name, c) in enumerate(blocks):
        if l is None:
            lab[(t_e > t0) & (t_s < t1)] = -2      # excluded segment (eyes-closed baseline), real time
            continue
        inside = (ts >= t0 + guard) & (te <= t1 - guard) & (te <= duration) & analysed
        lab[inside] = l; blk[inside] = i; cyc[inside] = c; tib[inside] = te[inside] - t0
    return lab, blk, cyc, tib
