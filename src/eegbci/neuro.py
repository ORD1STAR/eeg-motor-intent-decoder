"""Event-related spectral perturbation at block transitions.

Each block change is treated as an event. Band power is baseline-corrected on the seconds just
before the change, so slow drift cancels. A genuine state change shows opposite-signed responses
at onsets (rest->task) and offsets (task->rest), which drift cannot produce.

python -m eegbci.neuro
"""
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon

from . import config as C
from .data import load_recordings

STEP = 0.25
PRE, POST = (-12.0, -2.0), (-12.0, 20.0)


def transitions(rec):
    """List of (time, kind) where kind in rest->task, task->rest, push->walk, walk->push"""
    out = []
    bl = [b for b in rec.blocks if b[2] is not None]
    for a, b in zip(bl[:-1], bl[1:]):
        if a[1] != b[0]:
            continue
        la, lb = a[2], b[2]
        if la == C.NEUTRAL and lb != C.NEUTRAL:
            out.append((b[0], "rest->task", lb))
        elif la != C.NEUTRAL and lb == C.NEUTRAL:
            out.append((b[0], "task->rest", la))
        elif la == C.PUSH and lb == C.WALK:
            out.append((b[0], "push->walk", lb))
        elif la == C.WALK and lb == C.PUSH:
            out.append((b[0], "walk->push", lb))
    return out


def band_envelope(x, lo, hi, sfreq=C.SFREQ, win_s=1.0):
    sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    p = sosfiltfilt(sos, x, axis=1) ** 2
    w, s = int(win_s * sfreq), int(STEP * sfreq)
    n = (p.shape[1] - w) // s + 1
    cs = np.cumsum(np.pad(p, ((0, 0), (1, 0))), axis=1)
    pw = (cs[:, w + np.arange(n) * s] - cs[:, np.arange(n) * s]) / w
    t = (np.arange(n) * s + w / 2) / sfreq
    return 10 * np.log10(pw + 1e-30), t


def ersp(recs, band, kinds=("rest->task", "task->rest", "push->walk")):
    rel = np.arange(POST[0], POST[1] + 1e-9, STEP)
    per_rec = {k: [] for k in kinds}
    for r in recs:
        env, t = band_envelope(r.data, *band)
        by_kind = {k: [] for k in kinds}
        for t0, kind, _ in transitions(r):
            if kind not in kinds:
                continue
            idx = np.searchsorted(t, t0 + rel)
            if idx[0] < 0 or idx[-1] >= len(t):
                continue
            seg = env[:, idx]
            base = seg[:, (rel >= PRE[0]) & (rel < PRE[1])].mean(1, keepdims=True)
            by_kind[kind].append(seg - base)
        for k, v in by_kind.items():
            if v:
                per_rec[k].append(np.mean(v, axis=0))
    return rel, {k: np.array(v) for k, v in per_rec.items()}


def main():
    recs = load_recordings()
    out = {}
    for bname, band in [("mu", (8, 13)), ("beta", (13, 30)), ("theta", (4, 8))]:
        rel, R = ersp(recs, band)
        post = (rel >= 3) & (rel <= 15)
        for kind, A in R.items():
            eff = A[:, :, post].mean(2)                                    # (n_rec, n_ch) dB
            stats = []
            for ci, ch in enumerate(C.CHANNELS):
                p = wilcoxon(eff[:, ci]).pvalue if len(eff) > 5 else np.nan
                stats.append((ch, float(np.median(eff[:, ci])), float(p)))
            out[f"{bname}|{kind}"] = dict(n_rec=len(A), rel=rel.tolist(), mean_curve={
                g: A[:, [C.CHANNELS.index(c) for c in chs]].mean((0, 1)).tolist()
                for g, chs in [("motor", C.MOTOR), ("occipital", ["O1", "Oz", "O2"]), ("frontopolar", C.FRONTOPOLAR)]},
                channels=stats)
            top = sorted(stats, key=lambda s: s[2])[:6]
            print(f"{bname:<5} {kind:<11} n={len(A):<3} motor {np.median(eff[:, [C.CHANNELS.index(c) for c in C.MOTOR]]):+.2f} dB"
                  f" | occ {np.median(eff[:, [C.CHANNELS.index(c) for c in ('O1', 'Oz', 'O2')]]):+.2f}"
                  f" | Fp {np.median(eff[:, [0, 29]]):+.2f} | strongest: "
                  + ", ".join(f"{c} {m:+.2f}dB p={p:.3f}" for c, m, p in top))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(C.RESULTS_DIR / "ersp.json", "w"))


if __name__ == "__main__":
    main()
