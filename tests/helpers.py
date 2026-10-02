"""Shared helpers for the eegbci QA tests: synthetic EEG, fake recordings, a chunked feeder for the
streaming decoder and an exact replica of the realtime decision rule (used to quantify the
offline/online decision mismatch)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from eegbci import config as C

PO_NAMES = ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def make_signal(seconds=72.0, seed=0, sfreq=C.SFREQ, spike_at_s=None, bad_channel=None):
    """(30, n) volts: spatially mixed 1/f-like noise (~10 uV) + 10 Hz parieto-occipital bursts whose
    amplitude switches every 12 s. n is a multiple of the decimation factor (4)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sfreq))
    n -= n % 4
    n_ch = len(C.CHANNELS)
    pink = lfilter([1.0], [1.0, -0.95], rng.standard_normal((n_ch, n)), axis=1)
    mix = np.eye(n_ch) + 0.3 * rng.standard_normal((n_ch, n_ch)) / np.sqrt(n_ch)
    x = mix @ pink
    t = np.arange(n) / sfreq
    env = 1.0 + 1.5 * (np.sin(2 * np.pi * t / 24.0) > 0)
    x[[C.CHANNELS.index(c) for c in PO_NAMES]] += 2.0 * env * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
    x *= 3e-6
    if spike_at_s is not None:
        i = int(spike_at_s * sfreq)
        x[C.CHANNELS.index("Cz"), i:i + 25] += 400e-6
    if bad_channel is not None:
        x[C.CHANNELS.index(bad_channel)] *= 50.0
    return x


@dataclass
class FakeRec:
    """Duck-typed stand-in for data.Recording (only what windows/evaluation/realtime use)."""
    data: np.ndarray
    rec_id: str = "T00-1"
    subject: str = "T00"
    session: str = "1"
    scenario: int = 2
    blocks: list = field(default_factory=list)
    sfreq: float = C.SFREQ
    ch_names: list = field(default_factory=lambda: list(C.CHANNELS))

    @property
    def n_times(self):
        return self.data.shape[1]

    @property
    def duration(self):
        return self.data.shape[1] / self.sfreq


def toy_meta(duration, win_s=2.0, hop_s=0.5, first_end=None):
    te = np.arange(win_s if first_end is None else first_end, duration + 1e-9, hop_s)
    return pd.DataFrame(dict(t_end=te, t_start=te - win_s, ptp_max=np.zeros(len(te))))


def feed(dec, x, sizes):
    """Push x through a BrainSwitchDecoder in chunks whose sizes cycle through ``sizes``."""
    outs, i, k = [], 0, 0
    while i < x.shape[1]:
        n = int(sizes[k % len(sizes)])
        k += 1
        outs += dec.push(x[:, i:i + n])
        i += n
    return outs


def rt_rule(p, margin=0.02, min_on=2, tau_s=120.0, step_s=0.5, warm=40):
    """Replica of BrainSwitchDecoder's adaptive decision (realtime.py, `_window`)."""
    ta = 1.0 - np.exp(-step_s / tau_s)
    m, k, above, active, out, thr_seq = None, 0, 0, False, [], []
    for x in p:
        thr = x if m is None else m
        thr_seq.append(thr)
        on, off = x > thr + margin, x < thr - margin
        k += 1
        m = x if m is None else (m + (x - m) / k if k <= warm else (1 - ta) * m + ta * x)
        above = above + 1 if on else 0
        if above >= min_on:
            active = True
        if off:
            active = False
        out.append(active)
    return np.array(out), np.array(thr_seq)


class ScriptedModel:
    """Stands in for the fitted LR: predict_proba returns a scripted P(active) sequence."""
    classes_ = np.array([0, 1])

    def __init__(self, seq):
        self.seq = np.asarray(seq, float)
        self.i = 0

    def predict_proba(self, X):
        p = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return np.array([[1.0 - p, p]])


def ref_label(ts, te, blocks, duration, guard=C.GUARD_S):
    """Independent reference for windows.label_windows on ONE window already mapped to protocol time."""
    if te > duration:
        return -1, -1, -1
    for i, (t0, t1, lab, _name, cyc) in enumerate(blocks):
        if ts >= t0 + guard and te <= t1 - guard:
            return (-2, -1, -1) if lab is None else (lab, i, cyc)
    return -1, -1, -1


def require(cond, msg):
    """Precondition check that must NOT be mistaken for the expected failure of an xfail test."""
    if not cond:
        raise RuntimeError(f"test precondition violated: {msg}")


# ====================================================================================================
# Reference implementations of the PROPOSED patches (see results/code_review.md). They live here so that
# each recommended fix is validated by a test; production code is untouched.
# ====================================================================================================
def proposed_label_windows(meta, blocks, duration, shift=0.0, guard=C.GUARD_S):
    """windows.label_windows with the circular shift restricted to the analysed span: segments with label None
    (eyes-closed baseline, a property of the DATA) stay excluded in real time under any shift."""
    n = len(meta)
    lab = np.full(n, -1); blk = np.full(n, -1); cyc = np.full(n, -1); tib = np.full(n, np.nan)
    lo = min((b[0] for b in blocks if b[2] is not None), default=0.0)
    span = duration - lo
    t_s, t_e = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    ts = lo + (t_s - lo - shift) % span
    te = ts + (t_e - t_s)
    analysed = t_s >= lo
    for i, (t0, t1, lb, _name, c) in enumerate(blocks):
        if lb is None:
            lab[(t_e > t0) & (t_s < t1)] = -2
            continue
        inside = (ts >= t0 + guard) & (te <= t1 - guard) & (te <= duration) & analysed
        lab[inside] = lb; blk[inside] = i; cyc[inside] = c; tib[inside] = te[inside] - t0
    return lab, blk, cyc, tib


def proposed_detrend_causal(F, tau_s=120.0, step_s=0.5, warmup_s=20.0):
    """features.detrend(mode='causal') using the mean of PAST windows only."""
    a = 1.0 - np.exp(-step_s / tau_s)
    n_warm = max(1, int(warmup_s / step_s))
    out, m = np.empty_like(F), F[0].copy()
    for i in range(len(F)):
        out[i] = F[i] - m
        m = m + (F[i] - m) / (i + 1) if i + 1 < n_warm else (1 - a) * m + a * F[i]
    return out


def proposed_detections(rec, outs):
    """realtime.online_metrics onset detection: an onset counts only if the switch turns ON inside the block."""
    t = np.array([o.t for o in outs]); act = np.array([o.active for o in outs])
    rise_t = t[np.flatnonzero(np.diff(act.astype(int)) == 1) + 1]
    bl = [b for b in rec.blocks if b[2] is not None]
    n_on, lats = 0, []
    for a, b in zip(bl[:-1], bl[1:]):
        if a[2] == C.NEUTRAL and b[2] != C.NEUTRAL:
            n_on += 1
            r = rise_t[(rise_t > b[0]) & (rise_t <= b[1])]
            if len(r):
                lats.append(r[0] - b[0])
    return n_on, len(lats), lats


def make_proposed_decoder_class():
    """BrainSwitchDecoder with two proposed fixes:
    (1) buffers are trimmed AFTER the completed windows are emitted (no overrun for long chunks);
    (2) the deep branch runs on its own 2 s grid (ends 250 + 62k, from t = 2 s) so the causal Euclidean
        alignment sees every 2 s window, and each 4 s window is paired with the latest deep output (the 2 s
        window ending 2 decimated samples earlier) -- exactly the offline construction of ensemble.py."""
    from eegbci.realtime import BrainSwitchDecoder
    from eegbci.windows import DECIM

    class ProposedDecoder(BrainSwitchDecoder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.next_end_deep = self.n_t if self.net is not None else np.inf
            self.p_deep_last = float("nan")

        def _deep(self, sl):                       # called from _window: reuse the latest 2 s-grid output
            return self.p_deep_last

        def push(self, chunk):
            outs = []
            filtered = {b: f(chunk) for b, f in self.filters.items()}
            broad = self.broad(chunk)
            keep = np.arange(chunk.shape[1])[(self.n_in + np.arange(chunk.shape[1])) % DECIM == 0]
            self.n_in += chunk.shape[1]
            for b in self.buf:
                self.buf[b] = np.concatenate([self.buf[b], filtered[b][:, keep]], axis=1)
            self.bbuf = np.concatenate([self.bbuf, broad[:, keep]], axis=1)
            self.n_dec += len(keep)
            while True:
                if self.next_end_deep <= self.next_end and self.n_dec >= self.next_end_deep:
                    stop = self.bbuf.shape[1] - (self.n_dec - self.next_end_deep)
                    self.p_deep_last = BrainSwitchDecoder._deep(self, slice(stop - self.n_t, stop))
                    self.next_end_deep += self.step
                elif self.n_dec >= self.next_end:
                    outs.append(self._window(self.n_dec - self.next_end))
                    self.next_end += self.step
                else:
                    break
            n_keep = self.win + 4 * self.step
            for b in self.buf:
                self.buf[b] = self.buf[b][:, -n_keep:]
            self.bbuf = self.bbuf[:, -n_keep:]
            return outs

    return ProposedDecoder
