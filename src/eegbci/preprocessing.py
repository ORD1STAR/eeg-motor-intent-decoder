"""Signal conditioning. Every filter here is causal (SOS + carried state), so the offline
pipeline sees exactly what the online decoder will see."""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import config as C


def bandpass_sos(lo, hi, sfreq=C.SFREQ, order=4):
    return butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")


class CausalFilter:
    """Stateful causal band-pass; call repeatedly on consecutive chunks (n_channels, n_samples)."""

    def __init__(self, lo, hi, n_channels, sfreq=C.SFREQ, order=4):
        self.sos = bandpass_sos(lo, hi, sfreq, order)
        self.zi = None
        self.n_channels = n_channels

    def __call__(self, x):
        if self.zi is None:
            # start from the steady state of the first sample to avoid a large onset transient
            # sosfilt along axis=1 of (n_ch, n_s) expects zi of shape (n_sections, n_ch, 2)
            self.zi = sosfilt_zi(self.sos)[:, None, :] * x[:, 0][None, :, None]
        y, self.zi = sosfilt(self.sos, x, axis=1, zi=self.zi)
        return y


def causal_bandpass(x, lo, hi, sfreq=C.SFREQ, order=4):
    return CausalFilter(lo, hi, x.shape[0], sfreq, order)(x)


def common_average(x):
    return x - x.mean(axis=0, keepdims=True)


def find_bad_channels(x, ch_names, z_thresh=4.0):
    """Robust z-score of log-variance (1-40 Hz). Returns names of outlier channels."""
    xf = causal_bandpass(x, 1, 40)
    lv = np.log(xf[:, int(5 * C.SFREQ):].var(axis=1))
    med = np.median(lv)
    mad = np.median(np.abs(lv - med)) * 1.4826 + 1e-12
    z = (lv - med) / mad
    # Fp1/Fp2 always carry ocular activity; only flag them if they are extreme
    return [ch for ch, zz in zip(ch_names, z) if zz > z_thresh + (2 if ch in C.FRONTOPOLAR else 0) or zz < -z_thresh]


def interpolation_matrix(ch_names, bads):
    """(n_ch, n_ch) linear operator M so that M @ x interpolates ``bads`` (identity if none). Spline
    interpolation is linear per sample, so M is obtained by interpolating the identity."""
    n = len(ch_names)
    if not bads:
        return np.eye(n)
    return interpolate_bads(np.eye(n), ch_names, bads)


def interpolate_bads(x, ch_names, bads, sfreq=C.SFREQ):
    if not bads:
        return x
    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(np.array(x, copy=True), info, verbose=False)
    raw.set_montage("standard_1005")
    raw.info["bads"] = list(bads)
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw.get_data()
