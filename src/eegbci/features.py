"""Riemannian feature extraction with drift handling.

All operations work per recording, in time order, and use no labels:

* ``recenter``    whiten every covariance by a reference mean of the same recording
                  ('recording' = mean of all windows, 'causal' = exponentially-weighted mean of the past)
* ``tangent``     log-map at identity -> upper-triangular vector (Euclidean space)
* ``detrend``     remove slow drift from feature vectors ('poly' over the recording, or 'causal' EWMA)
"""
from __future__ import annotations

import numpy as np


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    # relative floor: scale-invariant (covariances are in V^2, an absolute 1e-12 clipped most eigenvalues)
    return (V * fn(np.maximum(w, w[..., -1:] * 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)


def logm(C):
    return _eig_fn(C, np.log)


def expm(C):
    w, V = np.linalg.eigh(C)
    return (V * np.exp(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def invsqrtm(C):
    return _eig_fn(C, lambda w: 1.0 / np.sqrt(w))


def log_euclid_mean(covs):
    return expm(logm(covs).mean(axis=0))


def upper(M):
    n = M.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return M[..., iu[0], iu[1]] * coef


def recenter(covs, mode="recording", tau_s=60.0, step_s=0.5, warmup_s=20.0, ref_mask=None):
    """covs: (n, ch, ch) time-ordered windows of ONE recording (single band)."""
    covs = covs.astype(np.float64)
    if mode == "none":
        return covs
    if mode == "recording":
        src = covs if ref_mask is None else covs[ref_mask]
        P = invsqrtm(log_euclid_mean(src))
        return P @ covs @ P
    if mode == "causal":
        # exponentially-weighted log-Euclidean mean of PAST windows (strictly before the current one)
        return CausalRecenter(tau_s, step_s, warmup_s).run(covs)
    raise ValueError(mode)


class CausalRecenter:
    """Online whitening by an exponentially-weighted log-Euclidean mean of PAST windows.
    During the first ``warmup_s`` the reference is the plain mean of the windows seen so far."""

    def __init__(self, tau_s=60.0, step_s=0.5, warmup_s=20.0):
        self.a = 1.0 - np.exp(-step_s / tau_s)
        self.n_warm = max(1, int(warmup_s / step_s))
        self.m, self.k = None, 0

    def step(self, C):
        L = logm(C)
        ref = L if self.m is None else self.m
        P = invsqrtm(expm(ref))
        self.k += 1
        if self.m is None:
            self.m = L
        elif self.k <= self.n_warm:
            self.m = self.m + (L - self.m) / self.k
        else:
            self.m = (1 - self.a) * self.m + self.a * L
        return P @ C @ P

    def run(self, covs):
        return np.stack([self.step(c) for c in covs])


def tangent(covs_aligned):
    return upper(logm(covs_aligned))


def detrend(F, t, mode="none", degree=2, tau_s=120.0, step_s=0.5):
    """F: (n, d) time-ordered features of ONE recording, t: (n,) seconds."""
    if mode == "none":
        return F
    if mode == "poly":
        tt = (t - t.mean()) / (t.std() + 1e-9)
        A = np.vander(tt, degree + 1)
        coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        return F - A @ coef
    if mode == "causal":
        a = 1.0 - np.exp(-step_s / tau_s)
        n_warm = max(1, int(20 / step_s))
        out, m = np.empty_like(F), F[0].copy()
        for i in range(len(F)):                     # mean of PAST windows only
            out[i] = F[i] - m
            m = m + (F[i] - m) / (i + 1) if i + 1 < n_warm else (1 - a) * m + a * F[i]
        return out
    raise ValueError(mode)


def reference_reset_s(meta):
    """Time at which the causal reference must restart: scenario 1 opens with 120 s of eyes closed, which
    must never feed the reference of the eyes-open data (QA finding C1)."""
    if "scenario" in meta and len(meta) and int(meta["scenario"].iloc[0]) == 1:
        return 120.0
    return None


def recording_features(W, bands="fb", align="recording", tau_s=60.0, detrend_mode="none", detrend_tau=120.0,
                       channels=None, kind="ts"):
    """Feature matrix for one recording's windows dict (from windows.build_windows).

    bands: 'fb' = filter bank (5 bands), 'mi' = single 8-30 Hz band, or list of band indices.
    kind:  'ts' tangent space, 'logbp' log band-power (diagonal only, recentred), 'ts+time' adds nothing
    """
    covs = W["covs"]
    idx = {"fb": list(range(covs.shape[1] - 1)), "mi": [covs.shape[1] - 1]}[bands] if isinstance(bands, str) else bands
    t = W["meta"]["t_end"].to_numpy()
    feats = []
    for b in idx:
        Cb = covs[:, b]
        if channels is not None:
            Cb = Cb[:, channels][:, :, channels]
        reset = reference_reset_s(W["meta"]) if align == "causal" else None
        if reset is None:
            A = recenter(Cb, align, tau_s=tau_s)
        else:
            after = W["meta"]["t_start"].to_numpy() >= reset
            A = np.empty(Cb.shape)
            if (~after).any():
                A[~after] = recenter(Cb[~after], align, tau_s=tau_s)
            if after.any():
                A[after] = recenter(Cb[after], align, tau_s=tau_s)
        if kind == "ts":
            feats.append(tangent(A))
        elif kind == "logbp":
            feats.append(np.log(np.maximum(np.diagonal(A, axis1=1, axis2=2), 1e-12)))
    F = np.concatenate(feats, axis=1)
    return detrend(F, t, detrend_mode, tau_s=detrend_tau)
