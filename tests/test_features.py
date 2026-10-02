"""Causal re-centering, tangent space and detrending."""
import numpy as np
import pytest

from eegbci import features as Fm
from eegbci.features import (CausalRecenter, detrend, expm, invsqrtm, logm, recenter, recording_features, tangent,
                             upper)


def spd(n, ch=8, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, ch, 3 * ch))
    return scale * (A @ np.swapaxes(A, 1, 2) / (3 * ch) + 0.1 * np.eye(ch))


def naive_causal_recenter(covs, tau_s=60.0, step_s=0.5, warmup_s=20.0):
    """Reference: window k is whitened by the log-Euclidean mean of windows < k (cumulative mean during
    the warm-up, EMA afterwards); window 0 is whitened by itself."""
    a = 1.0 - np.exp(-step_s / tau_s)
    n_warm = max(1, int(warmup_s / step_s))
    L = np.stack([logm(c) for c in covs])
    out, m = [], None
    for k, c in enumerate(covs):
        ref = L[0] if k == 0 else m
        P = invsqrtm(expm(ref))
        out.append(P @ c @ P)
        if k == 0:
            m = L[0]
        elif k + 1 <= n_warm:
            m = m + (L[k] - m) / (k + 1)
        else:
            m = (1 - a) * m + a * L[k]
    return np.stack(out)


@pytest.mark.parametrize("kw", [dict(), dict(tau_s=20.0, step_s=0.5, warmup_s=5.0)])
def test_causal_recenter_run_equals_recenter_causal(kw):
    covs = spd(90, seed=1).astype(np.float32)          # cached covariances are float32
    a = recenter(covs, "causal", **kw)
    b = CausalRecenter(**(dict(tau_s=60.0, step_s=0.5, warmup_s=20.0) | kw)).run(covs.astype(np.float64))
    assert np.array_equal(a, b)


def test_causal_recenter_matches_naive_reference():
    covs = spd(80, seed=2)
    out = CausalRecenter(tau_s=10.0, step_s=0.5, warmup_s=5.0).run(covs)
    np.testing.assert_allclose(out, naive_causal_recenter(covs, 10.0, 0.5, 5.0), rtol=0, atol=1e-10)


def test_causal_recenter_reference_is_strictly_past():
    covs = spd(60, seed=4)
    out1 = CausalRecenter().run(covs)                  # warm-up = 40 windows
    for k in (1, 25, 40, 59):
        c2 = covs.copy()
        c2[k] = spd(1, seed=99)[0] * 7.0
        out2 = CausalRecenter().run(c2)
        assert np.array_equal(out1[:k], out2[:k])                       # past unaffected
        if k <= 40:   # during warm-up the reference at k is the plain log-Euclidean mean of windows 0..k-1
            P = invsqrtm(expm(np.mean([logm(c) for c in covs[:k]], axis=0)))
            np.testing.assert_allclose(out2[k], P @ c2[k] @ P, rtol=1e-9, atol=1e-12)


def test_first_window_is_whitened_by_itself():
    """Documented edge case: window 0 has no past, so it is whitened by itself (features = 0)."""
    out = CausalRecenter().run(spd(5, seed=5))
    np.testing.assert_allclose(out[0], np.eye(out.shape[1]), atol=1e-10)


def test_upper_preserves_frobenius_norm():
    M = logm(spd(1, seed=6)[0])
    assert np.isclose(np.linalg.norm(upper(M)), np.linalg.norm(M))


def test_logm_expm_roundtrip_unit_scale():
    C1 = spd(1, seed=7)[0]
    np.testing.assert_allclose(expm(logm(C1)), C1, rtol=1e-10, atol=1e-12)


def test_recentred_features_are_scale_invariant():
    covs = spd(60, seed=8)
    F = lambda c: np.c_[np.log(np.diagonal(recenter(c, "causal"), axis1=1, axis2=2)), tangent(recenter(c, "causal"))]
    f_unit = F(covs)
    f_volts = F(covs * 1e-13)                       # realistic scale: eigenvalues ~1e-14 .. 1e-12 V^2
    np.testing.assert_allclose(f_volts, f_unit, rtol=0, atol=1e-6)


def test_detrend_causal_is_causal():
    rng = np.random.default_rng(9)
    F = rng.standard_normal((100, 3))
    t = 0.5 * np.arange(100)
    out1 = detrend(F, t, "causal")
    F2 = F.copy()
    F2[30] += 100.0
    out2 = detrend(F2, t, "causal")
    assert np.array_equal(out1[:30], out2[:30])


# ---------------------------------------------------------------- reference reset (scenario 1: 120 s eyes closed)
def _fake_W(scenario, n=290, ch=6, nb=2, seed=3):
    import pandas as pd
    covs = np.stack([spd(n, ch=ch, seed=seed + b) for b in range(nb)], axis=1).astype(np.float32)
    te = 4.0 + 0.5 * np.arange(n)
    meta = pd.DataFrame(dict(t_end=te, t_start=te - 4.0, scenario=scenario))
    return dict(covs=covs, meta=meta)


def _slice_W(W, m):
    meta = W["meta"][m].reset_index(drop=True)
    meta["scenario"] = 2                      # a plain (no-reset) causal run on the slice
    return dict(covs=W["covs"][m], meta=meta)


@pytest.mark.parametrize("kind", ["ts", "logbp"])
def test_scenario1_reference_restarts_at_120s(kind):
    W = _fake_W(1)
    F = recording_features(W, bands=[0, 1], align="causal", kind=kind)
    after = W["meta"]["t_start"].to_numpy() >= 120.0
    n_after = after.sum()
    assert 10 < n_after < len(after) - 10
    np.testing.assert_allclose(F[~after], recording_features(_slice_W(W, ~after), bands=[0, 1], align="causal", kind=kind))
    np.testing.assert_allclose(F[after], recording_features(_slice_W(W, after), bands=[0, 1], align="causal", kind=kind))
    # the eyes-open part is independent of the eyes-closed part
    W2 = _fake_W(1)
    W2["covs"][~after] *= 50.0
    np.testing.assert_allclose(recording_features(W2, bands=[0, 1], align="causal", kind=kind)[after], F[after])


def test_scenario2_has_no_reset_and_other_modes_unaffected():
    W1, W2 = _fake_W(1), _fake_W(2)
    f2 = recording_features(W2, bands=[0, 1], align="causal")
    ref = np.concatenate([tangent(recenter(W2["covs"][:, b], "causal")) for b in (0, 1)], axis=1)
    np.testing.assert_allclose(f2, ref)
    assert not np.allclose(recording_features(W1, bands=[0, 1], align="causal"), f2)
    np.testing.assert_allclose(recording_features(W1, bands=[0, 1], align="recording"),
                               recording_features(W2, bands=[0, 1], align="recording"))
