"""CausalFilter: chunked streaming must equal one-shot filtering, and the filter must be causal."""
import numpy as np
import pytest

from eegbci import config as C
from eegbci.preprocessing import CausalFilter, causal_bandpass


@pytest.fixture(scope="module")
def x():
    rng = np.random.default_rng(3)
    return np.cumsum(rng.standard_normal((len(C.CHANNELS), 6000)), axis=1) * 1e-6 + 2e-5   # drifting, DC offset


@pytest.mark.parametrize("band", [(4, 8), (24, 30), (1, 40), (8, 30)])
def test_chunked_equals_one_shot(x, band):
    ref = causal_bandpass(x, *band)
    rng = np.random.default_rng(band[0])
    f = CausalFilter(*band, x.shape[0])
    parts, i = [], 0
    while i < x.shape[1]:
        n = int(rng.integers(1, 700))
        parts.append(f(x[:, i:i + n]))
        i += n
    out = np.concatenate(parts, axis=1)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_single_sample_chunks(x):
    ref = causal_bandpass(x[:, :400], 8, 12)
    f = CausalFilter(8, 12, x.shape[0])
    out = np.concatenate([f(x[:, i:i + 1]) for i in range(400)], axis=1)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_filter_is_causal(x):
    k = 3000
    y1 = causal_bandpass(x, 4, 8)
    x2 = x.copy()
    x2[:, k:] = 0.0
    y2 = causal_bandpass(x2, 4, 8)
    assert np.array_equal(y1[:, :k], y2[:, :k])


def test_initial_state_removes_dc_transient():
    """zi = steady state for the first sample: a pure DC input gives (almost) zero band-pass output."""
    dc = np.full((2, 2000), 5e-5)
    y = causal_bandpass(dc, 4, 8)
    assert np.abs(y).max() < 1e-12
