"""Validates the reference implementations of the patches proposed in results/code_review.md."""
import numpy as np
import pytest

from eegbci import config as C
from eegbci.realtime import offline_features
from tests.helpers import (feed, make_proposed_decoder_class, proposed_detections, proposed_detrend_causal,
                                  proposed_label_windows, toy_meta)
from eegbci.windows import label_windows


def test_proposed_labels_equal_current_without_shift():
    meta = toy_meta(480.0, win_s=4.0, hop_s=0.496)
    a, b = label_windows(meta, C.PROTOCOLS[1], 480.0), proposed_label_windows(meta, C.PROTOCOLS[1], 480.0)
    assert np.array_equal(np.maximum(a[0], -1), np.maximum(b[0], -1))     # usable windows identical (-1/-2 both excluded)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


@pytest.mark.parametrize("s", [60.0, 150.0, 290.0])
def test_proposed_labels_keep_real_ec_excluded(s):
    meta = toy_meta(480.0, win_s=4.0, hop_s=0.496)
    lab = proposed_label_windows(meta, C.PROTOCOLS[1], 480.0, s)[0]
    assert np.all(lab[meta["t_end"].to_numpy() <= 120] < 0)
    assert (lab >= 0).sum() > 400


def test_proposed_detrend_is_causal():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((100, 3)); F2 = F.copy(); F2[30] += 100
    assert np.array_equal(proposed_detrend_causal(F)[:30], proposed_detrend_causal(F2)[:30])


def test_proposed_detection_needs_rising_edge():
    from eegbci.realtime import Output
    from tests.helpers import FakeRec
    rec = FakeRec(data=np.zeros((1, 500 * 80)), blocks=[(0, 40, 0, "r", 0), (40, 80, 1, "p", 0)])
    t = 4.0 + 0.5 * np.arange(153)
    mk = lambda act: [Output(t=a, p_raw=.5, p=.5, active=bool(b), artifact=False) for a, b in zip(t, act)]
    assert proposed_detections(rec, mk(np.ones(len(t))))[:2] == (1, 0)
    n, d, lats = proposed_detections(rec, mk(t >= 46.0))
    assert (n, d) == (1, 1) and lats[0] == pytest.approx(6.0)


def test_proposed_decoder_handles_long_chunks(synth, W4, riemann_bundle):
    Dec = make_proposed_decoder_class()
    p_off = riemann_bundle["model"].predict_proba(offline_features(W4))[:, 1]
    outs = feed(Dec(riemann_bundle, use_deep=False), synth.data, [3000, 7, 5000])
    np.testing.assert_allclose([o.p_riemann for o in outs], p_off, rtol=0, atol=1e-10)
