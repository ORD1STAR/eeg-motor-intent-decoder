"""Streaming decoder vs offline pipeline: Riemann parity under random chunking, decimation phase,
artifact flag, smoothing, adaptive threshold parity, deep-branch alignment, online metrics."""
import numpy as np
import pytest
import torch

from eegbci import config as C
from eegbci import deep as Dm
from eegbci.deep import ShallowNet, causal_euclidean_align, predict
from eegbci.ensemble import adaptive_decision
from eegbci.evaluation import ema_causal, make_model
from eegbci.preprocessing import causal_bandpass
from eegbci.realtime import FINAL_CFG, BrainSwitchDecoder, Output, offline_features, online_metrics
from tests.helpers import (FakeRec, ScriptedModel, feed, make_signal, proposed_detections,
                                  require, rt_rule)
from eegbci.windows import DECIM, FS_D, build_windows

SAFE_MAX_CHUNK = 996        # raw samples; see test_large_chunks_overrun_the_ring_buffer


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ShallowNet(n_t=250).eval()


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(Dm, "DEV", torch.device("cpu"))       # predict() must run where the decoder runs


def offline_deep(net, raw_uV):
    return predict(net, causal_euclidean_align(np.asarray(raw_uV) / 10.0))[:, 1]


def nearest_2s(t2, t4):
    """ensemble.py lines 67-68, verbatim logic."""
    j = np.clip(np.searchsorted(t2, t4), 1, len(t2) - 1)
    return np.where(np.abs(t2[j - 1] - t4) < np.abs(t2[j] - t4), j - 1, j)


# ---------------------------------------------------------------- Riemann branch parity
@pytest.mark.parametrize("seed", [0, 1])
def test_riemann_parity_random_chunks(synth, W4, riemann_bundle, seed):
    sizes = np.random.default_rng(seed).integers(1, SAFE_MAX_CHUNK + 1, size=500)
    outs = feed(BrainSwitchDecoder(riemann_bundle, use_deep=False), synth.data, sizes)
    p_off = riemann_bundle["model"].predict_proba(offline_features(W4))[:, 1]
    assert len(outs) == len(p_off)
    np.testing.assert_allclose([o.p_riemann for o in outs], p_off, rtol=0, atol=1e-10)
    np.testing.assert_allclose([o.t for o in outs], W4["meta"]["t_end"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose([o.p for o in outs], ema_causal(p_off[:, None], 8.0)[:, 0], rtol=0, atol=1e-10)
    require(np.ptp(p_off) > 0.3, "probabilities vary enough for the comparison to be meaningful")


def test_decimation_phase_with_odd_chunk_sizes(synth, W4, riemann_bundle):
    outs = feed(BrainSwitchDecoder(riemann_bundle, use_deep=False), synth.data, [3, 7, 5, 1, 9, 250, 11, 2])
    p_off = riemann_bundle["model"].predict_proba(offline_features(W4))[:, 1]
    np.testing.assert_allclose([o.p_riemann for o in outs], p_off, rtol=0, atol=1e-10)


def test_artifact_flag_matches_offline_ptp(synth, W4, riemann_bundle):
    outs = feed(BrainSwitchDecoder(riemann_bundle, use_deep=False), synth.data, [50])
    flag_off = W4["meta"]["ptp_max"].to_numpy() > C.PTP_REJECT_UV
    require(flag_off.any(), "the injected spike produces artifact windows")
    assert [o.artifact for o in outs] == list(flag_off)


def test_large_chunks_do_not_overrun_the_ring_buffer(synth, W4, riemann_bundle):
    outs = feed(BrainSwitchDecoder(riemann_bundle, use_deep=False), synth.data, [3000])
    p_off = riemann_bundle["model"].predict_proba(offline_features(W4))[:, 1]
    np.testing.assert_allclose([o.p_riemann for o in outs], p_off, rtol=0, atol=1e-10)


def test_parity_with_a_bad_channel():
    rec = FakeRec(data=make_signal(48.0, seed=7, bad_channel="T7"))
    W = build_windows(rec, win_s=4.0, interp_bads=True, cache=False)
    require("T7" in W["bads"], "T7 detected as bad offline")
    F = offline_features(W)
    bundle = dict(model=make_model("lr", 1.0).fit(F, (np.arange(len(F)) // 24) % 2), cfg=FINAL_CFG)
    outs = feed(BrainSwitchDecoder(bundle, use_deep=False, bads=W["bads"]), rec.data, [50])
    np.testing.assert_allclose([o.p_riemann for o in outs], bundle["model"].predict_proba(F)[:, 1], atol=1e-6)
    # without the calibration-time bads the decoder is NOT in parity (interpolation matters)
    raw = feed(BrainSwitchDecoder(bundle, use_deep=False), rec.data, [50])
    assert not np.allclose([o.p_riemann for o in raw], [o.p_riemann for o in outs], atol=1e-6)


def test_reset_at_s_matches_scenario1_offline_reference_restart():
    rec = FakeRec(data=make_signal(135.0, seed=3), scenario=1)
    W = build_windows(rec, win_s=4.0, interp_bads=False, cache=False)
    F = offline_features(W)
    t_end = W["meta"]["t_end"].to_numpy()
    bundle = dict(model=make_model("lr", 1.0).fit(F, (np.arange(len(F)) // 24) % 2), cfg=FINAL_CFG)
    p_off = bundle["model"].predict_proba(F)[:, 1]
    outs = feed(BrainSwitchDecoder(bundle, use_deep=False, reset_at_s=120.0), rec.data, [333])
    np.testing.assert_allclose([o.p_riemann for o in outs], p_off, rtol=0, atol=1e-8)
    # and it differs from a decoder that never restarts, only after the reset
    noreset = feed(BrainSwitchDecoder(bundle, use_deep=False), rec.data, [333])
    d = np.abs(np.array([o.p_riemann for o in noreset]) - p_off)
    after = t_end - 4.0 >= 120.0
    require(after.sum() > 10, "windows after the reset")
    assert d[~after].max() < 1e-8 and d[after].max() > 1e-6


# ---------------------------------------------------------------- decision rule
def scripted_outs(synth, **kw):
    rng = np.random.default_rng(5)
    seq = np.clip(0.5 + 0.25 * np.sin(np.arange(400) / 9.0) + 0.15 * rng.standard_normal(400), 0, 1)
    outs = feed(BrainSwitchDecoder(dict(model=ScriptedModel(seq), cfg=FINAL_CFG), use_deep=False, **kw), synth.data, [500])
    return seq[:len(outs)], outs


def test_adaptive_threshold_parity_with_ensemble_rule(synth):
    """margin=0 and min-on = 1 window reproduce ensemble.adaptive_decision exactly (the rule behind bacc_adaptive)."""
    seq, outs = scripted_outs(synth, margin=0.0, min_on_s=0.5)
    p = np.array([o.p for o in outs])
    np.testing.assert_allclose(p, ema_causal(seq[:, None], 8.0)[:, 0], rtol=0, atol=1e-12)
    assert np.array_equal(np.array([o.active for o in outs]), adaptive_decision(p))
    np.testing.assert_allclose([o.thr for o in outs], rt_rule(p, margin=0.0, min_on=1)[1], rtol=0, atol=1e-15)
    require(len(p) > 45, "warm-up (40 windows) and EMA phases are both exercised")


def test_default_online_rule_matches_replica(synth):
    """Validates helpers.rt_rule (used in the review to quantify margin=0.02 / min-on=2 vs adaptive_decision)."""
    _, outs = scripted_outs(synth)
    p = np.array([o.p for o in outs])
    act, thr = rt_rule(p)
    assert np.array_equal(np.array([o.active for o in outs]), act)
    np.testing.assert_allclose([o.thr for o in outs], thr, rtol=0, atol=1e-15)


def test_adaptive_decision_is_causal():
    rng = np.random.default_rng(1)
    p = rng.random(300)
    out = adaptive_decision(p)
    assert not out[0]
    for k in (5, 41, 200):
        p2 = p.copy()
        p2[k:] = rng.random(300 - k)
        assert np.array_equal(adaptive_decision(p2)[:k], out[:k])


# ---------------------------------------------------------------- deep branch
def test_ensemble_grid_matching_is_causal(W2, W4):
    t4, t2 = W4["meta"]["t_end"].to_numpy(), W2["meta"]["t_end"].to_numpy()
    j = nearest_2s(t2, t4)
    assert np.all(t2[j] <= t4)
    np.testing.assert_allclose(t4 - t2[j], 2 / FS_D)            # 2 decimated samples = 16 ms earlier
    assert np.array_equal(j, np.arange(len(t4)) + 4)


def test_deep_branch_parity_on_a_matching_grid(synth, W2, net, cpu):
    """With 2 s Riemann windows the online deep input IS the offline 2 s window and EA runs on the same grid."""
    F2 = offline_features(W2)
    lr2 = make_model("lr", 1.0).fit(F2, (np.arange(len(F2)) // 30) % 2)
    bundle = dict(model=lr2, cfg=dict(FINAL_CFG, win_s=2.0), net=dict(state=net.state_dict(), n_t=250))
    outs = feed(BrainSwitchDecoder(bundle), synth.data, [50])
    p_deep = offline_deep(net, W2["raw"])
    np.testing.assert_allclose([o.p_deep for o in outs], p_deep, rtol=0, atol=1e-5)
    np.testing.assert_allclose([o.p_raw for o in outs], 0.5 * (lr2.predict_proba(F2)[:, 1] + p_deep), rtol=0, atol=1e-5)


def test_online_deep_branch_equals_ensemble_offline_construction(synth, W2, W4, riemann_bundle, net, cpu):
    outs = feed(BrainSwitchDecoder(dict(riemann_bundle, net=dict(state=net.state_dict(), n_t=250))), synth.data, [50])
    p_sh = offline_deep(net, W2["raw"])[nearest_2s(W2["meta"]["t_end"].to_numpy(), W4["meta"]["t_end"].to_numpy())]
    np.testing.assert_allclose([o.p_deep for o in outs], p_sh, rtol=0, atol=1e-5)


def test_deep_branch_is_chunking_invariant(synth, riemann_bundle, net, cpu):
    bundle = dict(riemann_bundle, net=dict(state=net.state_dict(), n_t=250))
    a = feed(BrainSwitchDecoder(bundle), synth.data, [50])
    b = feed(BrainSwitchDecoder(bundle), synth.data, [3000, 7, 1234])          # long chunks: no ring overrun
    for other in (b,):
        assert len(other) == len(a)
        for f in ("p_deep", "p_riemann", "p_raw", "p"):
            np.testing.assert_allclose([getattr(o, f) for o in other], [getattr(o, f) for o in a], rtol=0, atol=1e-5)


# ---------------------------------------------------------------- online metrics
def _outs(t, active):
    return [Output(t=float(a), p_raw=0.5, p=float(b), active=bool(b), artifact=False) for a, b in zip(t, active)]


def test_online_metrics_detects_a_real_onset():
    rec = FakeRec(data=np.zeros((1, 500 * 80)), blocks=[(0, 40, 0, "rest#1", 0), (40, 80, 1, "push#1", 0)])
    t = 4.0 + 0.5 * np.arange(153)
    m = online_metrics(rec, _outs(t, t >= 46.0), 4.0)
    assert (m["onsets"], m["detected"]) == (1, 1) and m["latency_s"] == pytest.approx(6.0)
    assert m["false_on_per_min_rest"] == 0.0


def test_online_metrics_requires_a_rising_edge_after_onset():
    rec = FakeRec(data=np.zeros((1, 500 * 80)), blocks=[(0, 40, 0, "rest#1", 0), (40, 80, 1, "push#1", 0)])
    t = 4.0 + 0.5 * np.arange(153)
    m = online_metrics(rec, _outs(t, np.ones(len(t), bool)), 4.0)
    assert m["onsets"] == 1 and m["detected"] == 0


def test_online_metrics_agrees_with_proposed_detections():
    rec = FakeRec(data=np.zeros((1, 500 * 80)), blocks=[(0, 40, 0, "rest#1", 0), (40, 80, 1, "push#1", 0)])
    t = 4.0 + 0.5 * np.arange(153)
    for act in (t >= 46.0, np.ones(len(t), bool), t >= 20.0, np.zeros(len(t), bool)):
        outs = _outs(t, act)
        n, d, lats = proposed_detections(rec, outs)
        m = online_metrics(rec, outs, 4.0)
        assert (m["onsets"], m["detected"]) == (n, d)
        if lats:
            assert m["latency_s"] == pytest.approx(np.median(lats))
