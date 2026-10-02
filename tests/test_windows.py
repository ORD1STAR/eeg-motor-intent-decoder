"""Windowing (causality, grid) and protocol labelling (guard, circular shift, wrap-around, clipping)."""
import numpy as np
import pandas as pd
import pytest

from eegbci import config as C
from tests.helpers import FakeRec, make_signal, proposed_label_windows, ref_label, require, toy_meta
from eegbci.windows import DECIM, FS_D, build_windows, label_windows

D = 100.0
TOY = [(0, 20, None, "EC", 0), (20, 40, 0, "rest", 0), (40, 60, 1, "push", 0), (60, 80, 0, "rest2", 1),
       (80, 100, 2, "walk", 1)]


def ref_labels(meta, blocks, duration, shift, guard=C.GUARD_S):
    """Independent reference (new semantics): the circular shift acts on the analysed span [lo, duration) only
    (lo = onset of the first labelled block); windows overlapping a label-None block (eyes closed) are -2 in real
    time; windows starting before lo are never labelled."""
    lo = min(b[0] for b in blocks if b[2] is not None)
    span = duration - lo
    t_s, t_e = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    out = []
    for a, e in zip(t_s, t_e):
        if any(b[2] is None and e > b[0] and a < b[1] for b in blocks):
            out.append((-2, -1, -1))
        elif a < lo:
            out.append((-1, -1, -1))
        else:
            ts = lo + (a - lo - shift) % span
            out.append(ref_label(ts, ts + (e - a), blocks, duration, guard))
    return np.array(out).T


# ---------------------------------------------------------------- label_windows, no shift
def test_guard_at_block_edges():
    meta = toy_meta(D, win_s=2.0, hop_s=0.5)
    lab, blk, cyc, tib = label_windows(meta, TOY, D)
    by_start = dict(zip(meta["t_start"].round(3), lab))
    assert by_start[20.5] == -1 and by_start[21.0] == 0          # first usable window starts 1 s after onset
    assert by_start[37.0] == 0 and by_start[37.5] == -1          # last usable window ends 1 s before offset
    assert by_start[41.0] == 1 and by_start[38.0] == -1 and by_start[39.5] == -1   # windows straddling a change
    ok = lab >= 0
    np.testing.assert_allclose(tib[ok], meta["t_end"].to_numpy()[ok] - np.array([TOY[b][0] for b in blk[ok]]))
    assert np.all(cyc[ok] == np.array([TOY[b][4] for b in blk[ok]]))


def test_eyes_closed_baseline_is_excluded():
    meta = toy_meta(D, win_s=2.0, hop_s=0.5)
    lab, blk, cyc, _ = label_windows(meta, TOY, D)
    ec = meta["t_end"].to_numpy() <= 20
    assert np.all(lab[ec] < 0) and np.any(lab[ec] == -2)
    assert np.all(blk[lab == -2] == -1) and np.all(cyc[lab == -2] == -1)


def test_matches_independent_reference_without_shift():
    meta = toy_meta(D, win_s=4.0, hop_s=0.496)
    got = np.array(label_windows(meta, TOY, D)[:3])
    assert np.array_equal(got, ref_labels(meta, TOY, D, 0.0))


# ---------------------------------------------------------------- circular shift (drift-null)
@pytest.mark.parametrize("shift", [0.25, 13.0, 37.3, 50.0, 61.7, 99.5])
def test_shift_matches_independent_reference(shift):
    meta = toy_meta(D, win_s=4.0, hop_s=0.496)
    got = np.array(label_windows(meta, TOY, D, shift)[:3])     # labels, blocks AND cycles recomputed
    assert np.array_equal(got, ref_labels(meta, TOY, D, shift))


@pytest.mark.parametrize("shift", [10.0, 33.3, 71.1])
def test_shift_never_labels_windows_that_wrap(shift):
    meta = toy_meta(D, win_s=4.0, hop_s=0.25)
    lab = label_windows(meta, TOY, D, shift)[0]
    lo = 20.0
    ts = lo + (meta["t_start"].to_numpy() - lo - shift) % (D - lo)
    wraps = (ts + 4.0 > D) & (meta["t_start"].to_numpy() >= lo)
    require(wraps.any(), "some windows straddle the protocol seam")
    assert np.all(lab[wraps] == -1)


def test_full_period_shift_is_identity():
    meta = toy_meta(D, win_s=2.0, hop_s=0.5)
    a, b = label_windows(meta, TOY, D, 0.0), label_windows(meta, TOY, D, D - 20.0)   # period = analysed span
    for u, v in zip(a[:3], b[:3]):
        assert np.array_equal(u, v)


def test_shift_preserves_class_time_up_to_the_seam():
    meta = toy_meta(D, win_s=2.0, hop_s=0.5)
    n0 = np.bincount(label_windows(meta, TOY, D)[0] + 2, minlength=5)
    for s in (17.0, 44.5, 70.0):
        n1 = np.bincount(label_windows(meta, TOY, D, s)[0] + 2, minlength=5)
        assert np.all(np.abs(n1[2:] - n0[2:]) <= 2 * 4 / 0.5)     # at most the windows lost at the seam


def test_label_windows_equals_proposed_reference():
    meta = toy_meta(480.0, win_s=4.0, hop_s=0.496)
    for s in (0.0, 33.3, 150.0, 290.0, 359.9):
        a, b = label_windows(meta, C.PROTOCOLS[1], 480.0, s), proposed_label_windows(meta, C.PROTOCOLS[1], 480.0, s)
        for u, v in zip(a[:3], b[:3]):
            assert np.array_equal(u, v)
        np.testing.assert_allclose(a[3], b[3], equal_nan=True)


def test_shift_stays_in_analysed_span():
    """No window starting before the first labelled block (real time) is ever labelled, whatever the shift."""
    meta = toy_meta(D, win_s=2.0, hop_s=0.5)
    for s in (0.0, 7.0, 55.5):
        lab = label_windows(meta, TOY, D, s)[0]
        assert np.all(lab[meta["t_start"].to_numpy() < 20.0] < 0)


def test_real_eyes_closed_windows_stay_excluded_under_shift():
    blocks = C.PROTOCOLS[1]
    meta = toy_meta(480.0, win_s=4.0, hop_s=0.496)
    real_ec = meta["t_end"].to_numpy() <= 120.0
    for s in (100.0, 200.0, 300.0):
        lab = label_windows(meta, blocks, 480.0, s)[0]
        assert np.all(lab[real_ec] < 0), f"shift {s}: {np.mean(lab[real_ec] >= 0):.0%} of real EC windows labelled"


# ---------------------------------------------------------------- clipped recording (S07-2 is 440 s long)
def test_clipped_protocol_440s():
    dur = 440.0
    blocks = [(t0, min(t1, dur), l, n, c) for t0, t1, l, n, c in C.PROTOCOLS[1] if t0 < dur]   # as data.load_recordings
    assert blocks[-1][:2] == (420, 440.0)
    meta = toy_meta(dur, win_s=4.0, hop_s=0.496)
    for s in (0.0, 75.0, 333.0):
        lab, blk, *_ = label_windows(meta, blocks, dur, s)
        te = 120.0 + (meta["t_start"].to_numpy() - 120.0 - s) % (dur - 120.0) + 4.0
        assert np.all(te[lab >= 0] <= dur - C.GUARD_S)
    lab, blk, *_ = label_windows(meta, blocks, dur)
    assert (blk == len(blocks) - 1).sum() > 0                       # the clipped rest#4 is still usable


# ---------------------------------------------------------------- build_windows
def test_window_grid_and_end_index(W4, W2, synth):
    for W, win_s in ((W4, 4.0), (W2, 2.0)):
        t_end = W["meta"]["t_end"].to_numpy()
        assert np.allclose(np.diff(t_end), 62 / FS_D)                 # hop = 62 decimated samples = 0.496 s, not 0.5
        assert t_end[0] == win_s and np.allclose(W["meta"]["t_start"], t_end - win_s)
        assert t_end[-1] <= synth.duration
    n_dec = synth.data.shape[1] // DECIM
    assert len(W4["meta"]) == len(np.arange(500, n_dec + 1, 62))


def test_build_windows_is_causal(synth):
    x = synth.data
    k = 30_000                                                       # raw sample index (60 s)
    x2 = x.copy()
    x2[:, k:] = np.random.default_rng(1).standard_normal(x2[:, k:].shape) * 1e-4
    a = build_windows(FakeRec(data=x), win_s=4.0, interp_bads=False, cache=False)
    b = build_windows(FakeRec(data=x2), win_s=4.0, interp_bads=False, cache=False)
    ends = np.round(a["meta"]["t_end"].to_numpy() * FS_D).astype(int)
    past = ends * DECIM <= k                                          # every raw sample used is < k
    require(past.sum() > 50, "enough past windows")
    assert np.array_equal(a["covs"][past], b["covs"][past])
    assert np.array_equal(np.asarray(a["raw"])[past], np.asarray(b["raw"])[past])
    assert not np.array_equal(a["covs"][~past], b["covs"][~past])


def test_bad_channel_interpolation_is_causal():
    """Bad channels are detected on the first 60 s only: a channel that goes bad later cannot change earlier windows."""
    x = make_signal(80.0, seed=5)
    x2 = x.copy()
    x2[C.CHANNELS.index("T7"), 31_000:] *= 60.0                    # T7 becomes bad after 62 s
    a = build_windows(FakeRec(data=x), win_s=4.0, interp_bads=True, cache=False)
    b = build_windows(FakeRec(data=x2), win_s=4.0, interp_bads=True, cache=False)
    assert list(a["bads"]) == list(b["bads"]) == []
    early = a["meta"]["t_end"].to_numpy() <= 62.0
    require(early.sum() > 50, "enough early windows")
    assert np.array_equal(a["covs"][early], b["covs"][early])


def test_bad_channel_detected_in_prefix_is_interpolated_from_t0():
    x = make_signal(40.0, seed=5, bad_channel="T7")
    a = build_windows(FakeRec(data=x), win_s=4.0, interp_bads=True, cache=False)
    b = build_windows(FakeRec(data=x), win_s=4.0, interp_bads=False, cache=False)
    assert list(a["bads"]) == ["T7"]
    assert not np.array_equal(a["covs"][0], b["covs"][0])
