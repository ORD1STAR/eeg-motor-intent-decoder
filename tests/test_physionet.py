"""PhysioNet eegmmidb adapter: channel mapping and interpolation, resampling, trial blocks, cycles and the
shorter label guard. Synthetic Raw objects only; the real files are optional (skipped when absent)."""
import mne
import numpy as np
import pytest

from eegbci import config as C
from eegbci import physionet as P
from eegbci.evaluation import Bundle, random_shifts
from eegbci.windows import label_windows
from tests.helpers import FakeRec, toy_meta

mne.set_log_level("ERROR")

PHYSIONET_64 = ["Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..",
                "C6..", "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.",
                "Afz.", "Af4.", "Af8.", "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..", "Ft7.",
                "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..",
                "P4..", "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz.."]
TRIALS = ["T0", "T2", "T0", "T1", "T0", "T1", "T0", "T2", "T0", "T1"]      # the dataset's random order


def fake_physionet_raw(seconds=42.0, sfreq=160.0, seed=0):
    """64-channel Raw in PhysioNet naming with T0/T1/T2 trials of 4.2 / 4.1 s, like S001R06."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    x = rng.standard_normal((64, n)) * 10e-6
    t = np.arange(n) / sfreq
    x[PHYSIONET_64.index("C3..")] += 20e-6 * np.sin(2 * np.pi * 10 * t)        # a landmark to track through conversion
    raw = mne.io.RawArray(x, mne.create_info(PHYSIONET_64, sfreq, "eeg"), verbose=False)
    onsets = np.cumsum([0.0] + [4.2 if d == "T0" else 4.1 for d in TRIALS[:-1]])
    durs = [4.2 if d == "T0" else 4.1 for d in TRIALS]
    raw.set_annotations(mne.Annotations(onsets, durs, TRIALS))
    return raw


@pytest.fixture(scope="module")
def rec():
    return P.recording_from_raw(fake_physionet_raw(), "P001", 6)


def test_channels_are_the_headset_montage_in_order(rec):
    assert rec.ch_names == C.CHANNELS
    assert rec.data.shape[0] == len(C.CHANNELS)
    assert rec.sfreq == C.SFREQ and rec.rec_id == "P001-R06" and rec.scenario == P.SCENARIO


def test_resampled_to_500hz_and_signal_preserved(rec):
    assert rec.n_times == pytest.approx(42.0 * C.SFREQ, abs=C.SFREQ / 160 + 1)
    c3 = rec.data[C.CHANNELS.index("C3")]
    f = np.fft.rfftfreq(len(c3), 1 / C.SFREQ)
    peak = f[np.argmax(np.abs(np.fft.rfft(c3)))]
    assert peak == pytest.approx(10.0, abs=0.2)                        # the 10 Hz landmark survives resampling


def test_missing_electrodes_are_interpolated_not_zero(rec):
    for ch in P.MISSING:
        x = rec.data[C.CHANNELS.index(ch)]
        assert np.all(np.isfinite(x)) and x.std() > 0
        assert x.std() < 5 * rec.data.std()                                # and not an outlier channel


def test_blocks_follow_the_annotations(rec):
    assert len(rec.blocks) == len(TRIALS)
    labs = [b[2] for b in rec.blocks]
    assert labs == [P.LABELS[d] for d in TRIALS]
    assert rec.blocks[0][:2] == (0.0, 4.2) and rec.blocks[1][0] == pytest.approx(4.2)
    assert rec.blocks[-1][1] <= rec.duration + 1e-9
    assert [b[3] for b in rec.blocks][:4] == ["rest#1", "feet#1", "rest#2", "fists#1"]
    assert sorted({b[4] for b in rec.blocks}) == [0, 1, 2]              # three cycles for leave-one-cycle-out
    cyc = [b[4] for b in rec.blocks]
    assert cyc == sorted(cyc)                                              # consecutive trials share a cycle


def test_clipped_trial_at_the_end():
    blocks = P.blocks_from_annotations([0.0, 4.2], [4.2, 4.1], ["T0", "T1"], duration=6.0)
    assert blocks[-1][1] == 6.0 and blocks[-1][2] == C.PUSH


def test_half_second_guard_gives_windows_in_every_trial(rec):
    meta = toy_meta(rec.duration, win_s=P.WIN_S, hop_s=0.496)
    lab_1s = label_windows(meta, rec.blocks, rec.duration, guard=C.GUARD_S)[0]
    lab, blk, cyc, _ = label_windows(meta, rec.blocks, rec.duration, guard=P.GUARD_S)
    assert set(blk[lab >= 0]) == set(range(len(rec.blocks)))             # every trial yields at least one window
    assert (lab >= 0).sum() > (lab_1s >= 0).sum()
    te, ts = meta["t_end"].to_numpy(), meta["t_start"].to_numpy()
    for i, (t0, t1, *_) in enumerate(rec.blocks):
        assert np.all(ts[blk == i] >= t0 + P.GUARD_S - 1e-9) and np.all(te[blk == i] <= t1 - P.GUARD_S + 1e-9)


def test_bundle_passes_its_guard_to_the_labels(rec):
    meta = toy_meta(rec.duration, win_s=P.WIN_S, hop_s=0.496)
    fake = FakeRec(data=np.zeros((1, 10)), rec_id=rec.rec_id, subject=rec.subject, blocks=rec.blocks, scenario=P.SCENARIO)
    fake.data = np.zeros((1, int(rec.duration * C.SFREQ)))
    F = {rec.rec_id: np.zeros((len(meta), 2), np.float32)}
    tight = Bundle([fake], {rec.rec_id: dict(meta=meta)}, F, guard=P.GUARD_S)
    loose = Bundle([fake], {rec.rec_id: dict(meta=meta)}, F)
    assert loose.guard == C.GUARD_S
    assert tight.labels(rec.rec_id, "push_walk")[1].sum() > loose.labels(rec.rec_id, "push_walk")[1].sum()


def test_random_shifts_min_s_fits_short_recordings(rec):
    fake = FakeRec(data=np.zeros((1, int(rec.duration * C.SFREQ))), rec_id=rec.rec_id, blocks=rec.blocks)
    B = Bundle([fake], {}, {})
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = random_shifts(B, rng, min_s=10.0)[rec.rec_id]
        assert 10.0 <= s <= rec.duration - 10.0


def test_feature_configs_use_existing_channels():
    for cfgs in P.FEATURES.values():
        for c in cfgs:
            assert c.get("channels") is None or max(c["channels"]) < len(C.CHANNELS)


@pytest.mark.skipif(not list(P.DATA_DIR.rglob("S001R06.edf")), reason="PhysioNet files not downloaded")
def test_real_run_converts_to_the_headset_format():
    recs = [r for r in P.load_physionet(1, verbose=False) if r.session == "R06"]
    assert len(recs) == 1
    r = recs[0]
    assert r.ch_names == C.CHANNELS and r.duration == pytest.approx(125.0, abs=0.5)
    assert 28 <= len(r.blocks) <= 30 and {b[2] for b in r.blocks} == {0, 1, 2}
    assert r.data.shape == (len(C.CHANNELS), r.n_times)


def test_relative_artifact_limit_keeps_a_noisy_but_clean_recording(rec):
    meta = toy_meta(rec.duration, win_s=P.WIN_S, hop_s=0.496)
    meta["ptp_max"] = 200.0                                              # whole recording above 150 µV: amplifier level
    meta.loc[meta.index[::10], "ptp_max"] = 2000.0                       # every 10th window a real artifact
    fake = FakeRec(data=np.zeros((1, int(rec.duration * C.SFREQ))), rec_id=rec.rec_id, subject=rec.subject,
                   blocks=rec.blocks, scenario=P.SCENARIO)
    F = {rec.rec_id: np.zeros((len(meta), 2), np.float32)}
    fixed = Bundle([fake], {rec.rec_id: dict(meta=meta)}, F, guard=P.GUARD_S)
    rel = Bundle([fake], {rec.rec_id: dict(meta=meta)}, F, guard=P.GUARD_S, reject_uv=None, reject_rel=P.REJECT_REL)
    assert fixed.labels(rec.rec_id, "task_rest")[1].sum() == 0           # the fixed limit drops everything
    assert rel.artifact_limit(rec.rec_id) == pytest.approx(600.0)
    ok = rel.labels(rec.rec_id, "task_rest", reject_artifacts=False)[1]
    kept = rel.labels(rec.rec_id, "task_rest")[1]
    assert kept.sum() > 0 and not np.any(kept & (meta["ptp_max"].to_numpy() > 600)) and kept.sum() < ok.sum()
