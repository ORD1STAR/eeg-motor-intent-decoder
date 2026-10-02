"""Scenario detection, file-name parsing, de-duplication and block clipping (real files read-only;
skipped when the EDF data dir (EEGBCI_DATA_DIR) or the cache is absent). No participant name is written here."""
import re

import mne
import numpy as np
import pandas as pd
import pytest

from eegbci import config as C
from eegbci.data import detect_scenario, load_recordings, parse_name
from eegbci.windows import _params_key, label_windows

HAS_DATA = C.DATA_DIR.exists() and any(C.DATA_DIR.glob("*.edf"))


def _raw_with(onsets, desc=None):
    raw = mne.io.RawArray(np.zeros((1, 10 * 520)), mne.create_info(["Cz"], 10.0, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), desc or ["EO"] * len(onsets)))
    return raw


def test_detect_scenario():
    assert detect_scenario(_raw_with(C.PROTOCOL_MARKERS[1])) == 1
    assert detect_scenario(_raw_with(C.PROTOCOL_MARKERS[2])) == 2
    assert detect_scenario(_raw_with(C.PROTOCOL_MARKERS[2][:-1])) == -1
    assert detect_scenario(_raw_with([m + 0.4 for m in C.PROTOCOL_MARKERS[1]])) == 1     # rounding tolerance
    assert detect_scenario(_raw_with([m + 0.6 for m in C.PROTOCOL_MARKERS[1]])) == -1


def test_pause_annotation_is_silently_ignored():
    """Documents current behaviour (S10-0 has a PAUSE at 148.56 s): the recording is accepted and the nominal
    timeline is assumed after the pause."""
    on = sorted(C.PROTOCOL_MARKERS[1] + [148.56])
    assert detect_scenario(_raw_with(on, ["PAUSE" if o == 148.56 else "EO" for o in on])) == 1


@pytest.mark.skipif(not HAS_DATA, reason="the EDF data dir (EEGBCI_DATA_DIR) not available")
def test_parse_name_gives_pseudonymised_unique_ids():
    ids = [parse_name(p.name) for p in sorted(C.DATA_DIR.glob("*.edf"))]
    assert all(s is None or re.fullmatch(r"S(0[1-9]|10)", s) for s, _ in ids)
    named = [x for x in ids if x[0]]
    assert len(named) == len(set(named))          # rec_id is the cache key: a collision would mix recordings


@pytest.fixture(scope="module")
def recs():
    if not HAS_DATA:
        pytest.skip("the EDF data dir (EEGBCI_DATA_DIR) not available")
    return load_recordings(subjects=["S05", "S07"], verbose=False)      # md5 of every file, headers of 2 subjects


def test_dedup_and_scenario_filter(recs):
    ids = [r.rec_id for r in recs]
    assert len(ids) == len(set(ids))
    assert sorted(r.session for r in recs if r.subject == "S05") == ["3", "4", "5", "6"]    # 5d/6d = byte copies
    assert sorted(r.session for r in recs if r.subject == "S07") == ["2", "3", "4", "5"]    # S07-0 has no protocol
    assert {r.scenario for r in recs} <= {1, 2}


def test_blocks_are_clipped_to_the_recording(recs):
    r = next(r for r in recs if r.rec_id == "S07-2")
    assert r.duration == pytest.approx(440.0, abs=0.01)
    assert r.blocks[-1][3] == "rest#4" and r.blocks[-1][1] == pytest.approx(r.duration)
    assert all(b[1] <= q.duration for q in recs for b in q.blocks)


def test_real_cache_labels_440s_recording(recs):
    key = _params_key(dict(reref="none", interp=True, win=4.0, step=C.STEP_S, bands=C.BANDS, raw_band=(1, 40), v=4))
    path = C.CACHE_DIR / f"S07-2_{key}.npz"
    if not path.exists():
        pytest.skip("cache for S07-2 not built")
    meta = pd.DataFrame(np.load(path, allow_pickle=True)["meta"].tolist())        # read-only
    r = next(r for r in recs if r.rec_id == "S07-2")
    assert np.allclose(np.diff(meta["t_end"]), 0.496)
    for s in (0.0, 123.4, 300.0):
        lab, blk, cyc, _ = label_windows(meta, r.blocks, r.duration, s)
        te = (meta["t_start"].to_numpy() - s) % r.duration + 4.0
        assert np.all(te[lab >= 0] <= r.duration - C.GUARD_S)
        assert set(np.unique(cyc[lab >= 0])) <= {0, 1}
    lab = label_windows(meta, r.blocks, r.duration)[0]
    assert np.all(lab[meta["t_end"].to_numpy() <= 120] < 0)                      # eyes-closed baseline excluded
    assert (lab == 0).sum() > (lab > 0).sum() > 0
