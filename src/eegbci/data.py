"""Loading the NeuroVisor EDF recordings.

One source of truth (``EEGBCI_DATA_DIR``), duplicates removed by content hash, the protocol of each
file detected from its embedded markers, never from folder names.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import mne
import numpy as np

from . import config as C

mne.set_log_level("ERROR")


@dataclass
class Recording:
    rec_id: str          # "S05-3"
    subject: str         # "S05"
    session: str         # "3"
    scenario: int        # 1 or 2
    path: str
    n_times: int
    sfreq: float
    ch_names: list
    blocks: list = field(default_factory=list)   # [(t0, t1, label, name, cycle), ...] clipped to duration
    _data: np.ndarray = field(default=None, repr=False)

    @property
    def data(self):
        """(n_channels, n_samples) float64 volts, read from the EDF on first access."""
        if self._data is None:
            raw = mne.io.read_raw_edf(self.path, preload=True)
            self._data = raw.get_data()
        return self._data

    def release(self):
        self._data = None

    @property
    def duration(self):
        return self.n_times / self.sfreq


# Recordings are pseudonymised: <session-tag>_<subject-id>.edf, e.g. "3_S05.edf" or "S07.edf" (no tag = "0").
# A tag is a session number ("1", "2", ...), optionally with a "d" suffix (byte-identical duplicate), or "test"/"ftN".
_NAME_RE = re.compile(r"^(?P<tag>(?:test\d*|ft\d|\d+d?))?[-_ ]*(?P<subj>S(?:0[1-9]|10))(?:_.*)?\.edf$")


def parse_name(fname: str):
    """'3_S05.edf' -> ('S05', '3');  'S07.edf' -> ('S07', '0');  anything else -> (None, '')."""
    m = _NAME_RE.match(fname)
    if not m:
        return None, ""
    return m.group("subj"), (m.group("tag") or "0")


def detect_scenario(raw) -> int:
    onsets = [round(float(o)) for o, d in zip(raw.annotations.onset, raw.annotations.description) if d != "PAUSE"]
    for sc, marks in C.PROTOCOL_MARKERS.items():
        if onsets == marks:
            return sc
    return -1


def load_recordings(scenarios=(1, 2), subjects=None, verbose=True) -> list[Recording]:
    seen, recs = set(), []
    for path in sorted(C.DATA_DIR.glob("*.edf")):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        subj, tag = parse_name(path.name)
        if subj is None or (subjects and subj not in subjects):
            continue
        raw = mne.io.read_raw_edf(path, preload=False)
        sc = detect_scenario(raw)
        if sc not in scenarios:
            continue
        raw.rename_channels(lambda ch: ch.replace("-TP", ""))
        assert raw.ch_names == C.CHANNELS, raw.ch_names
        dur = raw.n_times / raw.info["sfreq"]
        blocks = [(t0, min(t1, dur), lbl, name, cyc) for t0, t1, lbl, name, cyc in C.PROTOCOLS[sc] if t0 < dur]
        recs.append(Recording(rec_id=f"{subj}-{tag}", subject=subj, session=tag, scenario=sc, path=str(path),
                              n_times=raw.n_times, sfreq=raw.info["sfreq"], ch_names=list(raw.ch_names), blocks=blocks))
    if verbose:
        n1 = sum(r.scenario == 1 for r in recs)
        print(f"loaded {len(recs)} recordings ({n1} scenario 1, {len(recs) - n1} scenario 2) "
              f"from {len({r.subject for r in recs})} subjects")
    return recs
