"""Scenario extraction for the PFE EEG recordings (NeuroVisor EDF exports).

The scenario of each file is detected from the markers embedded in the EDF
(not from the folder it sits in), and every recording is cut into labelled
segments.

    python scripts/audit/scenarios.py       # writes scripts/audit/scenario_manifest.csv
"""
import csv
import hashlib
import os
import re
from glob import glob

import mne

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("EEGBCI_DATA_DIR", os.path.join(ROOT, "data", "raw"))
LABELS = {0: "neutral", 1: "push", 2: "walk"}

# (start_s, stop_s, label, segment name). Scenario 1's first 240 s neutral block
# is split at the 120 s marker: 0-120 s is eyes-closed, 120-240 s eyes-open.
SCENARIOS = {
    1: [(0, 120, 0, "baseline-EC"), (120, 240, 0, "baseline-EO"), (240, 270, 1, "push#1"),
        (270, 290, 0, "rest#1"), (290, 320, 1, "push#2"), (320, 340, 0, "rest#2"),
        (340, 370, 2, "walk#1"), (370, 390, 0, "rest#3"), (390, 420, 2, "walk#2"), (420, 480, 0, "rest#4")],
    2: [(0, 80, 1, "push#1"), (80, 160, 2, "walk#1"), (160, 180, 0, "rest#1"), (180, 260, 1, "push#2"),
        (260, 340, 2, "walk#2"), (340, 360, 0, "rest#2"), (360, 430, 1, "push#3"), (430, 500, 2, "walk#3")],
}
MARKERS = {1: [0, 120, 240, 270, 290, 320, 340, 370, 390, 420], 2: [0, 80, 160, 180, 260, 340, 360, 430],
           0: [0, 120, 240, 360]}


def parse_name(fname):
    """'3_S05.edf' -> (subject 'S05', session tag '3')."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^(?P<tag>(?:test\d*|ft\d|\d+d?))?[-_ ]*(?P<subj>S(?:0[1-9]|10))$", base)
    return (m.group("subj"), m.group("tag") or "") if m else (base, "")


def detect_scenario(raw):
    onsets = [round(float(o)) for o, d in zip(raw.annotations.onset, raw.annotations.description) if d != "PAUSE"]
    for sc, marks in MARKERS.items():
        if onsets == marks:
            return sc
    return -1  # short test / free-run recording without a protocol


def build_manifest(data_dir=DATA_DIR):
    rows, seen = [], {}
    for path in sorted(glob(os.path.join(data_dir, "*.edf"))):
        fname = os.path.basename(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        duplicate_of = seen.setdefault(md5, fname)
        subj, tag = parse_name(fname)
        sc = detect_scenario(raw)
        dur = raw.n_times / raw.info["sfreq"]
        notes = []
        if duplicate_of != fname:
            notes.append(f"byte-identical copy of {duplicate_of}")
        if "PAUSE" in raw.annotations.description:
            notes.append("PAUSE marker at %.2f s" % raw.annotations.onset[list(raw.annotations.description).index("PAUSE")])
        if "BAD_ACQ_SKIP" in raw.annotations.description:
            notes.append("BAD_ACQ_SKIP at end")
        segs = SCENARIOS.get(sc, [(0, dur, -1, "unlabelled")])
        for t0, t1, lbl, name in segs:
            if t0 >= dur:
                notes_seg = "missing (recording too short)"
            elif t1 > dur:
                notes_seg = f"truncated to {dur:.0f} s"
            else:
                notes_seg = ""
            rows.append(dict(file=fname, subject=subj, session=tag, scenario=sc, segment=name,
                             label=lbl, label_name=LABELS.get(lbl, "unknown"), t_start=t0, t_end=min(t1, dur),
                             duration_file=round(dur, 2), duplicate_of="" if duplicate_of == fname else duplicate_of,
                             notes="; ".join(filter(None, notes + [notes_seg]))))
    return rows


def load_segments(path, l_freq=None, h_freq=None):
    """Yield (segment_name, label, data[channels, samples]) for one recording."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    if l_freq or h_freq:
        raw.filter(l_freq, h_freq, verbose=False)
    sc = detect_scenario(raw)
    dur = raw.n_times / raw.info["sfreq"]
    for t0, t1, lbl, name in SCENARIOS.get(sc, []):
        if t0 < dur:
            a, b = raw.time_as_index([t0, min(t1, dur)])
            yield name, lbl, raw.get_data(start=a, stop=b)


if __name__ == "__main__":
    rows = build_manifest()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_manifest.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} segments from {len({r['file'] for r in rows})} files -> {out}")
