"""Public-dataset check: PhysioNet EEG Motor Movement/Imagery (eegmmidb) through the same pipeline.

    python -m eegbci.physionet download [n_subjects]   # fetch the imagery runs (MNE downloader, ~2 MB per run)
    python -m eegbci.physionet run [n_subjects]        # windows -> features -> LOSO with nulls and time-only control
    python -m eegbci.physionet summary                 # print the table from results/physionet_loso.json

Why this dataset. Runs 6, 10 and 14 are motor imagery of *both fists* (T1) and *both feet* (T2) separated by
rest (T0), in **randomised order**. That is the closest public analogue of push (hands) / walk (feet) / idle, and
the random order is exactly what the project's own protocols lack: the time-only control is expected at chance here,
so a push-vs-walk score that beats the drift-null is a motor-imagery signature, not elapsed time.

What is converted, and what is not. The 64-channel 160 Hz recordings are resampled to 500 Hz (a one-off, offline
conversion of the raw file, not part of the causal path), the two headset electrodes that PhysioNet lacks (FT9, FT10)
are spline-interpolated from the 64 channels with the same MNE routine the project uses for bad channels, and the
30 channels of ``config.CHANNELS`` are kept in the headset order. Trials last about 4.1 s, so the label guard is
0.5 s instead of 1 s and the windows are 2 s long; the 8 s probability smoothing of the brain switch is not applied
(it would smear across trials).

Labels: T0 -> idle (0), T1 both fists -> push (1), T2 both feet -> walk (2). Scenario id 3.

Artifact rule. The headset's fixed 150 uV peak-to-peak limit removed about 95 % of the PhysioNet windows (several
subjects sit near 200 uV on every channel with this amplifier), which left recordings with a single class. Here a
window is rejected above 3 x its recording's median peak-to-peak instead (``REJECT_REL``).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C

SCENARIO = 3                                    # distinct from the two headset protocols
RUNS = (6, 10, 14)                              # imagined both fists (T1) / both feet (T2), rest (T0)
LABELS = {"T0": C.NEUTRAL, "T1": C.PUSH, "T2": C.WALK}
NAMES = {C.NEUTRAL: "rest", C.PUSH: "fists", C.WALK: "feet"}
GUARD_S = 0.5                                   # trials are ~4.1 s: 1 s on each side would leave no 2 s window
WIN_S = 2.0
N_CYCLES = 3                                    # trials of a run are grouped into 3 cycles for leave-one-cycle-out
MISSING = [c for c in C.CHANNELS if c in ("FT9", "FT10")]   # headset electrodes absent from the 64-channel cap
DATA_DIR = C.DATA_DIR / "physionet"
RESULT = C.RESULTS_DIR / "physionet_loso.json"


@dataclass
class PublicRecording:
    """Same duck type as data.Recording, built from a PhysioNet run."""
    rec_id: str          # "P001-R06"
    subject: str         # "P001"
    session: str         # "R06"
    scenario: int
    path: str
    n_times: int
    sfreq: float
    ch_names: list
    blocks: list = field(default_factory=list)
    _data: np.ndarray = field(default=None, repr=False)

    @property
    def data(self):
        if self._data is None:
            self._data = convert_raw(_read(self.path)).get_data()
        return self._data

    def release(self):
        self._data = None

    @property
    def duration(self):
        return self.n_times / self.sfreq


def _read(path):
    import mne
    mne.set_log_level("ERROR")
    return mne.io.read_raw_edf(path, preload=True, verbose=False)


def convert_raw(raw, sfreq=C.SFREQ):
    """PhysioNet Raw (64 ch, 160 Hz, names like 'Fc5.') -> Raw with ``config.CHANNELS`` in headset order at
    ``sfreq``. Missing headset electrodes are spline-interpolated from the full cap."""
    import mne
    mne.set_log_level("ERROR")
    raw = raw.copy().load_data()
    mne.datasets.eegbci.standardize(raw)                          # 'Fc5.' -> 'FC5', 'Fp1.' -> 'Fp1'
    raw.set_montage("standard_1005", on_missing="ignore")
    missing = [c for c in C.CHANNELS if c not in raw.ch_names]
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], "eeg")
        pad = mne.io.RawArray(np.zeros((len(missing), raw.n_times)), info, verbose=False)
        raw.add_channels([pad], force_update_info=True)
        raw.set_montage("standard_1005", on_missing="ignore")
        raw.info["bads"] = missing
        raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.pick(C.CHANNELS)
    raw.reorder_channels(C.CHANNELS)
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, npad="auto", verbose=False)
    return raw


def blocks_from_annotations(onsets, durations, descriptions, duration, n_cycles=N_CYCLES):
    """(t0, t1, label, name, cycle) per trial, clipped to the recording. Trials are split into ``n_cycles``
    consecutive groups so that leave-one-cycle-out never splits a trial."""
    trials = [(float(o), min(float(o + d), duration), LABELS[desc]) for o, d, desc in zip(onsets, durations, descriptions)
              if desc in LABELS and o < duration]
    per = max(1, int(np.ceil(len(trials) / n_cycles)))
    counts = {l: 0 for l in LABELS.values()}
    blocks = []
    for k, (t0, t1, lab) in enumerate(trials):
        counts[lab] += 1
        blocks.append((t0, t1, lab, f"{NAMES[lab]}#{counts[lab]}", k // per))
    return blocks


def recording_from_raw(raw, subject, run, path="", sfreq=C.SFREQ):
    """Build a PublicRecording from an already loaded PhysioNet Raw (used by the tests without any download)."""
    conv = convert_raw(raw, sfreq)
    dur = conv.n_times / conv.info["sfreq"]
    a = raw.annotations
    blocks = blocks_from_annotations(a.onset, a.duration, a.description, dur)
    rec = PublicRecording(rec_id=f"{subject}-R{run:02d}", subject=subject, session=f"R{run:02d}", scenario=SCENARIO,
                          path=str(path), n_times=conv.n_times, sfreq=conv.info["sfreq"], ch_names=list(conv.ch_names),
                          blocks=blocks, _data=conv.get_data())
    return rec


def download(n_subjects=20, runs=RUNS, path=DATA_DIR):
    """Fetch the runs with MNE's downloader (PhysioNet, open licence). Returns the EDF paths per subject."""
    import mne
    mne.set_log_level("ERROR")
    path.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in range(1, n_subjects + 1):
        out[s] = mne.datasets.eegbci.load_data(s, list(runs), path=str(path), update_path=False, verbose=False)
    return out


def load_physionet(n_subjects=20, runs=RUNS, path=DATA_DIR, verbose=True):
    """Recordings for the first ``n_subjects`` participants (files must have been downloaded)."""
    files = sorted(path.rglob("S*R*.edf"))
    recs = []
    for f in files:
        subj, run = int(f.stem[1:4]), int(f.stem[5:7])
        if subj > n_subjects or run not in runs:
            continue
        raw = _read(f)
        conv_dur = raw.n_times / raw.info["sfreq"]
        a = raw.annotations
        blocks = blocks_from_annotations(a.onset, a.duration, a.description, conv_dur)
        recs.append(PublicRecording(rec_id=f"P{subj:03d}-R{run:02d}", subject=f"P{subj:03d}", session=f"R{run:02d}",
                                    scenario=SCENARIO, path=str(f), n_times=int(round(conv_dur * C.SFREQ)),
                                    sfreq=C.SFREQ, ch_names=list(C.CHANNELS), blocks=blocks))
    if verbose:
        print(f"loaded {len(recs)} PhysioNet runs from {len({r.subject for r in recs})} subjects")
    return recs


# ---------------------------------------------------------------- evaluation
PO = [C.CHANNELS.index(c) for c in ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]]
MOTOR = [C.CHANNELS.index(c) for c in C.MOTOR]
FEATURES = {
    # the project's fusion model (brain switch): log band power everywhere + tangent space over parieto-occipital
    "fusion": [dict(bands="fb", align="causal", tau_s=60, kind="logbp"),
               dict(bands=[1, 2, 3, 4], align="causal", tau_s=60, kind="ts", channels=PO)],
    # the textbook motor-imagery feature: tangent space over the motor strip, 8-30 Hz
    "ts-motor": [dict(bands="mi", align="causal", tau_s=60, kind="ts", channels=MOTOR)],
}
N_PERM = 20
REJECT_REL = 3.0        # BCI2000 amplifier: some subjects sit near 200 µV peak-to-peak everywhere, so the headset's
                        # fixed 150 µV limit dropped ~95 % of windows; reject at 3 x the recording's median instead
C_REG = 0.1


def build_bundle(recs, feat_cfgs, win_s=WIN_S):
    from .evaluation import Bundle
    from .features import recording_features
    from .windows import build_windows
    W = {r.rec_id: build_windows(r, win_s=win_s) for r in recs}
    for r in recs:
        r.release()
    lite = {rid: dict(covs=W[rid]["covs"], meta=W[rid]["meta"][["t_end", "t_start", "scenario"]]) for rid in W}
    F = {rid: np.concatenate([recording_features(lite[rid], **c) for c in feat_cfgs], axis=1).astype(np.float32)
         for rid in W}
    return Bundle(recs, {rid: dict(meta=W[rid]["meta"]) for rid in W}, F, guard=GUARD_S, reject_uv=None, reject_rel=REJECT_REL)


def run(n_subjects=20, n_perm=N_PERM, out=RESULT):
    from functools import partial
    from .evaluation import eval_groups, eval_within, loso_groups, null_distribution, report, time_features
    recs = load_physionet(n_subjects)
    if not recs:
        raise SystemExit(f"no PhysioNet files under {DATA_DIR}; run: python -m eegbci.physionet download")
    results = {"n_subjects": len({r.subject for r in recs}), "n_runs": len(recs), "rows": []}
    t = time.time()
    for fname, cfgs in FEATURES.items():
        B = build_bundle(recs, cfgs)
        print(f"[{fname}] features in {time.time() - t:.0f}s", flush=True)
        for scheme in ("task_rest", "push_walk"):
            for split in ("loso", "within"):
                if split == "loso":
                    fn = partial(eval_groups, B, scheme, loso_groups(B), "lr", C_reg=C_REG, smooth_s=0.0)
                else:
                    fn = partial(eval_within, B, scheme, "lr", C_reg=C_REG)
                key = "auc"
                obs = fn(shifts=None)
                null = null_distribution(lambda s: fn(shifts=s), B, n_perm=n_perm, key=key, min_s=10.0) if n_perm else None
                s = report(f"PhysioNet {fname} | {scheme} | {split}", obs, null, key)
                results["rows"].append(dict(features=fname, scheme=scheme, split=split, **s,
                                            per_rec={k: v.get("auc") for k, v in obs.items()}))
        for scheme in ("task_rest", "push_walk") if fname == next(iter(FEATURES)) else ():   # timing-only: once
            fn = partial(eval_groups, B, scheme, loso_groups(B), "lr", C_reg=C_REG, feats=time_features(B), smooth_s=0.0)
            s = report(f"PhysioNet time-only | {scheme} | loso", fn(shifts=None), None, "auc")
            results["rows"].append(dict(features="time-only", scheme=scheme, split="loso", **s))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"saved {out}")
    return results


def summary(path=RESULT):
    r = json.load(open(path))
    print(f"PhysioNet eegmmidb, {r['n_subjects']} subjects, {r['n_runs']} runs (imagery, randomised order)")
    print(f"{'features':<10} {'scheme':<10} {'split':<7} {'AUC':>6} {'null':>14} {'p':>6}")
    for row in r["rows"]:
        null = f"{row['null_mean']:.3f}+-{row['null_std']:.3f}" if "null_mean" in row else "-"
        p = f"{row['p']:.3f}" if "p" in row else "-"
        print(f"{row['features']:<10} {row['scheme']:<10} {row['split']:<7} {row['mean']:6.3f} {null:>14} {p:>6}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "run"
    if cmd == "download":
        n = int(argv[1]) if len(argv) > 1 else 20
        files = download(n)
        print(f"{sum(len(v) for v in files.values())} files under {DATA_DIR}")
    elif cmd == "run":
        run(int(argv[1]) if len(argv) > 1 else 20)
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
