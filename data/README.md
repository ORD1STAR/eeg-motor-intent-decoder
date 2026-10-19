# Data

**No recording is distributed with this repository.** The EEG was recorded from volunteers and is not public.
This page describes exactly what the code expects, so that you can run the pipeline on your own recordings made with
the same protocol, or adapt it to another dataset.

## Where the code looks

| Variable | Default | Purpose |
|---|---|---|
| `EEGBCI_DATA_DIR` | `data/raw/` | folder with the `*.edf` recordings (flat, no sub-folders) |
| `EEGBCI_CACHE_DIR` | `.cache/eegbci/` | windows / covariances / raw epochs, rebuilt by `build_cache` (several GB) |
| `EEGBCI_RESULTS_DIR` | `results/` | logs, json, figures written by every step |
| `EEGBCI_MODELS_DIR` | `models/` | the trained real-time model |

## Recording format

* EDF, **30 channels, 500 Hz**, signal in volts (the loader reads EDF through MNE; the real-time path accepts
  microvolts with a flag).
* Channel order (suffix `-TP` is stripped, the reference is the "TP" electrode):
  `Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 CP5 CP1 Pz P3 P7 O1 Oz O2 P4 P8 CP6 CP2 Cz C4 T8 FT10 FC6 FC2 F4 F8 Fp2`
  (`eegbci.config.CHANNELS`; the loader asserts it).
* Block changes are stored as EDF annotations. The protocol of a file is **detected from the annotation onsets**,
  not from its name (`eegbci.data.detect_scenario`).

## File naming (pseudonymised)

```
<session-tag>_<subject-id>.edf        e.g.  3_S05.edf   1_S01.edf   5d_S05.edf   S07.edf   test1_S05.edf
```

* `subject-id` is `S01` ... `S10`.
* `session-tag` is a session number, optionally with a `d` suffix for a byte-identical duplicate (dropped by
  content hash), or `test` / `ftN` for short free-run recordings (no protocol, skipped). No tag means session `0`.
* The original files carried participant names; they were replaced by IDs and **no mapping is stored in this
  repository**. `scripts/rename_edf.py` renames a folder from a CSV mapping that you provide yourself.
  EDF headers can also contain patient information: anonymise them before sharing a file.

## The two protocols

Each participant imagined a command ("push" or "walk") on cue, alternating with rest. Labels: `0` neutral/idle,
`1` push, `2` walk. Times in seconds from the start of the file.

**Scenario 1** (480 s, `PROTOCOL_MARKERS[1]`)

| 0-120 | 120-240 | 240-270 | 270-290 | 290-320 | 320-340 | 340-370 | 370-390 | 390-420 | 420-480 |
|---|---|---|---|---|---|---|---|---|---|
| eyes-closed baseline (excluded) | eyes-open baseline (idle) | push | rest | push | rest | walk | rest | walk | rest |

**Scenario 2** (500 s, `PROTOCOL_MARKERS[2]`)

| 0-80 | 80-160 | 160-180 | 180-260 | 260-340 | 340-360 | 360-430 | 430-500 |
|---|---|---|---|---|---|---|---|
| push | walk | rest | push | walk | rest | push | walk |

In both protocols push always precedes walk. This ordering is the reason push-versus-walk cannot be separated from
elapsed time (see `docs/evaluation.md`).

## Dataset used for the reported results

10 participants, 38 files, 28 usable recordings after de-duplication and protocol detection (12 scenario 1,
16 scenario 2), about 4.3 h of EEG in total. `scripts/audit/scenario_manifest.csv` lists every file, its detected
scenario and its segments (pseudonymised names).

## Legacy datasets

`experiments/baselines_2025/` (first-phase notebooks) read concatenated `.npz` training/test sets built by
`experiments/baselines_2025/extraction/`. They are not needed for the main pipeline; point
`EEGBCI_LEGACY_DATA_DIR` at them if you want to re-run those notebooks.

## Public alternatives

The pipeline is generic for 30-channel / 500 Hz EDF with the annotation scheme above. For a public check of the
Riemannian and deep components, adapt `config.py` (channels, protocols) to a public motor-imagery set such as
PhysioNet EEGBCI or BCI Competition IV 2a.
