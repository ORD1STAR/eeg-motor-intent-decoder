"""Paths, protocol definitions and default parameters.

Locations are configurable through environment variables (nothing is hard-coded to a machine):

    EEGBCI_DATA_DIR      folder holding the EDF recordings            (default: <repo>/data/raw)
    EEGBCI_CACHE_DIR     window / feature cache (large, rebuildable)   (default: <repo>/.cache/eegbci)
    EEGBCI_RESULTS_DIR   logs, json and figures written by the steps   (default: <repo>/results)
    EEGBCI_MODELS_DIR    trained real-time model(s)                    (default: <repo>/models)
"""
import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]
ROOT = _REPO if (_REPO / "pyproject.toml").exists() else Path.cwd()


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


DATA_DIR = _env_path("EEGBCI_DATA_DIR", ROOT / "data" / "raw")
CACHE_DIR = _env_path("EEGBCI_CACHE_DIR", ROOT / ".cache" / "eegbci")
RESULTS_DIR = _env_path("EEGBCI_RESULTS_DIR", ROOT / "results")
MODELS_DIR = _env_path("EEGBCI_MODELS_DIR", ROOT / "models")
ASSETS_DIR = _PKG / "assets"

SFREQ = 500.0
CHANNELS = ["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz", "P3", "P7", "O1",
            "Oz", "O2", "P4", "P8", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"]
FRONTOPOLAR = ["Fp1", "Fp2"]
MOTOR = ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"]

NEUTRAL, PUSH, WALK = 0, 1, 2
LABEL_NAMES = {NEUTRAL: "neutral", PUSH: "push", WALK: "walk"}

# (start_s, stop_s, label, block name, cycle). "cycle" groups blocks so that a cycle can be held out
# without splitting any block. In scenario 1 the eyes-closed baseline gets label None (excluded).
PROTOCOLS = {
    1: [(0, 120, None, "baseline-EC", 0), (120, 240, NEUTRAL, "baseline-EO", 0), (240, 270, PUSH, "push#1", 0),
        (270, 290, NEUTRAL, "rest#1", 0), (290, 320, PUSH, "push#2", 1), (320, 340, NEUTRAL, "rest#2", 1),
        (340, 370, WALK, "walk#1", 0), (370, 390, NEUTRAL, "rest#3", 0), (390, 420, WALK, "walk#2", 1),
        (420, 480, NEUTRAL, "rest#4", 1)],
    2: [(0, 80, PUSH, "push#1", 0), (80, 160, WALK, "walk#1", 0), (160, 180, NEUTRAL, "rest#1", 0),
        (180, 260, PUSH, "push#2", 1), (260, 340, WALK, "walk#2", 1), (340, 360, NEUTRAL, "rest#2", 1),
        (360, 430, PUSH, "push#3", 2), (430, 500, WALK, "walk#3", 2)],
}
PROTOCOL_MARKERS = {1: [0, 120, 240, 270, 290, 320, 340, 370, 390, 420], 2: [0, 80, 160, 180, 260, 340, 360, 430]}

# Filter bank used for covariance features (Hz)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 30)]
MI_BAND = (8, 30)

WIN_S = 2.0       # window length
STEP_S = 0.5      # hop between windows
GUARD_S = 1.0     # seconds dropped after each block start and before its end (reaction time)
PTP_REJECT_UV = 150.0
