import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # keep the suite CPU-light
    os.environ.setdefault(_k, "1")

import numpy as np
import pytest

from eegbci.evaluation import make_model
from eegbci.realtime import FINAL_CFG, offline_features
from tests.helpers import FakeRec, make_signal
from eegbci.windows import build_windows

# toy protocol on a 72 s synthetic recording (labels follow the 12 s alpha switching of make_signal)
SYNTH_BLOCKS = [(0, 12, 1, "push#1", 0), (12, 24, 0, "rest#1", 0), (24, 36, 2, "walk#1", 1), (36, 48, 0, "rest#2", 1),
                (48, 60, 1, "push#2", 2), (60, 72, 0, "rest#3", 2)]


@pytest.fixture(scope="session")
def synth():
    return FakeRec(data=make_signal(72.0, seed=0, spike_at_s=50.0), blocks=SYNTH_BLOCKS)


@pytest.fixture(scope="session")
def W4(synth):
    return build_windows(synth, win_s=4.0, interp_bads=False, cache=False)      # cache=False: never touches disk


@pytest.fixture(scope="session")
def W2(synth):
    return build_windows(synth, win_s=2.0, interp_bads=False, cache=False)


@pytest.fixture(scope="session")
def riemann_bundle(W4):
    F = offline_features(W4)
    t = W4["meta"]["t_end"].to_numpy()
    y = (np.sin(2 * np.pi * (t - 2.0) / 24.0) > 0).astype(int)       # alpha-high periods -> varied P(active)
    return dict(model=make_model("lr", 1.0).fit(F, y), cfg=FINAL_CFG)
