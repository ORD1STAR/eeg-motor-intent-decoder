"""Figures for the results page: scalp maps of the transition effect + time courses.

python -m eegbci.figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from . import config as C
from .data import load_recordings
from .neuro import ersp

PO = ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def main():
    recs = load_recordings()
    info = mne.create_info(C.CHANNELS, C.SFREQ, "eeg"); info.set_montage("standard_1005")
    curves, maps = {}, {}
    for bname, band in [("alpha", (8, 13)), ("beta", (13, 30))]:
        rel, R = ersp(recs, band, kinds=("rest->task", "task->rest"))
        post = (rel >= 3) & (rel <= 15)
        for kind, A in R.items():
            maps[(bname, kind)] = np.median(A[:, :, post].mean(2), axis=0)
            po = A[:, [C.CHANNELS.index(c) for c in PO]].mean(1)          # (n_rec, n_t)
            curves[f"{bname}|{kind}"] = dict(rel=rel.tolist(), mean=po.mean(0).tolist(),
                                              sem=(po.std(0) / np.sqrt(len(po))).tolist(), n=len(po))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
    vmax = max(np.abs(v).max() for v in maps.values())
    for i, bname in enumerate(("alpha", "beta")):
        for j, kind in enumerate(("rest->task", "task->rest")):
            ax = axes[i, j]
            im, _ = mne.viz.plot_topomap(maps[(bname, kind)], info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(-vmax, vmax), contours=4, sensors=True)
            ax.set_title(f"{bname}  {kind.replace('->', ' → ')}", fontsize=10)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label="median change, dB (3–15 s after vs 2–12 s before)")
    fig.savefig(C.RESULTS_DIR / "fig_topomaps.png", dpi=160, bbox_inches="tight", facecolor="white")
    json.dump(curves, open(C.RESULTS_DIR / "transition_curves.json", "w"))
    print("saved fig_topomaps.png and transition_curves.json")


if __name__ == "__main__":
    main()
