"""Summary figures built only from files stored in results/ (no number is typed in by hand except the QA table,
whose values are quoted from results/code_review.md and are marked as such).

    python scripts/plot_summary.py            # writes docs/figures/auc_summary.png and docs/figures/qa_corrections.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, RED, GREEN = "#2f6fb0", "#8a8f98", "#c0504d", "#3a8f5c"


def load(name):
    return json.load(open(RES / name, encoding="utf-8"))


def rec_auc(d, model):
    return np.array([v[model]["auc"] for v in d.values()])


ens = load("ensemble_loso.json")
eegnet = np.array([v["auc"] for v in load("deep_eegnet_loso.json").values()])
nested = load("nested_loso.json")
nested_auc = np.array([a for s in nested.values() for a in s["test"].values()])
nulls = [load(f"ensemble_loso_null{i}.json") for i in (1, 2)]

bars = [("EEGNet", eegnet), ("ShallowConvNet", rec_auc(ens, "shallow")), ("Riemann fusion\n+ LR", rec_auc(ens, "riemann")),
        ("Nested LOSO\n(selection inside)", nested_auc), ("Ensemble\n(Riemann + Shallow)", rec_auc(ens, "ensemble"))]
null_ens = np.mean([rec_auc(n, "ensemble").mean() for n in nulls])

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
x = np.arange(len(bars))
means = [b[1].mean() for b in bars]
ax[0].bar(x, means, color=BLUE, width=0.62)
for i, (_, v) in enumerate(bars):
    ax[0].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), v, s=9, color="white", edgecolor="k", lw=0.4, zorder=3)
    ax[0].text(i, 0.325, f"{v.mean():.3f}", ha="center", fontsize=10, fontweight="bold", color="white")
ax[0].axhline(0.5, color=GREY, ls="--", lw=1, label="chance 0.5")
ax[0].axhline(null_ens, color=RED, ls=":", lw=1.6, label=f"drift-null, ensemble (mean of 2 shifts) {null_ens:.3f}")
ax[0].legend(loc="upper left", fontsize=8, frameon=False)
ax[0].set_xticks(x, [b[0] for b in bars], fontsize=7.5)
ax[0].set_ylim(0.3, 1.05)
ax[0].set_ylabel("AUC, smoothed P(active), leave-one-subject-out")
ax[0].set_title("Rest vs active ('brain switch'): mean over 28 recordings\n(dots = recordings; pre-QA numbers, see qa_corrections.png)", fontsize=9)

e = rec_auc(ens, "ensemble")
subj = np.array([k.split("-")[0] for k in ens])
order = np.argsort([e[subj == s].mean() for s in sorted(set(subj))])
ss = np.array(sorted(set(subj)))[order]
for i, s in enumerate(ss):
    v = e[subj == s]
    ax[1].scatter(v, np.full(len(v), i), color=BLUE, s=22, zorder=3)
    ax[1].plot([v.min(), v.max()], [i, i], color=GREY, lw=0.8, zorder=2)
ax[1].set_yticks(range(len(ss)), ss)
ax[1].axvline(0.5, color=GREY, ls="--", lw=1)
ax[1].set_xlabel("ensemble AUC per recording")
ax[1].set_title("Per participant (a held-out person each time)\nlarge spread between and within people", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "auc_summary.png", dpi=150)
plt.close(fig)

# --- QA corrections (values quoted from results/code_review.md and results/FINDINGS_phase1_switch.md) ---
items = [("As reported\n(Riemann LOSO)", 0.671, BLUE),
         ("Eyes-open baseline\nnot scored (scen. 1)", 0.638, RED),
         ("Reference restarted\nat eyes-open onset", 0.606, RED),
         ("Scale-invariant\neigenvalue floor", 0.641, RED)]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax[0].bar(range(len(items)), [i[1] for i in items], color=[i[2] for i in items], width=0.6)
for i, it in enumerate(items):
    ax[0].text(i, it[1] + 0.008, f"{it[1]:.3f}", ha="center", fontweight="bold", fontsize=9)
ax[0].axhline(0.5, color=GREY, ls="--", lw=1)
ax[0].set_xticks(range(len(items)), [i[0] for i in items], fontsize=8)
ax[0].set_ylim(0.4, 0.75)
ax[0].set_ylabel("Riemann fusion LOSO AUC")
ax[0].set_title("What the independent code review changed", fontsize=9)
sc = [("Scenario 1\n(12 rec.)", 0.800), ("Scenario 2\n(16 rec.)", 0.574)]
ax[1].bar(range(2), [s[1] for s in sc], color=[GREEN, BLUE], width=0.5)
for i, s in enumerate(sc):
    ax[1].text(i, s[1] + 0.008, f"{s[1]:.3f}", ha="center", fontweight="bold", fontsize=9)
ax[1].axhline(0.5, color=GREY, ls="--", lw=1)
ax[1].set_xticks(range(2), [s[0] for s in sc], fontsize=8)
ax[1].set_ylim(0.4, 0.9)
ax[1].set_title("Same model, per protocol:\nthe eyes-closed baseline inflates scenario 1", fontsize=9)
fig.text(0.01, 0.005, "values quoted from results/code_review.md", fontsize=7, color=GREY)
fig.tight_layout()
fig.savefig(OUT / "qa_corrections.png", dpi=150)
print("ok", [round(m, 3) for m in means], round(null_ens, 3))
