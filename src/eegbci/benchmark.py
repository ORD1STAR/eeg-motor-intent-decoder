"""Final tests.

nested   nested leave-one-subject-out: for every held-out person, the configuration is selected
         by an inner LOSO on the other 9 people only, then trained on those 9 and tested once.
         This is the unbiased estimate of the whole procedure (selection included).

python -m eegbci.benchmark nested
"""
import json
import sys

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .evaluation import eval_groups, summarize
from .explore import PARIETO_OCC, load_bundle

PO = [C.CHANNELS.index(c) for c in PARIETO_OCC]
A = dict(bands="fb", align="causal", tau_s=60, kind="logbp")
T = dict(bands=[1, 2, 3, 4], align="causal", tau_s=60, kind="ts", channels=PO)
CANDIDATES = {f"{name}-win{w:g}": (cfg, w) for w in (2.0, 4.0)
              for name, cfg in (("logbp", A), ("tsPO", T), ("fusion", [A, T]))}
SMOOTH = 8.0
C_REG = 0.01


def groups_for(B, subjects_test, subjects_train):
    return [([q for q, r in B.recs.items() if r.subject == s],
             [q for q, r in B.recs.items() if r.subject in subjects_train and r.subject != s]) for s in subjects_test]


def nested():
    bundles = {k: load_bundle(cfg, win_s=w) for k, (cfg, w) in CANDIDATES.items()}
    any_B = next(iter(bundles.values()))
    subjects = sorted({r.subject for r in any_B.recs.values()})
    out = {}
    for s in subjects:
        inner = [x for x in subjects if x != s]

        def inner_score(k):
            B = bundles[k]
            return k, summarize(eval_groups(B, "task_rest", groups_for(B, inner, inner), "lr", C_reg=C_REG,
                                            smooth_s=SMOOTH), "auc")["mean"]
        scores = dict(Parallel(n_jobs=6, prefer="threads")(delayed(inner_score)(k) for k in bundles))
        best = max(scores, key=scores.get)
        B = bundles[best]
        res = eval_groups(B, "task_rest", groups_for(B, [s], inner), "lr", C_reg=C_REG, smooth_s=SMOOTH)
        out[s] = dict(selected=best, inner=scores, test={k: v["auc"] for k, v in res.items()},
                      test_bacc={k: v["bacc"] for k, v in res.items()})
        print(f"{s}: selected {best:<14} (inner {scores[best]:.3f}) -> test AUC "
              + " ".join(f"{k}:{v['auc']:.3f}" for k, v in res.items()), flush=True)
    aucs = np.array([v for d in out.values() for v in d["test"].values()])
    baccs = np.array([v for d in out.values() for v in d["test_bacc"].values()])
    per_subj = np.array([np.mean(list(d["test"].values())) for d in out.values()])
    print(f"NESTED LOSO task_rest: recording AUC mean {aucs.mean():.3f} median {np.median(aucs):.3f}, "
          f"bacc mean {baccs.mean():.3f}; subject-level AUC mean {per_subj.mean():.3f}, "
          f"{(per_subj > 0.5).sum()}/{len(per_subj)} subjects above 0.5")
    json.dump(out, open(C.RESULTS_DIR / "nested_loso.json", "w"), indent=1)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
