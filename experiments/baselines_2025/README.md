# Baselines, June-July 2025 (first phase)

The first phase of the project: standard classifiers on hand-crafted window features, one folder per method.
They are kept **as the historical starting point, not as results**. The audit in `docs/legacy_audit.html`
(code in `scripts/audit/`) shows why their numbers cannot be trusted:

* windows were split at random (overlapping neighbours leak), which lifts a model that sits at 36 % on held-out blocks
  to 98 %;
* hyper-parameters were selected on the test files (SVM-MRFO fitness, Optuna objective, best-of-26 files for logistic
  regression);
* the training data was standardised on the last channel only (`utils.py`), and CSP/ICA/PCA were fitted separately on
  train and test (`utils.py`, SVM notebook);
* the EEGNet notebook reshaped `(N, 250, 30)` instead of transposing it.

The code is left unchanged apart from anonymisation and paths. `utils.py` reads the concatenated `.npz` sets from
`EEGBCI_LEGACY_DATA_DIR` (not distributed). Notebook outputs were cleared (they contained participant file names).

| Folder | Method | Accuracy printed in `graphs.py` (3 classes, chance 33 %) |
|---|---|---|
| `knn/knn_csp.py` | KNN on filter-bank CSP | 50 |
| `fuzzy/fuzzy.ipynb` | fuzzy classifier | 49.61 |
| `knn/knn_brute.py` | KNN, brute-force distances | 48.2 |
| `logistic_regression/` | logistic regression | 47.5 |
| `svm/` | SVM tuned with MRFO | 44.3 |
| `eegnet/` | EEGNet (PyTorch) | 42 |
| `random_forest/` | random forest + Optuna | 35.89 |

The accuracy column is the list hard-coded in `graphs.py` (typed by hand from notebook outputs, not recomputed).
`extraction/` builds the concatenated training/test sets from the EDF files. A cell of the EEGNet notebook that drew a
synthetic accuracy curve with `random.random()` was removed (the audit flags it). The `rf_optuna.ipynb` notebook was
named after an unrelated project in the original folder; its content is the random-forest/Optuna search.

Figures from these runs: `docs/figures/legacy_*.png`.
