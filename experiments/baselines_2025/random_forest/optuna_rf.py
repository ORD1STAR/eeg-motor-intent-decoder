import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, FastICA
from datetime import datetime
import joblib

# Added imports and wrapper to leverage utils.py for data loading
import sys, os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import read_eeg, balance_labels, make_features

def read_data(files_dir, subject, window_size, labels=None, low_freq=None, high_freq=None,
              sfreq=500, bands=None, verbose=False, scalers=None, test=False):
    """Compatibility wrapper that fetches EEG data via utils.read_eeg and
    returns it in the format expected by the original Optuna pipeline."""
    # utils.read_eeg uses seconds rather than samples for its window size
    win_sec = max(1, int(round(window_size / 500)))
    X_raw, y = read_eeg(win_size=win_sec, why="test" if test else "train", verbose=verbose)

    if not test:
        X_raw, y = balance_labels(X_raw, y)

    # Convert raw windows to engineered feature vectors
    X_feat = make_features(X_raw, verbose=verbose)

    if test:
        return X_feat, y, None
    else:
        # Mirror training set into the *test* return slot to keep downstream code unchanged.
        return X_feat, y, X_feat, y, {}

# === PRESET CONFIGS ===
n_jobs = -1
cv_folds = 5
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=1)

# Paths
files = "data/scenario2"
labels = [[
    (0, 80, 1), (80, 160, 2), (160, 180, 0),
    (180, 260, 1), (260, 340, 2), (340, 360, 0),
    (360, 430, 1), (430, 500, 2)
]]
files2 = "data/old"
val_labels = [[(0, 10, 0), (10, 18, 2), (18, 26, 1)]]

sfreq = 500
bands = {
    'delta': (1, 4), 'theta': (4, 8),
    'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)
}

# === OBJECTIVE FUNCTION FOR OPTUNA ===
def objective(trial):
    # -------- OPTUNA PARAMETERS ----------
    win = trial.suggest_int("window_size", 250, 2500, step=250)
    lf = trial.suggest_int("low_freq", 0, 12, step=1)
    hf = trial.suggest_int("high_freq", 35, 40, step=5)

    if lf >= hf:
        raise optuna.exceptions.TrialPruned()

    # RANDOM FOREST HYPERPARAMETERS
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 50, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    }

    # -------- LOAD DATA ----------
    try:
        X_train_raw, y_train, X_test_raw, y_test, scalers = read_data(
            files, "S05",
            window_size=win,
            labels=labels *16,
            low_freq=lf,
            high_freq=hf,
            sfreq=sfreq,
            bands=bands,
            verbose=False
        )
        X_val_raw, y_val, _ = read_data(
            files2, "test21",
            window_size=win,
            labels=val_labels,
            scalers=scalers,
            low_freq=lf,
            high_freq=hf,
            sfreq=sfreq,
            bands=bands,
            verbose=False,
            test=True
        )
    except:
        raise optuna.exceptions.TrialPruned()

    # -------- FEATURE REDUCTION ----------
    try:
        pca = PCA(n_components=0.95, svd_solver='full', random_state=1)
        X_train_pca = pca.fit_transform(X_train_raw)
        X_val_pca = pca.transform(X_val_raw)
        X_test_pca = pca.transform(X_test_raw)

        ica = FastICA(n_components=min(X_train_pca.shape[1], 20), random_state=1)
        X_train = ica.fit_transform(X_train_pca)
        X_val = ica.transform(X_val_pca)
        X_test = ica.transform(X_test_pca)
    except:
        raise optuna.exceptions.TrialPruned()

    # -------- CV EVALUATION ON VALIDATION SET ----------
    val_scores = []
    for train_idx, _ in kf.split(X_train, y_train):
        clf = RandomForestClassifier(**rf_params, random_state=1, n_jobs=n_jobs)
        clf.fit(X_train[train_idx], y_train[train_idx])
        y_val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        val_scores.append(acc)

    return np.mean(val_scores)

# === RUN OPTIMIZATION ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
study_name = f"rf_optuna_{timestamp}"
study = optuna.create_study(direction="maximize", study_name=study_name)

# 🧠 Enqueue a first trial with user-specified starting parameters
study.enqueue_trial({
    "window_size": 1500,
    "low_freq": 7,
    "high_freq": 35,
    "n_estimators": 100,
    "max_depth": 45,
    "min_samples_split": 20,
    "min_samples_leaf": 9,
    "criterion": "entropy",
})
study.optimize(objective, n_trials=30)

# === RESULTS ===
print("\n=============================")
print("📊 Best Trial:")
print(f"  🏆 Accuracy: {study.best_value*100:.2f}%")
print("  🧠 Parameters:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")
print("=============================\n")


# ========== VISUALISATION ==========
import optuna.visualization as vis

print("📊 Génération des graphiques Optuna...")

# 1. Optimization History
vis.plot_optimization_history(study).write_html("plot_optimization_history.html", include_plotlyjs="cdn")

# 2. Parameter Importance
vis.plot_param_importances(study).write_html("plot_param_importance.html", include_plotlyjs="cdn")

# 3. Parallel Coordinates
vis.plot_parallel_coordinate(study).write_html("plot_parallel_coordinates.html", include_plotlyjs="cdn")

# 4. Slice Plot
vis.plot_slice(study).write_html("plot_slice.html", include_plotlyjs="cdn")

# 5. Contour plot for selected most relevant hyperparameters
# (You can adjust the params list depending on what was explored in the study)
contour_params = ["window_size", "low_freq", "high_freq", "n_estimators", "max_depth"]
vis.plot_contour(study, params=contour_params).write_html("plot_contour.html", include_plotlyjs="cdn")

print("✅ Plots sauvegardés en fichiers HTML.")
