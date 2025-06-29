import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score

import sys, os
# Ensure project root (parent of this file's directory) is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from utils import read_eeg, balance_labels, make_features

###############################################################################
# Utility functions
###############################################################################

def window_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """Compute the Frobenius (Euclidean) distance between two EEG windows.

    Parameters
    ----------
    w1, w2 : ndarray, shape (500, 30)
        Two EEG windows (time-samples × channels).

    Returns
    -------
    float
        Frobenius norm ‖w1 − w2‖_F.
    """
    # Ensure the inputs have the correct shape
    if w1.shape != w2.shape:
        raise ValueError(f"Shape mismatch: {w1.shape} vs {w2.shape}")
    return np.linalg.norm(w1 - w2)


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten EEG windows to shape (n_windows, 15_000)."""
    n_w, s, c = X.shape
    assert s == 500 and c == 30, "Unexpected window shape"
    return X.reshape(n_w, -1)

###############################################################################
# Main routine
###############################################################################

import datetime

from glob import glob

def main(k: int = 5, grid: bool = False, use_features: bool=False, pcs: int = 900, random_state: int | None = None, img_prefix: str = "KNN", testall: bool=False, plot2d: bool=False, plot3d: bool=False):
    """Train & evaluate a brute-force KNN on raw EEG windows."""
    # ------------------------------------------------------------------
    # 1. Data loading (same helper used in Logistic-Regression notebook)
    # ------------------------------------------------------------------
    X_raw, y = read_eeg(win_size=1, why="train", verbose=False)  # (N, 500, 30)
    X_raw, y = balance_labels(X_raw, y)  # class balancing
    print(f"Total windows: {X_raw.shape[0]}, window shape: {X_raw.shape[1:]}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique.astype(int), counts)))

    # ------------------------------------------------------------------
    # 2. Feature representation
    # ------------------------------------------------------------------
    if use_features:
        print("Extracting engineered features …")
        X_repr = make_features(X_raw, verbose=False)
    else:
        X_repr = flatten_windows(X_raw)

    # ------------------------------------------------------------------
    # 3. Split into train/validation
    # ------------------------------------------------------------------
    X_train_repr, X_test_repr, y_train, y_test = train_test_split(
        X_repr,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 4. K-NN classifier (brute-force, Euclidean distance)
    # Using Euclidean distance between 30-dim median feature vectors.
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Grid search parameter ranges
    comps_list = [500, 550, 600, 650, 700]  # 150, 300, ..., 1500

    ks = list(range(5, 120, 5))
    metrics = ["euclidean", "cosine", "manhattan"]
    results = []
    # store scaler reference globally for later plot
    scaler_global = StandardScaler()

    if use_features:
        # Scale feature vectors once
        scaler = scaler_global.fit(X_train_repr)
        X_train_scaled = scaler.transform(X_train_repr)
        X_test_scaled = scaler.transform(X_test_repr)

        metrics = ["euclidean", "cosine", "manhattan"]
        ks = [1,3,5,7,9,11,13,15,17,19] if grid else [k]
        for _metric in metrics:
            for _weights in (["uniform","distance"] if grid else ["uniform"]):
                for _k in ks:
                    knn = KNeighborsClassifier(n_neighbors=_k, metric=_metric, weights=_weights, algorithm="brute", n_jobs=-1)
                    knn.fit(X_train_scaled, y_train)
                    acc = knn.score(X_test_scaled, y_test)
                    results.append((acc, None, _k, _metric, _weights))
                    print(f"Features  K={_k:<2} metric={_metric:<8} w={_weights:<8} -> {acc*100:.2f}%")
    else:
        for n_comp in (comps_list if grid else [pcs]):
            scaler = StandardScaler().fit(X_train_repr)
            X_train_scaled = scaler.transform(X_train_repr)
            X_test_scaled = scaler.transform(X_test_repr)

            pca = PCA(n_components=n_comp, whiten=True, random_state=42)
            # cumulative variance diagnostic
            cum_var = np.cumsum(pca.fit(flatten_windows(X_raw)) if False else [0])  # placeholder
            X_train = pca.fit_transform(X_train_scaled)
            X_test = pca.transform(X_test_scaled)

            for _metric in metrics:
                for _weights in (['uniform','distance'] if grid else ['uniform']):
                    for _k in ks:
                        knn = KNeighborsClassifier(
                            n_neighbors=_k,
                            metric=_metric,
                            weights=_weights,
                            algorithm="brute",
                            n_jobs=-1,
                        )
                        knn.fit(X_train, y_train)
                        acc = knn.score(X_test, y_test)
                        results.append((acc, n_comp, _k, _metric, _weights))
                        print(f"PCs={n_comp:<4} K={_k:<2} metric={_metric:<8} w={_weights:<8} -> {acc*100:.2f}%")
        # report best
    best = max(results, key=lambda t: t[0])
    best_acc, best_comp, best_k, best_metric, best_weights = best
    print(f"\nBest: PCs={best_comp}, K={best_k}, metric={best_metric}, weights={best_weights} -> {best_acc*100:.2f}%\n")
    
    # ---------------- plotting all results (if results include PCs)
    if grid and len(results)>10:
        df = pd.DataFrame(results, columns=["acc","pcs","k","metric","weights"])
        pivot = df[df["metric"]==best_metric].pivot_table(index="pcs", columns="k", values="acc")
        plt.figure(figsize=(10,6))
        sns.heatmap(pivot*100, annot=False, cmap="viridis")
        plt.title("Accuracy heatmap (%), metric="+best_metric)
        plt.ylabel("PCA components")
        plt.xlabel("K")
        plt.tight_layout()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{img_prefix}_heatmap_{ts}.png")
        plt.close()


    # ------------------------------------------------------------------
    # Diagnostic report for best model
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_train_repr)
    X_train_scaled = scaler.transform(X_train_repr)
    X_test_scaled = scaler.transform(X_test_repr)

    if best_comp is None:
        X_train_best, X_test_best = X_train_scaled, X_test_scaled
    else:
        pca = PCA(n_components=best_comp, whiten=True, random_state=42)
        X_train_best = pca.fit_transform(X_train_scaled)
        X_test_best = pca.transform(X_test_scaled)

    knn_best = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=best_metric,
        weights=best_weights,
        algorithm="brute",
        n_jobs=-1,
    )
    knn_best.fit(X_train_best, y_train)
    y_pred = knn_best.predict(X_test_best)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    ts_c = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{img_prefix}_confusion_{ts_c}.png")
    plt.close()
    print(classification_report(y_test, y_pred, digits=3))

    # ---------------- 2D scatter plot
    if plot2d:
        try:
            from sklearn.decomposition import PCA as _PCA_2D
            n_vis = min(3000, X_train_repr.shape[0])
            X_vis = X_train_repr[:n_vis]
            y_vis = y_train[:n_vis]
            if scaler_global is not None:
                X_vis = scaler_global.transform(X_vis)
            pca2 = _PCA_2D(n_components=2, random_state=42)
            X_vis_2d = pca2.fit_transform(X_vis)
            plt.figure(figsize=(6,5))
            palette = sns.color_palette("bright", len(np.unique(y_vis)))
            for cls in np.unique(y_vis):
                idxs = y_vis == cls
                plt.scatter(X_vis_2d[idxs,0], X_vis_2d[idxs,1], s=8, label=str(int(cls)), alpha=0.6, color=palette[int(cls)])
            plt.legend(title="Class")
            plt.title("2D PCA projection of training data")
            plt.tight_layout()
            ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{img_prefix}_2d_{ts2}.png", dpi=200)
            plt.close()
        except Exception as e:
            print("2D plot failed:", e)

    # ---------------- 3D scatter plot (on TEST data)
    if plot3d:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            if X_test_best.shape[1] >= 3:
                X_vis3 = X_test_best
            else:
                from sklearn.decomposition import PCA as _PCA_3D
                X_vis3 = _PCA_3D(n_components=3, random_state=42).fit_transform(X_test_best)
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(projection='3d')
            palette3 = sns.color_palette("bright", len(np.unique(y_test)))
            for cls in np.unique(y_test):
                idxs = y_test == cls
                ax.scatter(X_vis3[idxs,0], X_vis3[idxs,1], X_vis3[idxs,2], s=8, label=str(int(cls)), alpha=0.6, color=palette3[int(cls)])
            ax.set_title('3D projection of test data')
            ax.legend(title='Class')
            ts3 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plt.savefig(f"{img_prefix}_3d_{ts3}.png", dpi=200)
            plt.close()
        except Exception as e:
            print("3D plot failed:", e)

    # ------------------------------------------------------------------
    #  Additional evaluation on every individual test scenario
    # ------------------------------------------------------------------
    if testall:
        test_paths = glob(os.path.join(os.environ.get("EEGBCI_LEGACY_DATA_DIR", "data"), "concat_scenario*_test", "*.npz"))
        print(f"\nEvaluating on {len(test_paths)} separate test files...\n")
        for idx, _ in enumerate(test_paths):
            X_t, y_t = read_eeg(win_size=1, why="test", who=idx, verbose=False)
            if use_features:
                X_t_repr = make_features(X_t)
                X_t_scaled = scaler.transform(X_t_repr)
                X_t_final = X_t_scaled
            else:
                X_t_flat = flatten_windows(X_t)
                X_t_scaled = scaler.transform(X_t_flat)
                X_t_final = pca.transform(X_t_scaled)
            y_pred_t = knn_best.predict(X_t_final)
            acc_t = accuracy_score(y_t, y_pred_t)
            print(f"Test file #{idx}: accuracy = {acc_t*100:.2f}%  (n={len(y_t)})")
            cm_t = confusion_matrix(y_t, y_pred_t)
            plt.figure(figsize=(4,3))
            sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Test#{idx} Confusion')
            plt.tight_layout()
            ts_any = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{img_prefix}_test{idx}_{ts_any}.png")
            plt.close()

    n_for_sil = min(50, X_train_best.shape[1])
    try:
        sil = silhouette_score(X_train_best[:, :min(n_for_sil, X_train_best.shape[1])], y_train)
        print(f"Silhouette score on first {n_for_sil} PCs: {sil:.3f}")
    except Exception as e:
        print("Silhouette score computation failed:", e)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    parser = argparse.ArgumentParser(description="Brute-force KNN on raw EEG windows")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours (used if --grid not set)")
    parser.add_argument("--grid", action="store_true", help="Run grid search over a list of K values")
    parser.add_argument("--features", action="store_true", help="Use engineered feature vectors instead of raw windows")
    parser.add_argument("--pcs", type=int, default=900, help="Number of PCA components when not using grid search")
    parser.add_argument("--prefix", type=str, default="KNN-Norm", help="Prefix for saved image files")
    parser.add_argument("--testall", action="store_true", help="Evaluate trained model on each separate test file")
    parser.add_argument("--plot2d", action="store_true", help="Save 2D scatter of training data after PCA")
    parser.add_argument("--plot3d", action="store_true", help="Save 3D scatter of test data")
    args = parser.parse_args()

    main(k=args.k, grid=args.grid, use_features=args.features, pcs=args.pcs, img_prefix=args.prefix, testall=args.testall, plot2d=args.plot2d, plot3d=args.plot3d)

# 50 %