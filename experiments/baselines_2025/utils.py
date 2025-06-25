import os
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
import glob
from mne.decoding import CSP
from sklearn.decomposition import FastICA, PCA

import pickle

# folder with the legacy concatenated .npz datasets (not distributed, see data/README.md)
DATA_ROOT = os.environ.get("EEGBCI_LEGACY_DATA_DIR", "data")

def create_windows(arr, labels, window_size, overlap=0, verbose=False):
    """
    Create windows of size `window_size` out of an array `arr` with optional overlap.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Input array of shape (n_samples, n_features)
    labels : numpy.ndarray
        Array of labels corresponding to each sample in arr
    window_size : int
        Size of each window in samples
    overlap : float, optional (default=0)
        Overlap ratio between consecutive windows (0 to 1). 
        For example, 0.5 means 50% overlap between windows.
    verbose : bool, optional (default=False)
        If True, print progress information
        
    Returns:
    --------
    X : numpy.ndarray
        Windowed data of shape (n_windows, window_size, n_features)
    y : numpy.ndarray
        Most common label for each window
    """
    if verbose:
        print("Creating windows...")
        
    window_size = int(window_size)
    overlap = float(overlap)
    
    # Validate overlap parameter
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in the range [0, 1)")
    
    # Calculate step size based on overlap
    step = max(1, int(window_size * (1 - overlap)))
    
    # Calculate number of windows that will fit in the data
    n_windows = max(1, (len(arr) - window_size) // step + 1)
    
    if n_windows < 1:
        raise ValueError(f"Window size {window_size} is too large for array of length {len(arr)} with overlap {overlap}")
    
    # Initialize lists to store windows and labels
    X = []
    y = []
    
    # Create sliding windows
    for i in range(0, len(arr) - window_size + 1, step):
        window = arr[i:i + window_size]
        window_labels = labels[i:i + window_size]
        
        # Get the most common label for the window
        most_common_label = stats.mode(window_labels, keepdims=False)[0]
        
        X.append(window)
        y.append(most_common_label)
        
        if verbose and i % (step * 10) == 0:
            print(f"Processed window starting at sample {i}/{len(arr)-window_size}...")
    
    if verbose:
        print(f"Created {len(X)} windows with size {window_size} and {overlap*100:.0f}% overlap")
    
    return np.array(X), np.array(y)

def read_eeg(win_size = 1, why="train", verbose=False, who=0, overlap=0):
    if why == "test":
        all_files = glob.glob(os.path.join(DATA_ROOT, "concat_scenario*_test", "*.npz"))
        file = all_files[who]
        if verbose:
            print(file)
    if verbose:
        print("Reading EEG data...")
    if why == "train":
        if verbose:
            print("Loading training data...")
        data = np.load(os.path.join(DATA_ROOT, "general_training_dataset.npz"))
    elif why == "test":
        if verbose:
            print("Loading test data...")
        data = np.load(file)
    else:
        raise ValueError("type must be 'train' or 'test'")
    
    if verbose:
        print("Data loaded")

    X = data['x']
    y = data['y']
    if why == "train":
        scalers = {}
        for i in range(X.shape[1]):
            scaler = StandardScaler()
            scaler.fit(X[:, i].reshape(-1, 1))
            scalers[i] = scaler
        X[:, i] = scaler.transform(X[:, i].reshape(-1, 1)).flatten()

        with open(f'scalers_{why}.pkl', 'wb') as f:
            pickle.dump(scalers, f)
    elif why == "test":
        with open(f'scalers_train.pkl', 'rb') as f:
            scalers = pickle.load(f)
        for i in range(X.shape[1]):
            X[:, i] = scalers[i].transform(X[:, i].reshape(-1, 1)).flatten()


    if verbose:
        print("Data shape:", X.shape, y.shape)

    X, y = create_windows(X, y, win_size*500, overlap, verbose=verbose)

    if verbose:
        print("Windows shape:", X.shape, y.shape)

    return X, y

def balance_labels(X, y):
    indexes0 = np.where(y == 0)[0]
    indexes1 = np.where(y == 1)[0]
    indexes2 = np.where(y == 2)[0]

    zeros = np.unique(y, return_counts=True)[1][0]
    ones = np.unique(y, return_counts=True)[1][1]
    twos = np.unique(y, return_counts=True)[1][2]

    MIN = min(zeros, ones, twos)

    random_indexes0 = np.random.choice(indexes0, size=zeros-MIN if zeros-MIN > 0 else 0, replace=False)
    random_indexes1 = np.random.choice(indexes1, size=ones-MIN if ones-MIN > 0 else 0, replace=False)
    random_indexes2 = np.random.choice(indexes2, size=twos-MIN if twos-MIN > 0 else 0, replace=False)
    # print(random_indexes)
    y = np.delete(y, np.concatenate((random_indexes0, random_indexes1, random_indexes2)))
    X = np.delete(X, np.concatenate((random_indexes0, random_indexes1, random_indexes2)), axis=0)
    
    return X, y

def zero_crossings(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum()


def extract_features(window):
    mean_val = np.mean(window, axis=1)
    std_val = np.std(window, axis=1)
    median_val = np.median(window, axis=1)
    var_val = np.var(window, axis=1)
    rms_val = np.sqrt(np.mean(window**2, axis=1))
    skew_val = skew(window, axis=1)
    kurt_val = kurtosis(window, axis=1)
    zcr_val = np.sum(np.diff(np.sign(window), axis=1) != 0, axis=1).astype(float)
    
    # Compute PSD per channel along the time/sample axis (axis=1)
    bands = {'alpha': (8, 12), 'beta': (12, 30)}
    # Ensure nperseg does not exceed the number of samples in the window to avoid UserWarning
    n_samples = window.shape[1]
    nperseg = 128 if n_samples >= 128 else n_samples  # adapt segment length dynamically
    freqs, psd = welch(window, fs=500, nperseg=nperseg, axis=1)
    
    # Compute mean power for each band **per channel**
    band_masks = [(freqs >= fmin) & (freqs <= fmax) for fmin, fmax in bands.values()]
    band_powers = [
        np.zeros(window.shape[0]) if not np.any(mask) else np.nanmean(psd[:, mask], axis=1)
        for mask in band_masks
    ]

    band_powers = np.stack(band_powers, axis=1)  # shape -> (n_channels, n_bands)
    # Combine all features per channel and flatten
    features = np.stack([median_val, std_val], axis=1)  # (n_channels, 2)
    # features = np.concatenate([features, band_powers], axis=1)  # (n_channels, 2 + n_bands)
    # Replace any remaining NaN / inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.flatten()

def make_features(X, verbose=False):
    X_train_f = []
    for i, window in enumerate(X):
        X_train_f.append(extract_features(window))
        if verbose and i % 1000 == 0:
            print(f"Processed {i} samples ...")
    X_train_f = np.array(X_train_f)
    return X_train_f

def apply_fbcsp(
    X: np.ndarray,
    y: np.ndarray,
    fs: int = 500,
    bands: list | None = None,
    n_c: int = 2,
    order: int = 4,
    verbose: bool = False,
) -> np.ndarray:
    """Filter-Bank Common Spatial Pattern (FBCSP) feature extraction.

    Parameters
    ----------
    X : ndarray, shape (epochs, samples, channels) **or** (epochs, channels, samples)
        EEG epochs.
    y : ndarray, shape (epochs,)
        Class labels (e.g. motor-imagery classes).
    fs : int, optional
        Sampling frequency of *X* in Hertz. Default is 500.
    bands : list[(low, high)], optional
        List of (low, high) band-pass edges in Hz. If *None*, uses 4-Hz wide
        overlapping bands from 4–40 Hz: (4–8, 8–12, …, 36–40).
    n_c : int, optional
        Number of CSP components **per band** (default 2 → 4 features/band
        because CSP returns log-variance, one per component).
    order : int, optional
        Butterworth filter order (default 4).
    verbose : bool, optional
        Print progress if *True*.

    Returns
    -------
    X_fbcsp : ndarray, shape (epochs, n_bands × n_c)
        Concatenated CSP log-variance features across all filter bands, ready
        for feeding into an SVM / LDA classifier.
    """
    # ------------ sanity checks & axis handling --------------------------------
    if X.ndim != 3:
        raise ValueError("X must be 3-D: (epochs, samples, channels) or (epochs, channels, samples)")

    if X.shape[1] > X.shape[2]:  # (epochs, samples, channels)
        X = np.transpose(X, (0, 2, 1))  # → (epochs, channels, samples)

    X = X.astype(np.float64)
    n_epochs, n_channels, _ = X.shape

    # ------------ default filter bank ------------------------------------------
    if bands is None:
        bands = [(4, 8), (8, 12), (12, 30), (30, 45)]  # 4-Hz bands 4–40 Hz

    from scipy.signal import butter, lfilter

    def _bandpass(data: np.ndarray, low: float, high: float) -> np.ndarray:
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
        return lfilter(b, a, data, axis=-1)

    features = []
    for idx, (low, high) in enumerate(bands):
        if verbose:
            print(f"Filtering {low:>2}-{high:<2} Hz  (band {idx+1}/{len(bands)}) …")
        X_band = _bandpass(X, low, high)
        csp = CSP(n_components=n_c, log=True, norm_trace=False)
        X_csp = csp.fit_transform(X_band, y)  # (epochs, n_c)
        features.append(X_csp)

    X_fbcsp = np.concatenate(features, axis=1)  # (epochs, n_bands*n_c)
    return X_fbcsp

def applyICA(X, y, n_components=2, verbose=False):
    ica = FastICA(n_components=n_components, random_state=1)
    X_ica = ica.fit_transform(X, y)
    return X_ica

def applyPCA(X, y, n_components=2, verbose=False):
    pca = PCA(n_components=n_components, random_state=1)
    X_pca = pca.fit_transform(X, y)
    return X_pca