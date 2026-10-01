"""Signal-level audit of the PFE EEG recordings + leakage-free baselines.

Outputs a JSON with everything the HTML report needs.
"""
import glob, os, re, json, sys, warnings
import numpy as np
import mne
from scipy.signal import welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
ROOT = os.environ.get("EEGBCI_DATA_DIR", "data/raw")
OUT = sys.argv[1]
FS = 500

SCEN1 = [(0, 120, 0, "neutral-EC"), (120, 240, 0, "neutral-EO"), (240, 270, 1, "push#1"), (270, 290, 0, "neutral"),
         (290, 320, 1, "push#2"), (320, 340, 0, "neutral"), (340, 370, 2, "walk#1"), (370, 390, 0, "neutral"),
         (390, 420, 2, "walk#2"), (420, 480, 0, "neutral")]
SCEN2 = [(0, 80, 1, "push#1"), (80, 160, 2, "walk#1"), (160, 180, 0, "neutral"), (180, 260, 1, "push#2"),
         (260, 340, 2, "walk#2"), (340, 360, 0, "neutral"), (360, 430, 1, "push#3"), (430, 500, 2, "walk#3")]
SCEN1_MARKS = [0, 120, 240, 270, 290, 320, 340, 370, 390, 420]
SCEN2_MARKS = [0, 80, 160, 180, 260, 340, 360, 430]

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
REGIONS = {"frontal-polar": ["Fp1-TP", "Fp2-TP"], "frontal": ["F3-TP", "Fz-TP", "F4-TP"],
           "central": ["C3-TP", "Cz-TP", "C4-TP"], "parietal": ["P3-TP", "Pz-TP", "P4-TP"],
           "occipital": ["O1-TP", "Oz-TP", "O2-TP"], "temporal": ["T7-TP", "T8-TP", "FT9-TP", "FT10-TP"]}


def subject_of(fname):
    """'3_S05.edf' -> 'S05' (recordings are pseudonymised, see data/README.md)."""
    m = re.search(r"S(?:0[1-9]|10)", fname)
    return m.group(0) if m else "?"


def classify_file(raw, fname):
    marks = [round(o) for o in raw.annotations.onset if True]
    descs = list(raw.annotations.description)
    dur = raw.times[-1] + 1 / FS
    if marks[:len(SCEN1_MARKS)] == SCEN1_MARKS or (len(marks) == 11 and "PAUSE" in descs):
        return 1
    if marks == SCEN2_MARKS:
        return 2
    if marks == [0, 120, 240, 360]:
        return 0  # 4 x 120 s protocol (unknown)
    return -1  # short test / free trial


def load(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    return raw


def quality(raw):
    x = raw.get_data() * 1e6  # µV
    f, p = welch(x, fs=FS, nperseg=2 * FS, axis=1)
    line = p[:, (f >= 49) & (f <= 51)].mean(1) / p[:, ((f >= 40) & (f < 47)) | ((f > 53) & (f <= 60))].mean(1)
    xf = mne.filter.filter_data(x, FS, 1, 45, verbose=False)
    std = xf.std(1)
    win = FS  # 1-s windows: fraction with peak-to-peak > 150 µV
    n = xf.shape[1] // win
    ptp = np.ptp(xf[:, :n * win].reshape(xf.shape[0], n, win), axis=2)
    bad_sec = (ptp > 150).any(0).mean()
    return dict(
        raw_offset_uV=float(np.median(np.abs(x.mean(1)))),
        std_uV_median=float(np.median(std)), std_uV_max=float(std.max()),
        worst_channel=raw.ch_names[int(std.argmax())],
        flat_channels=[raw.ch_names[i] for i in np.where(std < 1.0)[0]],
        noisy_channels=[raw.ch_names[i] for i in np.where(std > 3 * np.median(std))[0]],
        line_noise_ratio_median=float(np.median(line)), line_noise_ratio_max=float(line.max()),
        frac_seconds_ptp_gt150=float(bad_sec),
        fp_vs_occ_ptp=float(np.median(ptp[[raw.ch_names.index("Fp1-TP"), raw.ch_names.index("Fp2-TP")]]) /
                            np.median(ptp[[raw.ch_names.index("O1-TP"), raw.ch_names.index("O2-TP")]])),
    )


def seg_bandpower(raw, t0, t1):
    x = raw.get_data(start=int(t0 * FS), stop=int(t1 * FS)) * 1e6
    f, p = welch(x, fs=FS, nperseg=2 * FS, axis=1)
    out = {}
    for rname, chs in REGIONS.items():
        idx = [raw.ch_names.index(c) for c in chs]
        pr = p[idx].mean(0)
        out[rname] = {b: float(np.log10(pr[(f >= lo) & (f < hi)].mean())) for b, (lo, hi) in BANDS.items()}
    return out


# ---------------------------------------------------------------- inventory
files = sorted(glob.glob(os.path.join(ROOT, "*.edf")))
inventory, cache = [], {}
for p in files:
    fn = os.path.basename(p)
    raw = load(p)
    sc = classify_file(raw, fn)
    q = quality(raw)
    inventory.append(dict(file=fn, subject=subject_of(fn), scenario=sc, duration=float(raw.times[-1] + 1 / FS),
                          markers=[[round(float(o), 2), str(d)] for o, d in zip(raw.annotations.onset, raw.annotations.description)],
                          meas_date=str(raw.info["meas_date"])[:7], quality=q))
    print(fn, sc, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()}, flush=True)
    if sc in (1, 2) and not re.match(r"^\dd_", fn):
        raw.notch_filter(50, verbose=False)
        raw.filter(1, 45, verbose=False)
        cache[fn] = (raw, sc)

# ---------------------------------------------------------------- segment spectra
seg_spectra = []
for fn, (raw, sc) in cache.items():
    tt = SCEN1 if sc == 1 else SCEN2
    end = raw.times[-1]
    for t0, t1, lbl, name in tt:
        if t0 >= end:
            continue
        seg_spectra.append(dict(file=fn, subject=subject_of(fn), scenario=sc, seg=name, label=lbl,
                                t0=t0, t1=min(t1, end), bp=seg_bandpower(raw, t0, min(t1, end))))

# ---------------------------------------------------------------- windows for ML
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

WIN, STEP, GUARD = 2.0, 0.5, 1.0


def windows(raw, sc, band=(8, 30)):
    tt = SCEN1 if sc == 1 else SCEN2
    x = mne.filter.filter_data(raw.get_data(), FS, band[0], band[1], verbose=False)
    end = raw.times[-1]
    X, y, blk, tcent, segname = [], [], [], [], []
    for bi, (t0, t1, lbl, name) in enumerate(tt):
        t1 = min(t1, end)
        s = t0 + GUARD
        while s + WIN <= t1 - GUARD:
            a = int(s * FS)
            X.append(x[:, a:a + int(WIN * FS)])
            y.append(lbl); blk.append(bi); tcent.append(s + WIN / 2); segname.append(name)
            s += STEP
    return np.array(X, dtype=np.float32), np.array(y), np.array(blk), np.array(tcent), segname


def logbp_feats(X):
    out = []
    for i in range(0, len(X), 500):
        f, p = welch(X[i:i + 500], fs=FS, nperseg=FS // 2, axis=2)
        out.append(np.concatenate([np.log(p[:, :, (f >= lo) & (f < hi)].mean(2)) for lo, hi in BANDS.values()], axis=1))
    return np.concatenate(out)


def ts_model():
    return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=2000))


def bp_model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def cycle_folds(sc, blk):
    """Assign each block to a 'cycle' so that train/test never share a block."""
    if sc == 2:  # 3 cycles: push/walk/neutral
        cyc = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2}
    else:  # 2 cycles: (push#1, walk#1, neutral after them) vs (push#2, walk#2, ...)
        cyc = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0, 8: 1, 9: 1}
    return np.array([cyc[b] for b in blk])


res = dict(within_random=[], within_block=[], drift=[], cross_session=[], loso=[])
data = {}


def per_file(fn, raw, sc):
    X, y, blk, tc, sn = windows(raw, sc)
    out = {}
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = ts_model().fit(X[tr], y[tr]); accs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    out["within_random"] = dict(file=fn, scenario=sc, bacc=float(np.mean(accs)))
    cyc = cycle_folds(sc, blk)
    yt, yp = [], []
    for c in np.unique(cyc):
        tr, te = cyc != c, cyc == c
        if len(np.unique(y[tr])) < 3:
            continue
        m = ts_model().fit(X[tr], y[tr]); yt += list(y[te]); yp += list(m.predict(X[te]))
    out["within_block"] = dict(file=fn, scenario=sc, bacc=float(balanced_accuracy_score(yt, yp)),
                               cm=confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist())
    pb = [i for i, (_, _, l, n) in enumerate(SCEN1 if sc == 1 else SCEN2) if l == 1][:2]
    mask = np.isin(blk, pb)
    Xd, yd = X[mask], (blk[mask] == pb[1]).astype(int)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        m = ts_model().fit(Xd[tr], yd[tr]); accs.append(balanced_accuracy_score(yd[te], m.predict(Xd[te])))
    out["drift"] = dict(file=fn, scenario=sc, bacc_push1_vs_push2=float(np.mean(accs)))
    return fn, (X, y, blk, sc), out


from joblib import Parallel, delayed
for fn, d, out in Parallel(n_jobs=8)(delayed(per_file)(fn, raw, sc) for fn, (raw, sc) in cache.items()):
    data[fn] = d
    for k, v in out.items():
        res[k].append(v)
    print("ML", fn, out["within_random"]["bacc"], out["within_block"]["bacc"], out["drift"]["bacc_push1_vs_push2"], flush=True)
json.dump(dict(inventory=inventory, seg_spectra=seg_spectra, results=res), open(OUT, "w"), indent=1)

# (d) cross-session within subject, same scenario
by_subj = {}
for fn, (X, y, blk, sc) in data.items():
    by_subj.setdefault((subject_of(fn), sc), []).append(fn)
for (s, sc), fns in by_subj.items():
    if len(fns) < 2:
        continue
    for i, te_fn in enumerate(fns):
        tr_fns = [f for f in fns if f != te_fn]
        Xtr = np.concatenate([data[f][0] for f in tr_fns]); ytr = np.concatenate([data[f][1] for f in tr_fns])
        Xte, yte = data[te_fn][0], data[te_fn][1]
        m = ts_model().fit(Xtr, ytr)
        res["cross_session"].append(dict(subject=s, scenario=sc, test=te_fn,
                                         bacc=float(balanced_accuracy_score(yte, m.predict(Xte)))))

# (e) leave-one-subject-out (all scenarios pooled), TS+LR and log-bandpower+sLDA
subjects = sorted({subject_of(f) for f in data})
for s in subjects:
    tr = [f for f in data if subject_of(f) != s]; te = [f for f in data if subject_of(f) == s]
    Xtr = np.concatenate([data[f][0][::2] for f in tr]); ytr = np.concatenate([data[f][1][::2] for f in tr])
    Xte = np.concatenate([data[f][0] for f in te]); yte = np.concatenate([data[f][1] for f in te])
    m = ts_model().fit(Xtr, ytr); b1 = balanced_accuracy_score(yte, m.predict(Xte))
    m2 = bp_model().fit(logbp_feats(Xtr), ytr); b2 = balanced_accuracy_score(yte, m2.predict(logbp_feats(Xte)))
    res["loso"].append(dict(subject=s, n_files=len(te), bacc_ts=float(b1), bacc_bp=float(b2)))
    print("LOSO", s, b1, b2, flush=True)

json.dump(dict(inventory=inventory, seg_spectra=seg_spectra, results=res), open(OUT, "w"), indent=1)
print("done")
