import os
from glob import glob
import numpy as np
import mne

import os
from glob import glob
import numpy as np
import mne

# --------------------------------------------------------
# Main: Concatenate all train data, export per-file test sets
# --------------------------------------------------------
def process_scenario_concat_train(edf_dir, scenario):
    # dest_dir removed; caller handles final saving
    # Scenario-specific timetable (start, stop, label)
    if scenario == 1:
        timetable = [
            (0, 240, 0), (240, 270, 1), (270, 290, 0), (290, 320, 1),
            (320, 340, 0), (340, 370, 2), (370, 390, 0), (390, 420, 2), (420, 480, 0)
        ]
    elif scenario == 2:
        timetable = [
            (0, 80, 1), (80, 160, 2), (160, 180, 0), (180, 260, 1),
            (260, 340, 2), (340, 360, 0), (360, 430, 1), (430, 500, 2)
        ]
    else:
        raise ValueError("scenario must be 1 or 2")

    X_train, y_train = [], []
    edf_files = glob(os.path.join(edf_dir, '*.edf'))
    for path in edf_files:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.filter(l_freq=1, h_freq=45)
        sfreq = raw.info['sfreq']
        X_test, y_test = [], []
        for idx, (start, stop, lbl) in enumerate(timetable):
            # Include the final tmax sample ONLY for the last timetable segment to avoid
            # dropping the very last sample of the entire recording while still
            # preventing duplicate samples at internal boundaries.
            include_endpoint = idx == len(timetable) - 1
            # Avoid tiny floating-point overruns (e.g., 480.0 > 479.998)
            if include_endpoint:
                stop_adj = min(stop, raw.times[-1])
            else:
                stop_adj = stop
            # Convert timetable times to sample indices and slice the underlying data array
            start_idx = raw.time_as_index(start)[0]
            stop_idx_raw = raw.time_as_index(stop_adj)[0]
            slice_stop = stop_idx_raw + 1 if include_endpoint else stop_idx_raw  # stop is exclusive

            data_block = raw.get_data(start=start_idx, stop=slice_stop).astype(np.float32).T  # (samples, channels)
            n_samples = data_block.shape[0]
            split_idx = int( n_samples)

            # TRAIN window (first 80%)
            X_train.append(data_block)
            y_train.append(np.full((split_idx,), lbl, dtype=np.int8))
            # TEST window (last 20%)
            # if split_idx < n_samples:
            #     X_test.append(data_block[split_idx:])
            #     y_test.append(np.full((n_samples - split_idx,), lbl, dtype=np.int8))
        del raw
        # Save test set for this file
        # if X_test:
        #     X_test = np.concatenate(X_test, axis=0)
        #     y_test = np.concatenate(y_test, axis=0)
        #     subj = os.path.splitext(os.path.basename(path))[0]
        #     np.savez_compressed(os.path.join(dest_dir, f"{subj}_test.npz"), x=X_test, y=y_test)
    # Concatenate and return
    if X_train:
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        return X_train, y_train
    return None, None

# --------------------------------------------------------
# Run both scenarios and export combined training set
# --------------------------------------------------------
if __name__ == "__main__":
    prompt = input("Do you want to export scenarios data (1 test par personne, training par scenario) ? (yes/no)")
    if prompt.lower() == 'yes':
        X1, y1 = process_scenario_concat_train(
            edf_dir="data/scenario1",
            scenario=1,
        )
        X2, y2 = process_scenario_concat_train(
            edf_dir="data/scenario2",
            scenario=2,
        )
        X_comb = np.concatenate([x for x in [X1, X2] if x is not None], axis=0)
        y_comb = np.concatenate([y for y in [y1, y2] if y is not None], axis=0)
        os.makedirs("exports/concat_all_train", exist_ok=True)
        np.savez_compressed("exports/concat_all_train/X_train_all.npz", x=X_comb)
        np.savez_compressed("exports/concat_all_train/y_train_all.npz", y=y_comb)
