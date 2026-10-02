"""python -m eegbci.build_cache  -> builds and caches windows for every recording."""
import time
from joblib import Parallel, delayed
from .data import load_recordings
from .windows import build_windows, fill_cache


def main(reref="none", win_s="2.0"):
    recs = load_recordings()
    t = time.time()
    Parallel(n_jobs=6)(delayed(fill_cache)(r, reref=reref, win_s=float(win_s)) for r in recs)
    for r in recs:
        W = build_windows(r, reref=reref, win_s=float(win_s))
        print(f"{r.rec_id:<8} S{r.scenario} {len(W['meta'])} windows, bad channels: {W['bads']}, "
              f"artifact windows: {(W['meta'].ptp_max > 150).mean():.1%}")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
