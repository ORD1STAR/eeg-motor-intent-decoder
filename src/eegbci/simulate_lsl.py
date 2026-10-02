"""Stream a recording over LSL in real time, as the headset would (30 ch, 500 Hz, µV).

python -m eegbci.simulate_lsl [file substring] [start_s]
then, in another terminal:  python -m eegbci.realtime lsl NeuroVisor-sim uV
"""
import sys
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from . import config as C
from .data import load_recordings


def main(match="S05-3", start_s=0.0):
    rec = next(r for r in load_recordings(verbose=False) if r.rec_id == match)
    info = StreamInfo("NeuroVisor-sim", "EEG", len(C.CHANNELS), C.SFREQ, "float32", f"sim-{match}")
    ch = info.desc().append_child("channels")
    for name in C.CHANNELS:
        ch.append_child("channel").append_child_value("label", name)
    outlet = StreamOutlet(info, chunk_size=50)
    x = (rec.data * 1e6).astype(np.float32)
    i, n = int(float(start_s) * C.SFREQ), 50                       # 100 ms chunks
    print(f"streaming {match} from {start_s}s ({rec.duration:.0f}s total)", flush=True)
    t0 = time.perf_counter()
    while i < x.shape[1]:
        outlet.push_chunk(x[:, i:i + n].T.tolist())
        i += n
        time.sleep(max(0.0, (i - int(float(start_s) * C.SFREQ)) / C.SFREQ - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main(*sys.argv[1:])
