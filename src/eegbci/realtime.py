"""Real-time brain switch (idle vs active).

    python -m eegbci.realtime train               # fit on all recordings, save models/brain_switch.joblib
    python -m eegbci.realtime replay S05          # leave S05 out, stream S05's recordings through the online decoder
    python -m eegbci.realtime lsl [stream_name]   # live decoding from an LSL EEG stream (30 ch, 500 Hz, µV or V)

The streaming decoder reproduces the offline features sample for sample: same causal filters,
same decimation phase, same OAS covariances, same causal re-centering, same EMA smoothing.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
from pyriemann.estimation import Covariances

from . import config as C
from .evaluation import make_model
from .features import CausalRecenter, logm, upper
from .preprocessing import CausalFilter
from .windows import DECIM, FS_D

MODEL_PATH = C.MODELS_DIR / "brain_switch.joblib"
PO = [C.CHANNELS.index(c) for c in ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]]
FINAL_CFG = dict(
    features=[dict(bands=[0, 1, 2, 3, 4], channels=None, kind="logbp", tau_s=60.0),
              dict(bands=[1, 2, 3, 4], channels=PO, kind="ts", tau_s=60.0)],
    win_s=4.0, step_s=C.STEP_S, smooth_s=8.0, C_reg=0.01)


def offline_features(W, cfg=FINAL_CFG):
    from .features import recording_features
    parts = [recording_features(W, bands=f["bands"], align="causal", tau_s=f["tau_s"], kind=f["kind"],
                                channels=f["channels"]) for f in cfg["features"]]
    return np.concatenate(parts, axis=1)


def train_model(recs, cfg=FINAL_CFG, ensemble=True):
    from .evaluation import Bundle
    from .windows import build_windows
    W = {r.rec_id: build_windows(r, win_s=cfg["win_s"]) for r in recs}
    B = Bundle(recs, W, {rid: offline_features(W[rid], cfg) for rid in W})
    X, y, _ = B.subset(list(W), "task_rest")
    model = make_model("lr", cfg["C_reg"]).fit(np.concatenate(X), np.concatenate(y))
    bundle = dict(model=model, cfg=cfg, trained_on=sorted({r.rec_id for r in recs}))
    if ensemble:
        bundle["net"] = train_shallow(recs)
    return bundle


def train_shallow(recs, seed=0):
    """ShallowConvNet on 2 s causally aligned epochs; one held-out person drives early stopping."""
    import torch
    from .deep import ShallowNet, causal_euclidean_align, train
    from .evaluation import Bundle
    from .windows import build_windows
    torch.manual_seed(seed); np.random.seed(seed)
    W2 = {r.rec_id: build_windows(r, win_s=2.0) for r in recs}
    B2 = Bundle(recs, W2, {rid: np.zeros((len(W2[rid]["meta"]), 1)) for rid in W2})
    subjects = sorted({r.subject for r in recs})
    val = subjects[-1]
    data = {}
    for r in recs:
        y, ok, _, _ = B2.labels(r.rec_id, "task_rest")
        Xa = causal_euclidean_align(np.asarray(W2[r.rec_id]["raw"]) / 10.0)
        data[r.rec_id] = (Xa[ok], y[ok], r.subject)
    cat = lambda f: (np.concatenate([d[0] for d in data.values() if f(d[2])]),
                     np.concatenate([d[1] for d in data.values() if f(d[2])]))
    (Xtr, ytr), (Xva, yva) = cat(lambda s: s != val), cat(lambda s: s == val)
    net, vauc, n_ep = train(ShallowNet(n_t=Xtr.shape[2]), Xtr, ytr, Xva, yva)
    return dict(state={k: v.cpu() for k, v in net.state_dict().items()}, n_t=Xtr.shape[2], val_auc=vauc)


@dataclass
class Output:
    t: float            # seconds since stream start (end of the analysed window)
    p_raw: float        # P(active) for this window (ensemble if available)
    p: float            # smoothed P(active)
    active: bool        # decision after hysteresis
    artifact: bool
    p_riemann: float = float("nan")
    p_deep: float = float("nan")
    thr: float = float("nan")


class BrainSwitchDecoder:
    def __init__(self, bundle, threshold="adaptive", margin=0.02, thr_on=0.6, thr_off=0.45, min_on_s=1.0,
                 thr_tau_s=120.0, use_deep=True, bads=(), reset_at_s=None):
        """threshold='adaptive': active when the smoothed probability exceeds its own causal running
        mean (tau thr_tau_s) by ``margin`` (hysteresis), which removes the per-person offset without
        labels. threshold='fixed': classic thr_on / thr_off."""
        self.model, self.cfg = bundle["model"], bundle["cfg"]
        from .preprocessing import interpolation_matrix
        self.M = interpolation_matrix(C.CHANNELS, list(bads)) if bads else None   # frozen at calibration
        self.reset_at_s, self._reset_done = reset_at_s, reset_at_s is None
        self.threshold, self.margin = threshold, margin
        self.ta = 1.0 - np.exp(-self.cfg["step_s"] / thr_tau_s)
        self.m_thr, self.k = None, 0
        self.net = None
        if use_deep and "net" in bundle:
            import torch
            from .deep import ShallowNet
            self.torch = torch
            self.net = ShallowNet(n_t=bundle["net"]["n_t"])
            self.net.load_state_dict(bundle["net"]["state"]); self.net.eval()
            self.n_t = bundle["net"]["n_t"]
            self.ea_m, self.ea_k = None, 0
            self.ea_a = 1.0 - np.exp(-self.cfg["step_s"] / 60.0)
        bands_needed = sorted({b for f in self.cfg["features"] for b in f["bands"]})
        all_bands = list(C.BANDS)
        self.filters = {b: CausalFilter(*all_bands[b], len(C.CHANNELS)) for b in bands_needed}
        self.broad = CausalFilter(1, 40, len(C.CHANNELS))
        self.win = int(self.cfg["win_s"] * FS_D)
        self.step = int(self.cfg["step_s"] * FS_D)
        self.buf = {b: np.zeros((len(C.CHANNELS), 0)) for b in bands_needed}
        self.bbuf = np.zeros((len(C.CHANNELS), 0))
        self.recenter = [{b: CausalRecenter(f["tau_s"], self.cfg["step_s"]) for b in f["bands"]}
                         for f in self.cfg["features"]]
        self.cov = Covariances("oas")
        self.n_in = 0                     # raw samples received
        self.n_dec = 0                    # decimated samples produced
        self.next_end = self.win
        self.a = 1.0 - np.exp(-self.cfg["step_s"] / self.cfg["smooth_s"])
        self.p = None
        self.active = False
        self.thr_on, self.thr_off = thr_on, thr_off
        self.min_on = int(min_on_s / self.cfg["step_s"])
        self._above = 0
        self.next_end_deep = self.n_t if self.net is not None else np.inf
        self.p_deep_last = float("nan")

    def push(self, chunk):
        """chunk: (30, n) volts at 500 Hz. Returns a list of Output (one per completed window)."""
        outs = []
        if self.M is not None:
            chunk = self.M @ chunk
        filtered = {b: f(chunk) for b, f in self.filters.items()}
        broad = self.broad(chunk)
        keep = np.arange(chunk.shape[1])[(self.n_in + np.arange(chunk.shape[1])) % DECIM == 0]
        self.n_in += chunk.shape[1]
        for b in self.buf:
            self.buf[b] = np.concatenate([self.buf[b], filtered[b][:, keep]], axis=1)
        self.bbuf = np.concatenate([self.bbuf, broad[:, keep]], axis=1)
        self.n_dec += len(keep)
        while True:
            # the deep branch runs on its own 2 s grid (as offline in ensemble.py); each 4 s window uses the
            # latest deep output
            if self.next_end_deep <= self.next_end and self.n_dec >= self.next_end_deep:
                stop = self.bbuf.shape[1] - (self.n_dec - self.next_end_deep)
                self.p_deep_last = self._deep_at(stop)
                self.next_end_deep += self.step
            elif self.n_dec >= self.next_end:
                outs.append(self._window(self.n_dec - self.next_end))
                self.next_end += self.step
            else:
                break
        n_keep = self.win + 4 * self.step               # trim only after all windows were emitted
        for b in self.buf:
            self.buf[b] = self.buf[b][:, -n_keep:]
        self.bbuf = self.bbuf[:, -n_keep:]
        return outs

    def _window(self, off):
        sl = slice(self.buf_len() - off - self.win, self.buf_len() - off)
        if not self._reset_done and self.next_end / FS_D - self.cfg["win_s"] >= self.reset_at_s:
            self.recenter = [{b: CausalRecenter(f["tau_s"], self.cfg["step_s"]) for b in f["bands"]}
                             for f in self.cfg["features"]]
            self._reset_done = True
        feats = []
        for f, rc in zip(self.cfg["features"], self.recenter):
            for b in f["bands"]:
                x = self.buf[b][:, sl]
                Cb = self.cov.transform(x[None].astype(np.float64))[0].astype(np.float32).astype(np.float64)
                if f["channels"] is not None:
                    Cb = Cb[np.ix_(f["channels"], f["channels"])]
                A = rc[b].step(Cb)
                feats.append(np.log(np.maximum(np.diag(A), 1e-12)) if f["kind"] == "logbp" else upper(logm(A)))
        p_r = float(self.model.predict_proba(np.concatenate(feats)[None])[0, 1])
        p_d = self.p_deep_last if self.net is not None else float("nan")
        p_raw = p_r if self.net is None else 0.5 * (p_r + p_d)
        self.p = p_raw if self.p is None else (1 - self.a) * self.p + self.a * p_raw
        if self.threshold == "adaptive":
            thr = self.p if self.m_thr is None else self.m_thr
            on, off = self.p > thr + self.margin, self.p < thr - self.margin
            self.k += 1
            self.m_thr = self.p if self.m_thr is None else (
                self.m_thr + (self.p - self.m_thr) / self.k if self.k <= 40 else (1 - self.ta) * self.m_thr + self.ta * self.p)
        else:
            thr, on, off = self.thr_on, self.p >= self.thr_on, self.p <= self.thr_off
        self._above = self._above + 1 if on else 0
        if self._above >= self.min_on:
            self.active = True
        if off:
            self.active = False
        nonfp = [i for i in range(len(C.CHANNELS)) if C.CHANNELS[i] not in C.FRONTOPOLAR]
        ptp = np.ptp(self.bbuf[nonfp, sl], axis=1).max() * 1e6
        return Output(t=(self.next_end) / FS_D, p_raw=p_raw, p=self.p, active=self.active,
                      artifact=ptp > C.PTP_REJECT_UV, p_riemann=p_r, p_deep=p_d, thr=float(thr))

    def _deep_at(self, end):
        """ShallowConvNet on the 2 s of the broadband buffer ending at index ``end``, causally aligned."""
        from .features import invsqrtm
        x = (self.bbuf[:, end - self.n_t:end] * 1e6).astype(np.float32) / 10.0
        c = x.astype(np.float64) @ x.astype(np.float64).T / x.shape[1]
        ref = c if self.ea_m is None else self.ea_m
        xa = (invsqrtm(ref) @ x).astype(np.float32)
        self.ea_k += 1
        self.ea_m = c if self.ea_m is None else (
            self.ea_m + (c - self.ea_m) / self.ea_k if self.ea_k <= 40 else (1 - self.ea_a) * self.ea_m + self.ea_a * c)
        with self.torch.no_grad():
            return float(self.torch.softmax(self.net(self.torch.tensor(xa[None, None])), 1)[0, 1])

    def buf_len(self):
        return next(iter(self.buf.values())).shape[1]


def replay(rec, bundle, chunk_s=0.1, **kw):
    dec = BrainSwitchDecoder(bundle, **kw)
    n = int(chunk_s * C.SFREQ)
    outs, lat = [], []
    for i in range(0, rec.data.shape[1], n):
        t0 = time.perf_counter()
        o = dec.push(rec.data[:, i:i + n])
        if o:
            lat.append((time.perf_counter() - t0) / len(o))
        outs += o
    return outs, np.array(lat)


def online_metrics(rec, outs, win_s, guard=C.GUARD_S):
    """Switch metrics on the protocol: AUC, time in correct state, false activations / min of rest,
    detection rate and latency of rest->task onsets."""
    from sklearn.metrics import roc_auc_score
    t = np.array([o.t for o in outs]); p = np.array([o.p for o in outs]); act = np.array([o.active for o in outs])
    truth = np.full(len(t), -1)
    for t0, t1, l, name, cyc in rec.blocks:
        if l is None:
            continue
        m = (t - win_s >= t0 + guard) & (t <= t1 - guard)
        truth[m] = int(l != C.NEUTRAL)
    ok = truth >= 0
    rises = np.flatnonzero(np.diff(act.astype(int)) == 1) + 1
    rest_time = sum(t1 - t0 for t0, t1, l, *_ in rec.blocks if l == C.NEUTRAL) / 60
    false_on = sum(1 for i in rises if truth[i] == 0)
    lats, hits, n_on = [], 0, 0
    bl = [b for b in rec.blocks if b[2] is not None]
    for a, b in zip(bl[:-1], bl[1:]):
        if a[2] == C.NEUTRAL and b[2] != C.NEUTRAL:
            n_on += 1
            r = t[rises][(t[rises] > b[0]) & (t[rises] <= b[1])]      # must switch ON inside the block
            if len(r):
                hits += 1; lats.append(r[0] - b[0])
    return dict(auc=float(roc_auc_score(truth[ok], p[ok])) if len(set(truth[ok])) == 2 else np.nan,
                state_acc=float(np.mean([(act[ok & (truth == c)] == bool(c)).mean() for c in (0, 1)
                                         if (ok & (truth == c)).any()])),       # balanced over rest / active
                false_on_per_min_rest=false_on / max(rest_time, 1e-9), onsets=n_on, detected=hits,
                latency_s=float(np.median(lats)) if lats else np.nan)


def main():
    from .data import load_recordings
    cmd = sys.argv[1]
    if cmd == "train":
        recs = load_recordings()
        b = train_model(recs)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(b, MODEL_PATH)
        print(f"saved {MODEL_PATH} (trained on {len(b['trained_on'])} recordings)")
    elif cmd == "replay":
        held = sys.argv[2]
        recs = load_recordings()
        b = train_model([r for r in recs if r.subject != held], ensemble="--riemann-only" not in sys.argv)
        from .windows import build_windows
        import json
        rows = []
        for r in [r for r in recs if r.subject == held]:
            W = build_windows(r, win_s=b["cfg"]["win_s"])
            outs, lat = replay(r, b, bads=W["bads"], reset_at_s=120.0 if r.scenario == 1 else None)
            p_off = b["model"].predict_proba(offline_features(W, b["cfg"]))[:, 1]
            p_on = np.array([o.p_riemann for o in outs])
            n = min(len(p_on), len(p_off))
            m = online_metrics(r, outs, b["cfg"]["win_s"])
            print(f"{r.rec_id} S{r.scenario}: windows {len(outs)} | online/offline max |dp| {np.abs(p_on[:n] - p_off[:n]).max():.2e} "
                  f"| {1e3 * lat.mean():.1f} ms/window | AUC {m['auc']:.3f} | state acc {m['state_acc']:.2f} | "
                  f"false on {m['false_on_per_min_rest']:.2f}/min rest | onsets {m['detected']}/{m['onsets']} "
                  f"| latency {m['latency_s']:.1f}s")
            rows.append(dict(rec=r.rec_id, scenario=r.scenario, auc=m["auc"], state_acc=m["state_acc"],
                             false_on=m["false_on_per_min_rest"], detected=m["detected"], onsets=m["onsets"],
                             latency=m["latency_s"], ms=1e3 * lat.mean(), parity=float(np.abs(p_on[:n] - p_off[:n]).max())))
            r.release()
        json.dump(rows, open(C.RESULTS_DIR / f"replay_{held}.json", "w"), indent=1)
    elif cmd == "lsl":
        from pylsl import StreamInlet, resolve_byprop
        name = sys.argv[2] if len(sys.argv) > 2 else None
        streams = resolve_byprop("name", name, timeout=10) if name else resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            raise SystemExit("no LSL EEG stream found")
        inlet = StreamInlet(streams[0])
        scale = 1e-6 if (len(sys.argv) > 3 and sys.argv[3] == "uV") else 1.0
        dec = BrainSwitchDecoder(joblib.load(MODEL_PATH))
        print("decoding... (Ctrl+C to stop)")
        while True:
            chunk, ts = inlet.pull_chunk(timeout=0.2, max_samples=250)
            if chunk:
                for o in dec.push(np.asarray(chunk).T * scale):
                    print(f"t={o.t:7.1f}s  P(active)={o.p:.2f}  {'ACTIVE' if o.active else 'idle  '}"
                          f"{'  [artifact]' if o.artifact else ''}", flush=True)


if __name__ == "__main__":
    main()
