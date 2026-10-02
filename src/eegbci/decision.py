"""Part 2 (decision layer): compare causal rules that turn the fusion model's per-window P(active)
into an on/off decision, using realtime.online_metrics' own definitions (balanced state accuracy, false
activations / min of rest, onset detection rate, median latency).

Starts from the per-window LOSO P(active) of the fusion model, cached by classical2.py
(results/classical2_predictions.joblib) -- recomputed there once (cheap on CPU), reused here.

Rules compared, all causal (``step(p) -> (active, p_smooth)``):
  ema_adaptive  the current realtime.BrainSwitchDecoder rule: causal EMA + adaptive threshold
                (running mean of the smoothed probability) with hysteresis.
  ema_fixed     causal EMA + a plain fixed threshold (ablates the adaptive threshold).
  hmm           causal 2-state forward filter (sticky transitions) with a Gaussian emission model on
                logit(p), fit on the training subjects' true labels.
  cusum         two-sided CUSUM change detector on centred logit(p).

For every held-out subject (LOSO), every rule family's hyper-parameters are grid-searched on the other
9 subjects' recordings only (maximising mean balanced state accuracy, tie-break on fewer false
activations/min), then applied once, unchanged, to the held-out subject's recordings.

python -m eegbci.decision run
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .classical2 import PRED_PATH
from .data import load_recordings
from .evaluation import SCHEMES
from .realtime import Output, online_metrics
from .windows import label_windows

RESULTS = C.RESULTS_DIR
WIN_S = 4.0     # window length of the fusion model whose P(active) we consume


# ---------------------------------------------------------------- causal decision rules: step(p) -> (active, p_smooth)
class EmaAdaptive:
    """The rule currently in realtime.BrainSwitchDecoder: causal EMA (tau_s) of p, 'active' when that
    smoothed probability exceeds its own causal running mean (thr_tau_s) by >= margin (+hysteresis,
    min_on_s consecutive windows above threshold before switching on)."""

    def __init__(self, tau_s=8.0, thr_tau_s=120.0, margin=0.02, min_on_s=1.0, step_s=C.STEP_S, warm=40):
        self.a = 1.0 - np.exp(-step_s / tau_s)
        self.ta = 1.0 - np.exp(-step_s / thr_tau_s)
        self.margin, self.warm = margin, warm
        self.min_on = max(1, int(round(min_on_s / step_s)))
        self.p = None
        self.m_thr = None
        self.k = 0
        self.active = False
        self._above = 0

    def step(self, p):
        self.p = p if self.p is None else (1 - self.a) * self.p + self.a * p
        thr = self.p if self.m_thr is None else self.m_thr
        on, off = self.p > thr + self.margin, self.p < thr - self.margin
        self.k += 1
        self.m_thr = self.p if self.m_thr is None else (
            self.m_thr + (self.p - self.m_thr) / self.k if self.k <= self.warm else (1 - self.ta) * self.m_thr + self.ta * self.p)
        self._above = self._above + 1 if on else 0
        if self._above >= self.min_on:
            self.active = True
        if off:
            self.active = False
        return self.active, self.p


class EmaFixed:
    """Ablation: causal EMA + a plain fixed threshold (no adaptive baseline, no hysteresis)."""

    def __init__(self, tau_s=8.0, thr=0.5, step_s=C.STEP_S):
        self.a = 1.0 - np.exp(-step_s / tau_s)
        self.thr = thr
        self.p = None
        self.active = False

    def step(self, p):
        self.p = p if self.p is None else (1 - self.a) * self.p + self.a * p
        self.active = self.p >= self.thr
        return self.active, self.p


def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))


class HMMFilter:
    """Causal forward filter (2-state sticky HMM) on logit(p). Gaussian emission per state
    (mu/sigma fit on training-subject true labels); self-transition probabilities set from a target
    mean dwell time in each state (dwell_s -> stay = exp(-step_s/dwell_s), same parametrisation as the
    EMA time constants elsewhere in this project). p_smooth returned is the filtered posterior P(active)."""

    def __init__(self, mu0, s0, mu1, s1, dwell0_s=8.0, dwell1_s=8.0, step_s=C.STEP_S, eps=1e-3):
        self.mu = (mu0, mu1)
        self.s2 = (max(s0, eps) ** 2, max(s1, eps) ** 2)
        self.stay0 = np.exp(-step_s / dwell0_s)
        self.stay1 = np.exp(-step_s / dwell1_s)
        self.belief = np.array([0.5, 0.5])
        self.active = False

    def step(self, p):
        x = _logit(p)
        b0 = self.belief[0] * self.stay0 + self.belief[1] * (1 - self.stay1)
        b1 = self.belief[1] * self.stay1 + self.belief[0] * (1 - self.stay0)
        lik0 = np.exp(-0.5 * (x - self.mu[0]) ** 2 / self.s2[0]) / np.sqrt(self.s2[0])
        lik1 = np.exp(-0.5 * (x - self.mu[1]) ** 2 / self.s2[1]) / np.sqrt(self.s2[1])
        u0, u1 = b0 * lik0, b1 * lik1
        z = u0 + u1 + 1e-300
        self.belief = np.array([u0 / z, u1 / z])
        self.active = bool(self.belief[1] > 0.5)
        return self.active, float(self.belief[1])


class Cusum:
    """Two-sided CUSUM change detector on centred logit(p): drift k, decision threshold h. Switches
    'active' when the upward sum exceeds h, 'rest' when the downward sum exceeds h (both reset on any
    trigger). p_smooth is a plain causal EMA(report_tau_s) of p, kept only for plotting/AUC -- the CUSUM
    statistics themselves drive the decision."""

    def __init__(self, c=0.0, k=0.3, h=2.0, step_s=C.STEP_S, report_tau_s=8.0):
        self.c, self.k, self.h = c, k, h
        self.gp = self.gn = 0.0
        self.active = False
        self.a = 1.0 - np.exp(-step_s / report_tau_s)
        self.p = None

    def step(self, p):
        self.p = p if self.p is None else (1 - self.a) * self.p + self.a * p
        x = _logit(p) - self.c
        self.gp = max(0.0, self.gp + x - self.k)
        self.gn = max(0.0, self.gn - x - self.k)
        if self.gp > self.h:
            self.active, self.gp, self.gn = True, 0.0, 0.0
        elif self.gn > self.h:
            self.active, self.gp, self.gn = False, 0.0, 0.0
        return self.active, self.p


# ---------------------------------------------------------------- evaluation harness
def simulate(rule_factory, t, p):
    dec = rule_factory()
    outs = []
    for ti, pi in zip(t, p):
        active, psm = dec.step(float(pi))
        outs.append(Output(t=float(ti), p_raw=float(pi), p=float(psm), active=bool(active), artifact=False))
    return outs


def per_recording_metrics(rule_factory, rid_list, preds, recs_by_id, win_s=WIN_S):
    out = {}
    for rid in rid_list:
        outs = simulate(rule_factory, preds[rid]["t"], preds[rid]["p"])
        out[rid] = online_metrics(recs_by_id[rid], outs, win_s)
    return out


def fold_score(metrics):
    """Scalar used ONLY to pick hyper-parameters on the training subjects: mean balanced state accuracy,
    tie-broken by fewer false activations / min of rest."""
    bacc = np.array([m["state_acc"] for m in metrics.values() if not np.isnan(m["state_acc"])])
    fo = np.array([m["false_on_per_min_rest"] for m in metrics.values() if not np.isnan(m["false_on_per_min_rest"])])
    if len(bacc) == 0:
        return (-1e9, 0.0)
    return (round(float(bacc.mean()), 4), -float(fo.mean()) if len(fo) else 0.0)


def tune_family(Cls, grid, fixed_kwargs, rid_list, preds, recs_by_id):
    best = None
    for kw in grid:
        full_kw = {**fixed_kwargs, **kw}
        m = per_recording_metrics(lambda fk=full_kw: Cls(**fk), rid_list, preds, recs_by_id)
        key = fold_score(m)
        if best is None or key > best[0]:
            best = (key, full_kw)
    return best[1]


def true_labels(rec, t, win_s=WIN_S):
    meta = pd.DataFrame(dict(t_end=t, t_start=t - win_s))
    lab, blk, cyc, tib = label_windows(meta, rec.blocks, rec.duration, 0.0)
    y, ok = SCHEMES["task_rest"](lab)
    return y, ok


def fit_emissions(rid_list, preds, recs_by_id):
    xs0, xs1 = [], []
    for rid in rid_list:
        t, p = preds[rid]["t"], preds[rid]["p"]
        y, ok = true_labels(recs_by_id[rid], t)
        x = np.array([_logit(pi) for pi in p])
        xs0.append(x[ok & (y == 0)])
        xs1.append(x[ok & (y == 1)])
    x0, x1 = np.concatenate(xs0), np.concatenate(xs1)
    return float(x0.mean()), float(x0.std() + 1e-6), float(x1.mean()), float(x1.std() + 1e-6)


EMA_ADAPTIVE_GRID = [dict(tau_s=ta, thr_tau_s=tt, margin=mg)
                     for ta in (4.0, 8.0, 12.0, 16.0) for tt in (60.0, 120.0, 180.0) for mg in (0.01, 0.02, 0.04)]
EMA_FIXED_GRID = [dict(tau_s=ta, thr=th) for ta in (4.0, 8.0, 12.0, 16.0) for th in (0.45, 0.5, 0.55, 0.6)]
CUSUM_GRID = [dict(k=k, h=h) for k in (0.1, 0.3, 0.5) for h in (1.0, 2.0, 3.0, 5.0)]
HMM_GRID = [dict(dwell0_s=d0, dwell1_s=d1) for d0 in (4.0, 8.0, 16.0, 30.0) for d1 in (4.0, 8.0, 16.0, 30.0)]
BASELINE_KW = dict(tau_s=8.0, thr_tau_s=120.0, margin=0.02)   # realtime.BrainSwitchDecoder defaults


def run():
    recs = load_recordings(verbose=True)
    recs_by_id = {r.rec_id: r for r in recs}
    preds = joblib.load(PRED_PATH)
    preds = {rid: v for rid, v in preds.items() if rid in recs_by_id}
    subj = {rid: recs_by_id[rid].subject for rid in preds}
    subjects = sorted(set(subj.values()))

    test_metrics = {name: {} for name in ("ema_adaptive", "ema_fixed", "hmm", "cusum")}
    chosen = {name: [] for name in test_metrics}
    baseline_metrics = {}

    for s in subjects:
        train_rids = [r for r in preds if subj[r] != s]
        test_rids = [r for r in preds if subj[r] == s]

        kw = tune_family(EmaAdaptive, EMA_ADAPTIVE_GRID, {}, train_rids, preds, recs_by_id)
        test_metrics["ema_adaptive"].update(per_recording_metrics(lambda fk=kw: EmaAdaptive(**fk), test_rids, preds, recs_by_id))
        chosen["ema_adaptive"].append(kw)

        kw = tune_family(EmaFixed, EMA_FIXED_GRID, {}, train_rids, preds, recs_by_id)
        test_metrics["ema_fixed"].update(per_recording_metrics(lambda fk=kw: EmaFixed(**fk), test_rids, preds, recs_by_id))
        chosen["ema_fixed"].append(kw)

        mu0, s0, mu1, s1 = fit_emissions(train_rids, preds, recs_by_id)
        kw = tune_family(HMMFilter, HMM_GRID, dict(mu0=mu0, s0=s0, mu1=mu1, s1=s1), train_rids, preds, recs_by_id)
        test_metrics["hmm"].update(per_recording_metrics(lambda fk=kw: HMMFilter(**fk), test_rids, preds, recs_by_id))
        chosen["hmm"].append(kw)

        c = 0.5 * (mu0 + mu1)
        kw = tune_family(Cusum, CUSUM_GRID, dict(c=c), train_rids, preds, recs_by_id)
        test_metrics["cusum"].update(per_recording_metrics(lambda fk=kw: Cusum(**fk), test_rids, preds, recs_by_id))
        chosen["cusum"].append(kw)

        baseline_metrics.update(per_recording_metrics(lambda fk=BASELINE_KW: EmaAdaptive(**fk), test_rids, preds, recs_by_id))

        print(f"{s}: ema_adaptive {chosen['ema_adaptive'][-1]} | ema_fixed {chosen['ema_fixed'][-1]} | "
              f"hmm dwell={chosen['hmm'][-1]['dwell0_s']:g}/{chosen['hmm'][-1]['dwell1_s']:g} mu={mu0:.2f}/{mu1:.2f} | "
              f"cusum {chosen['cusum'][-1]}", flush=True)

    def agg(metrics, key):
        v = np.array([m[key] for m in metrics.values() if not np.isnan(m.get(key, np.nan))])
        return float(np.median(v)) if len(v) else float("nan")

    rows = {}
    for name, metrics in test_metrics.items():
        rows[name] = dict(bacc_median=agg(metrics, "state_acc"), false_on_median=agg(metrics, "false_on_per_min_rest"),
                           det_rate_median=agg({k: dict(v, det_rate=(v["detected"] / v["onsets"] if v["onsets"] else np.nan))
                                                for k, v in metrics.items()}, "det_rate"),
                           latency_median=agg(metrics, "latency_s"), n=len(metrics))
    rows["baseline_fixed_params"] = dict(
        bacc_median=agg(baseline_metrics, "state_acc"), false_on_median=agg(baseline_metrics, "false_on_per_min_rest"),
        det_rate_median=agg({k: dict(v, det_rate=(v["detected"] / v["onsets"] if v["onsets"] else np.nan))
                             for k, v in baseline_metrics.items()}, "det_rate"),
        latency_median=agg(baseline_metrics, "latency_s"), n=len(baseline_metrics))

    print("\nDecision-layer comparison (median over held-out recordings, LOSO, rules tuned on train subjects only):")
    print(f"{'rule':<22}{'bacc':>8}{'false_on/min':>14}{'onset rate':>12}{'latency(s)':>12}{'n':>5}")
    for name, r in rows.items():
        print(f"{name:<22}{r['bacc_median']:>8.3f}{r['false_on_median']:>14.3f}{r['det_rate_median']:>12.3f}"
              f"{r['latency_median']:>12.2f}{r['n']:>5}")

    json.dump(dict(rows=rows, chosen_params=chosen,
                    per_recording={name: m for name, m in test_metrics.items()},
                    baseline_fixed_params_per_recording=baseline_metrics),
              open(RESULTS / "decision_results.json", "w"), indent=1)
    print(f"\nsaved {RESULTS / 'decision_results.json'}")


if __name__ == "__main__":
    globals().get(sys.argv[1], run)() if len(sys.argv) > 1 else run()
