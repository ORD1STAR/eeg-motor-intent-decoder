"""Evaluation: causal smoothing, fold handling, smoothing over unlabelled windows, null thread-safety,
time-only control and the reported p-values."""
from functools import partial

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eegbci.evaluation import (Bundle, ema_causal, eval_groups, eval_within, loso_groups, make_model,
                               null_distribution, random_shifts, report, summarize, time_features)
from tests.helpers import FakeRec, toy_meta

D = 240.0
BLOCKS = [(0, 30, 0, "rest#1", 0), (30, 50, 1, "push#1", 0), (50, 70, 0, "rest#2", 0), (70, 100, 2, "walk#1", 1),
          (100, 140, 0, "rest#3", 1), (140, 160, 1, "push#2", 1), (160, 180, 0, "rest#4", 2), (180, 220, 2, "walk#2", 2),
          (220, 240, 0, "rest#5", 2)]


def toy_bundle(n_rec=4, signal=1.0, seed=0, blocks=BLOCKS, duration=D):
    rng = np.random.default_rng(seed)
    recs, W, F = [], {}, {}
    for i in range(n_rec):
        r = FakeRec(data=np.zeros((1, int(duration * 500))), rec_id=f"T{i:02d}-1", subject=f"T{i:02d}", blocks=blocks)
        meta = toy_meta(duration, win_s=2.0, hop_s=0.5)
        active = np.zeros(len(meta))
        for t0, t1, l, *_ in blocks:
            active[(meta.t_end > t0) & (meta.t_end <= t1)] = float(l is not None and l > 0)
        F[r.rec_id] = rng.standard_normal((len(meta), 4)) + signal * active[:, None] * np.array([1, 0.5, 0, 0])
        recs.append(r); W[r.rec_id] = dict(meta=meta)
    return Bundle(recs, W, F)


# ---------------------------------------------------------------- ema_causal
def test_ema_causal_is_causal_and_matches_recursion():
    rng = np.random.default_rng(0)
    P = rng.random((300, 2))
    out = ema_causal(P, 8.0)
    a = 1 - np.exp(-0.5 / 8.0)
    ref = [P[0]]
    for p in P[1:]:
        ref.append((1 - a) * ref[-1] + a * p)
    np.testing.assert_allclose(out, np.array(ref), rtol=0, atol=1e-15)
    for k in (1, 77, 299):
        P2 = P.copy()
        P2[k:] = rng.random(P2[k:].shape)
        assert np.array_equal(ema_causal(P2, 8.0)[:k], out[:k])
    assert ema_causal(P, 0.0) is P


# ---------------------------------------------------------------- eval_within
def test_eval_within_skips_folds_whose_training_set_lacks_a_class():
    # cycle 0 holds the only rest block -> leaving cycle 0 out leaves a single-class training set
    blocks = [(0, 20, 0, "rest", 0), (20, 40, 1, "push", 0), (40, 70, 1, "push2", 1), (70, 120, 2, "walk", 2)]
    B = toy_bundle(1, blocks=blocks, duration=120.0)
    res = eval_within(B, "task_rest")                       # must not raise
    rid = next(iter(B.F))
    (_, ), (y, ), (G, ) = B.subset([rid], "task_rest")
    assert res[rid]["n"] == int((G[:, 0] != 0).sum())       # only folds 1 and 2 evaluated
    assert "auc" not in res[rid]                            # pooled test set is single-class -> no AUC


def test_eval_within_auc_pools_all_folds():
    B = toy_bundle(1, signal=1.5)
    rid = next(iter(B.F))
    res = eval_within(B, "task_rest")
    (X, ), (y, ), (G, ) = B.subset([rid], "task_rest")
    yt, pp = [], []
    for c in np.unique(G[:, 0]):
        tr, te = G[:, 0] != c, G[:, 0] == c
        m = make_model("lr", 0.1).fit(X[tr], y[tr])
        yt.append(y[te]); pp.append(m.predict_proba(X[te])[:, 1])
    assert res[rid]["auc"] == pytest.approx(roc_auc_score(np.concatenate(yt), np.concatenate(pp)))


# ---------------------------------------------------------------- eval_groups
def test_eval_groups_smooths_over_all_windows_then_scores_labelled_ones():
    B = toy_bundle(3, signal=1.0)
    test, train = ["T00-1"], ["T01-1", "T02-1"]
    res = eval_groups(B, "task_rest", [(test, train)], smooth_s=4.0)
    X, Y, _ = B.subset(train, "task_rest")
    m = make_model("lr", 0.1).fit(np.concatenate(X), np.concatenate(Y))
    y, ok, _, _ = B.labels("T00-1", "task_rest")
    P_all = ema_causal(m.predict_proba(B.F["T00-1"]), 4.0)[ok]                    # online-like: EMA over every window
    P_lab = ema_causal(m.predict_proba(B.F["T00-1"][ok]), 4.0)                    # EMA over labelled windows only
    assert res["T00-1"]["auc"] == pytest.approx(roc_auc_score(y[ok], P_all[:, 1]))
    assert not np.isclose(roc_auc_score(y[ok], P_all[:, 1]), roc_auc_score(y[ok], P_lab[:, 1]))


def test_loso_groups_are_disjoint_by_subject():
    B = toy_bundle(4)
    for test, train in loso_groups(B):
        assert not ({B.recs[r].subject for r in test} & {B.recs[r].subject for r in train})
        assert set(test) | set(train) == set(B.recs)


# ---------------------------------------------------------------- null distribution
def test_random_shifts_range():
    B = toy_bundle(4)
    rng = np.random.default_rng(0)
    for _ in range(20):
        for rid, s in random_shifts(B, rng, min_s=10.0).items():
            assert 10.0 <= s <= B.recs[rid].duration - 10.0


def test_random_shifts_stay_within_analysed_span():
    B = toy_bundle(2)
    for rid, r in B.recs.items():
        r.blocks = [(0, 20, None, "EC", 0)] + [(max(a, 20), b, l, n, c) for a, b, l, n, c in r.blocks if b > 20]
    lo = 20.0
    for _ in range(20):
        for rid, s in random_shifts(B, np.random.default_rng(_), min_s=5.0).items():
            assert 5.0 <= s <= B.recs[rid].duration - lo - 5.0


def test_null_distribution_is_thread_safe_and_deterministic():
    B = toy_bundle(4, signal=0.8)
    fn = partial(eval_groups, B, "task_rest", loso_groups(B), "lr", smooth_s=2.0)
    a = null_distribution(lambda s: fn(shifts=s), B, n_perm=8, seed=3, n_jobs=1, key="auc")
    b = null_distribution(lambda s: fn(shifts=s), B, n_perm=8, seed=3, n_jobs=4, key="auc")
    assert np.array_equal(a, b)
    obs = summarize(fn(shifts=None), "auc")["mean"]
    assert obs > np.max(a)                                  # a real effect beats the drift-null


def test_report_p_value():
    obs = {"a": dict(auc=0.7), "b": dict(auc=0.6)}
    s = report("toy", obs, np.array([0.5, 0.66, 0.7, 0.4]), key="auc")
    assert s["mean"] == pytest.approx(0.65)
    assert s["p"] == pytest.approx((1 + 2) / (1 + 4))       # (1 + #null >= obs) / (1 + n_perm)


def test_report_handles_empty_results():
    s = report("empty", {}, None, key="auc")
    assert np.isnan(s["median"]) and np.isnan(s["mean"])


# ---------------------------------------------------------------- time-only control
def test_time_features():
    B = toy_bundle(2)
    tf = time_features(B)
    assert set(tf) == set(B.F)
    for rid, X in tf.items():
        t = B.W[rid]["meta"]["t_end"].to_numpy()
        assert X.shape == (len(t), 2)
        np.testing.assert_allclose(X[:, 0], t / 500.0)
        np.testing.assert_allclose(X[:, 1], (t / 500.0) ** 2)
    # the control is label-free: identical whatever the protocol shift
    tf2 = time_features(B)
    assert all(np.array_equal(tf[k], tf2[k]) for k in tf)
