# eegbci code review (QA): correctness, leakage, causality

Scope: config, data, preprocessing, windows, features, evaluation, explore, benchmark, deep, ensemble, realtime.
Method: line-by-line reading, plus quantification on the cache (read-only, single-threaded; scripts in the session
scratchpad) and a pytest suite (`tests`, synthetic data plus read-only access to the real files and cache).
I edited no existing module.

**Reproduction check:** I rebuilt the Riemann fusion LOSO (4 s windows, C=0.01, EMA 8 s) from the cache and got
**0.6711**. That equals `ensemble.txt` (riemann 0.671), so the numbers quoted below come from the same pipeline.

## Summary: what holds and what does not

| Claim | Verdict |
|---|---|
| No train/test leakage (LOSO / xsess / cycle splits, StandardScaler, early stopping on a non-test subject, nested selection of the 6 candidates) | **Holds.** No label leakage was found. |
| Causal filters, causal re-centering, EMA, adaptive threshold | **Holds**, with three exceptions: bad-channel detection, `detrend('causal')`, and the first window being whitened by itself (harmless). |
| Online = offline to 1e-14 | **Riemann branch only, and only for recordings without bad channels (23/28).** The deep branch of the deployed ensemble is computed differently online. |
| Drift-null is a correct control | **Partly.** (i) The eyes-closed exclusion moves with the shift. (ii) The EC→EO re-centering transient is locked to the protocol, so the null cannot control it. See C1. |
| LOSO 0.707 / 0.671 / nested 0.670 | These numbers are reproducible, but **about 0.03–0.07 of them come from a protocol confound in scenario 1** (C1). They also depend on a numerical artefact (H1). |

---

## CRITICAL

### C1. The EC→EO re-centering transient labels the eyes-open baseline as "rest" (scenario 1). The drift-null cannot catch it.
*Files:* `features.py:59-79` (CausalRecenter, tau 60 s), `config.py:21` (scenario 1 = 120 s eyes closed, then 120 s
eyes-open baseline labelled NEUTRAL), `windows.py:95-100`.

**Mechanism.** In scenario 1 the causal reference spends 120 s learning the eyes-closed state, which has very high
alpha. When the eyes open at 120 s, every eyes-open baseline window looks strongly alpha/beta-depressed relative to
that reference, and the effect decays with tau = 60 s. The model has learned "active ⇒ higher PO alpha/beta", so it
reads these windows as maximal rest. They make up about 54 % of the scenario-1 rest windows. The transition happens
at a fixed protocol time, so a circular shift breaks its link to the labels. The null therefore does not reproduce
the effect, and beating the null does not rule it out.

**Evidence** (Riemann fusion LOSO, current code). Mean P(active) through the eyes-open baseline by time bin:

| Time bin | 125–145 s | 145–165 | 165–185 | 185–205 | 205–225 | 225–240 | Other rest | Active |
|---|---|---|---|---|---|---|---|---|
| Mean P(active) | 0.151 | 0.148 | 0.221 | 0.258 | 0.291 | 0.373 | 0.365 | 0.477 |
| PO alpha log-power | −1.09 | −0.89 | −0.74 | −0.68 | −0.65 | −0.56 | −0.45 | −0.34 |

During the eyes-open baseline, P(active) rises back to the "other rest" level with the 60 s re-centering time
constant.

| Riemann LOSO AUC | All 28 | Scen. 1 (12) | Scen. 2 (16) |
|---|---|---|---|
| Current | **0.671** | 0.800 | 0.574 |
| Scen. 1 scored without eyes-open baseline windows | 0.638 | 0.724 | — |
| Reference restarted at 120 s (eyes closed never enters it) | 0.606 | 0.654 | 0.570 |
| …and scored without eyes-open baseline windows | — | 0.700 | — |

Every published model shows the same split between the two protocols: ensemble 0.781 vs 0.651, nested 0.794 vs
0.577, ShallowNet 0.703 vs 0.637.

**Impact estimate:**
- Riemann / nested: −0.03 to −0.07 AUC.
- Ensemble: probably −0.02 to −0.04 (the deep branch is less affected: 0.694 in scenario 1).
- A genuine effect remains in scenario 1 after removing the eyes-open baseline (0.70–0.72).
- Scenario 2, which has no eyes-closed segment, sits at about 0.57 for Riemann and 0.65 for ShallowNet.

**Proposed fix (evaluation):**
- Report AUC with the eyes-open baseline excluded from scoring, or report it per scenario.
- Start the causal reference at the eyes-open onset. Online this is a reset when the user opens their eyes; offline:

```diff
 # features.recording_features / realtime: never let excluded (label None) segments feed the reference
-        A = recenter(Cb, align, tau_s=tau_s)
+        A = np.empty_like(Cb, dtype=np.float64)
+        A[~excl] = recenter(Cb[~excl], align, tau_s=tau_s)   # excl = windows overlapping the EC baseline
+        A[excl] = recenter(Cb[excl], align, tau_s=tau_s)
```

At minimum, the README should state the scenario split and the number without the eyes-open baseline.

## HIGH

### H1. The absolute eigenvalue floor of 1e-12 clips covariances that are stored in V². Features become unit-dependent.
*File:* `features.py:15-17` (`_eig_fn`, used by `logm` and `invsqrtm`). Band covariances are in V²: **47–80 % of
their eigenvalues lie below 1e-12** (smallest about 1e-14). As a result:
- `logm` flattens most of the spectrum;
- the "log-Euclidean mean" and the whitening are wrong in the weak directions;
- multiplying the data by 1e12 changes the features by up to 2.08 (median |Δfeature| 0.48).

Parity is unaffected because online and offline share the bug. However, a decoder fed µV (for example the LSL path
without the `uV` flag) would compute different features.

**Impact.** With a scale-invariant floor the Riemann LOSO drops from **0.671 to 0.641** (scenario 1 0.800 → 0.752,
scenario 2 0.574 → 0.557). Adaptive balanced accuracy goes from 0.602 to 0.606. The headline therefore partly relies
on the artefact: the clipping acts as an implicit regulariser. Either keep it as a documented, deliberate shrinkage
(a floor in µV²) or fix it and re-run.

```diff
 def _eig_fn(C, fn):
     w, V = np.linalg.eigh(C)
-    return (V * fn(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
+    return (V * fn(np.maximum(w, w[..., -1:] * 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
```

If the regularisation is wanted, make it explicit instead, for example `Covariances('oas')` on µV data or
`C + eps*trace(C)/n*I`.

### H2. Under the drift-null, the eyes-closed exclusion moves with the shifted protocol.
*File:* `windows.py:93-99`. The −2 label is applied in **shifted protocol time**. Under a shift, 72–78 % of the real
eyes-closed windows receive task/rest labels, and a 120 s block of real task data is excluded instead. The null
therefore analyses different data from the observed run. Artifact rejection, by contrast, is correctly done in real
time.

**Measured.** Within-recording null for the confirm configuration (logbp, 2 s windows), scenario-1 recordings,
20 shifts:

| Variant | Null mean ± sd |
|---|---|
| Current | 0.503 ± 0.024 |
| Fixed (shift within the analysed span, eyes-closed excluded in real time) | 0.532 ± 0.026 |

The observed value is 0.713, so it is still far above the null. But the corrected null is **0.03 higher**, so the
published nulls of 0.47–0.50 are slightly too low. No claim flips. The fix is validated in
`tests/helpers.proposed_label_windows`:

```diff
-    ts = (meta["t_start"].to_numpy() - shift) % duration
+    lo = min(b[0] for b in blocks if b[2] is not None)          # analysed span starts after EC
+    t_s, t_e = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
+    ts = lo + (t_s - lo - shift) % (duration - lo)
     te = ts + (meta["t_end"].to_numpy() - meta["t_start"].to_numpy())
     for i, (t0, t1, l, name, c) in enumerate(blocks):
-        inside = (ts >= t0 + guard) & (te <= t1 - guard) & (te <= duration)
         if l is None:
-            lab[inside] = -2      # excluded segment (eyes-closed baseline)
+            lab[(t_e > t0) & (t_s < t1)] = -2                    # excluded in REAL time
             continue
+        inside = (ts >= t0 + guard) & (te <= t1 - guard) & (te <= duration) & (t_s >= lo)
```

`random_shifts` should also draw within `[min_s, span - min_s]`.

A related caveat (low): shifted task/rest labels are not independent of the true ones. The mean correlation (φ)
is −0.19 for scenario 1 and −0.02 for scenario 2, and 8–17 % of shifts reach φ > 0.3 (shifts near 180 s in
scenario 2). This makes the null slightly conservative or noisy. It is not anti-conservative.

### H3. The online deep branch is not the offline deep branch, so "online = offline" does not hold for the ensemble.
*Files:* `realtime.py:180-192` vs `ensemble.py:48,64-69`.
- **Online:** the deep input is the last 2 s of each 4 s window, and the causal Euclidean alignment runs on the 4 s
  grid starting at t = 4 s.
- **Offline (`ensemble.py`):** the nearest 2 s window, which ends 2 decimated samples (16 ms) earlier
  (`test_ensemble_grid_matching_is_causal` confirms j = k+4), with the alignment run on the 2 s grid. That grid
  includes 4 extra early windows, so the warm-up mean differs and only decays as exp(−t/60 s).
- `replay` only checks `p_riemann`, so the published 1e-14 is Riemann-only.

What is verified:
- Online deep equals the "last 2 s / 4 s grid" construction, and parity holds when the grids match (tests pass).
- Online deep differs from the ensemble construction (xfail test).

I could not finish the real-data measurement of |Δp| on S05-3 (the background job was killed by the session
interruption). The difference is expected to be small in AUC but non-zero, and largest in the first ~40 windows.

**Fix:** run the deep branch on its own 2 s grid inside the decoder and pair each 4 s window with the latest 2 s
output. This reproduces `ensemble.py` exactly. A reference implementation is in
`tests/helpers.make_proposed_decoder_class`.

### H4. Bad-channel handling is non-causal offline and missing online.
*Files:* `windows.py:41-43`, `preprocessing.py:41-49`.
- `find_bad_channels` uses the entire recording, so it looks ahead (xfail test). The test recording's interpolation
  in LOSO therefore uses future data.
- The decoder never interpolates, so parity fails for **S06-1 (CP2), S08-1 (Fp2), S08-2 (FT9, T8, FT10), S09-2 (Fp1)
  and S10-0 (O2, F4)**. O2 and CP2 are part of the PO tangent-space channel set.

**Impact on offline AUC:** small (5 recordings, interpolation is spatial only).

**Fix:** detect bad channels on a calibration prefix (for example the first 60 s, or the EC/EO baseline). Freeze the
list and pass it to the decoder (`bundle["bads"]`) along with the same `interpolate_bads` matrix.

## MEDIUM

### M1. `online_metrics` counts an onset as detected when the switch was already on.
*File:* `realtime.py:227-233`. The test `m = (t > b0) & (t <= b1) & act` gives latency ≈ 0 if the decoder was
already active at the onset. "S05-4 latency 0.2 s" is physically impossible with a 4 s window and 8 s smoothing, so
the claim "every onset detected" is inflated.

```diff
-            m = (t > b[0]) & (t <= b[1]) & act
-            if m.any():
-                hits += 1; lats.append(t[m][0] - b[0])
+            r = t[rises][(t[rises] > b[0]) & (t[rises] <= b[1])]
+            if len(r):
+                hits += 1; lats.append(r[0] - b[0])
```

Separately, false activations are only counted when the rising edge falls in a labelled rest window, so rising
edges inside guard windows are missed.

### M2. The ring buffer overruns for chunks longer than about 2 s.
*File:* `realtime.py:138-148`. Buffers keep `win + 4*step` = 748 decimated samples and are trimmed **before** the
windows are emitted. A chunk of 1200 or more raw samples gives the first pending window a negative slice start,
which raises `ValueError: 0 samples ... OAS`.
- Measured: 996–1000 raw samples is fine, 1200 crashes.
- LSL `max_samples=250` and the 0.1 s replay chunks are safe.
- An LSL backlog or a different chunk size would crash the decoder.

**Fix:** trim after the `while` loop (reference: `make_proposed_decoder_class`), or cap the chunk size inside `push`.

### M3. The calibration gain is mostly explained by drift.
Source: `results/calibration.txt`.

| | Observed | Null |
|---|---|---|
| Without calibration (loso) | 0.662 | 0.497 |
| With calibration (loso+cal) | 0.709 | 0.536 |
| Gain | +0.047 | +0.039 |

The task-specific gain is therefore about **+0.01**. With 10 nulls, the smallest p-value achievable is 0.091. The
README line "0.709 vs 0.662" overstates the benefit.

### M4. The ensemble's offline decision rule is not the deployed rule.
`ensemble.adaptive_decision` has no margin and a minimum on-time of 1 window. The decoder uses a margin of 0.02 and
a minimum on-time of 2 windows. The threshold schedules are otherwise identical (tested exactly with margin=0,
min_on_s=0.5). On the Riemann LOSO probabilities the two rules agree on 92 % of windows (balanced accuracy 0.602 vs
0.597). The "bacc 0.64 adaptive" figure belongs to the offline rule; report both.

### M5. Selection outside the nested loop.
Nested LOSO only nests the final choice among 6 candidates. Before that, 63 LOSO configurations were screened on all
10 subjects (`explore_log.jsonl`: max 0.671, median 0.569). The PO channel set and the bands were chosen from the
ERSP of all 28 recordings, and ShallowNet was picked over EEGNet using LOSO. The likely optimism is about 0.01–0.03,
within one standard error of 0.037. Say so in the README.

### M6. `eval_within` pools scores from different fold models into one AUC.
*File:* `evaluation.py:81-90`. In scenario 2, cycle 2 has no rest block, so its active windows (scored by a model
trained on 0–360 s) are ranked against rest windows scored by other models. The null has the same structure, but
the "within 0.640" number mixes calibrations. Prefer a per-fold AUC averaged over folds that contain both classes.

## LOW

- **`features.detrend('causal')` (`features.py:101`)** seeds its mean with the first 20 s, which are future windows.
  It is only used in round 1. Fix: `proposed_detrend_causal`.
- **`summarize` (`evaluation.py:144`)** omits `median` when empty, so `report` raises `KeyError`. `null_distribution`
  can return NaN means, and `null >= obs` then ignores them.
- **`CausalRecenter` and `causal_euclidean_align`** whiten the first window by itself (features = 0). This is causal
  and affects only one window per recording. The docstring "strictly before" is otherwise exact, and parity with
  `recenter('causal')` holds by construction (tested).
- **The hop is 62 decimated samples (0.496 s), but the EMA, re-centering and threshold constants assume 0.5 s.**
  Effective time constants are 0.8 % longer and the warm-up is 19.84 s. Labels use the true `t_end`. Negligible.
- **Decimation:** the 1–40 Hz deep input aliases 100 Hz mains to 25 Hz at about −32 dB. Online and offline are
  identical, so the impact is minor.
- **`data.py`:** S10-0 contains a `PAUSE` annotation at 148.56 s that is silently ignored. Every marker in the 28
  files sits exactly on its nominal time (deviation 0.000 s), so markers cannot reveal timing slips. Deduplication
  works (5d/6d are byte copies), and no near-duplicates were found among the 28 recordings (the closest pair has mean
  |Δlog-power| of 0.31). S07-0 is excluded (unrecognised marker set). The cache key is `rec_id` without a content
  hash; there is no collision today (tested).
- **`deep.train`:** class weights are balanced; if a validation AUC is NaN, `best_state` stays `None` and the call
  crashes. Early stopping uses a non-test subject, so there is no leakage.
- **Null thread safety:** OK. Every call builds fresh models and `Bundle` is read-only; a test confirms threads give
  identical results. p-value `(1+#≥)/(1+n)` is correct, with a floor of 0.032 at 30 shifts.
- **Neuro statistics:** the Wilcoxon test over 28 recordings treats the multiple sessions of a subject as
  independent (pseudo-replication) and is not corrected across 30 channels × 3 bands.

## Test results

`python -m pytest tests -q` → **65 passed, 9 xfailed in 76 s** (synthetic data; `test_data.py` reads the
headers and md5 of `all_data/` and one cached `meta`, read-only).

**Passing invariants:**
- CausalFilter: chunked == one-shot, causal, DC steady state.
- CausalRecenter: run == recenter('causal') == naive strictly-past reference.
- label_windows: guard, independent reference with and without shift (labels, blocks, cycles), wrap-around
  exclusion, full-period identity, 440 s clipping.
- Real S07-2 cache labels.
- ema_causal is causal.
- eval_within fold skipping; eval_groups smooths over all windows.
- LOSO groups are disjoint; null is thread-deterministic; p-value formula; time_features.
- Decoder vs offline Riemann parity with random chunks (≤ 996 samples) and odd chunk sizes: max |Δp| < 1e-10,
  identical `t`, smoothed p and artifact flags.
- Adaptive threshold parity (margin 0, min-on 1) with `ensemble.adaptive_decision`.
- Deep-branch parity on a matching grid; grid matching is causal (−16 ms).
- Dedup and scenario detection.
- Proposed patches validated.

**xfail (strict) = documented bugs:**
- scale-dependent eigenvalue floor (H1)
- eyes-closed exclusion under shift (H2)
- online deep ≠ ensemble offline (H3)
- bad-channel parity, bad-channel look-ahead (H4)
- onset detection (M1)
- long chunks (M2)
- `detrend('causal')`
- `report({})`

Once a bug is fixed, its test will XPASS and fail. Remove the marker at that point.
