# Phase 1: brain switch (rest vs active), archived findings (2026-10-01/02)

Archived before the pivot to action discrimination (neutral / push / walk). Every number is from `eegbci/` on
`all_data` (28 recordings, 10 participants S01–S10). AUC with chance 0.5, causal 8 s EMA smoothing.

## Evaluation framework (kept for phase 2)
- Splits: leave-one-cycle-out, cross-session, leave-one-subject-out (LOSO), nested LOSO. Never random window splits.
- Drift-null: the protocol timeline is circularly shifted per recording. Time-only control model.
- Causal everywhere: SOS filters with state, causal Riemannian re-centering (tau 60 s), EMA. The realtime decoder
  equals offline for the Riemann branch (1e-14).

## Push vs walk
- Within recording: EEG 0.593 vs time-only 0.677 (null 0.529). **Not decodable with naive windows.** Push always
  precedes walk in both protocols. Phase 2 re-tests this with onset-locked analysis.

## Brain switch results (before the QA corrections; treat as upper bounds)
| eval | model | AUC |
|---|---|---|
| within | logbp causal60 | 0.640 (null 0.471 ± 0.027) |
| cross-session | fusion win4 | 0.734 |
| LOSO | fusion LR (realtime.FINAL_CFG) | 0.671 (null 0.489 ± 0.040, n=20) |
| LOSO | ShallowConvNet / EEGNet | 0.666 / 0.631 |
| LOSO | ensemble 50/50 | 0.707 (10/10 subj > 0.5); nulls 0.514, 0.538 |
| nested LOSO | selection inside | 0.670 |
| LOSO + 3–5 min calibration | | 0.709 vs 0.662 (but +0.039 of the gain also appears under the null) |
| replay S05 (held out) | ensemble decoder | AUC 0.89/0.83/0.70/0.41, ~17 ms/decision |

## Corrections found by the QA review (results/code_review.md, tests in tests: 65 pass, 9 xfail)
1. CRITICAL: the scenario-1 eyes-closed baseline biases causal re-centering, so the eyes-open baseline looks
   "rest-like". Scenario 1 LOSO 0.80 vs scenario 2 0.57; ensemble 0.781 vs 0.651. Excluding the eyes-open
   baseline: 0.671 → 0.638. Restarting the reference: 0.606.
2. HIGH: the eigenvalue floor of 1e-12 is absolute while eigenvalues are in V² (47–80 % clipped). With a relative
   floor, LOSO goes 0.671 → 0.641.
3. HIGH: the null shifted the eyes-closed exclusion too, so the nulls are about 0.03 too low.
4. HIGH: the online deep branch uses a different window grid from offline, so parity only holds for the Riemann
   branch.
5. HIGH: bad channels are detected on the whole recording (looks ahead) and not interpolated online.
6. MEDIUM: the onset metric counts "already on" as detected; the decoder crashes on chunks ≥ 1200 samples; 63
   configurations were screened before the nested step (optimism ~0.01–0.03).

## Validity (results/confounds_*; confounds.py, confounds_decode.py)
- The posterior beta increase at rest→task is NEURAL: +0.33 dB, 26/28 recordings, p<1e-4, present in both
  scenarios. 89 % survives regression of eye/muscle proxies, and it is +0.44 dB on blink-free frames.
- The decoder is ENTANGLED with eye behaviour. Blink rate is 9.9 → 7.8/min (scenario 1 mainly); a model on eye
  proxies alone reaches 0.640; EEG with the proxies regressed out reaches 0.593.
  Ablations: no Fp/F7/F8 0.674, no frontal 0.666, occipital-only 0.644, centro-parietal 0.638.
- Scenario-1 alpha rise co-varies with VEOG; some task blocks have alpha at the eyes-closed level, suggesting
  intermittent eye closure.
- Interpretation: a "visual-disengagement / internal-attention" detector, not a pure mental command.

## Artifacts
- Code: eegbci/ (README). Final switch model: models/brain_switch.joblib.
- Reports: analysis/audit_report.html (audit), results/brain_switch_report.html (draft, not published,
  pre-QA numbers).
