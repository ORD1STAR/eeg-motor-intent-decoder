# Evaluation and results

All numbers come from files in `results/` (original run outputs). Chance AUC is 0.5, chance 3-class balanced
accuracy is 0.333. "Pre-QA" means before the corrections of the independent code review (section 3).
The archived write-up of the first analysis phase is `results/FINDINGS_phase1_switch.md`.

## 1. Three-class intent (idle / push / walk)

Within a recording, leave-one-cycle-out, filter-bank tangent space (`results/round1.txt`):

| Task | balanced accuracy | drift-null | time-only model |
|---|---|---|---|
| 3 classes (`cmd3`) | 0.385 | 0.324 +/- 0.007 | 0.352 |
| rest vs command (`task_rest`) | 0.551 | 0.503 +/- 0.008 | 0.357 |
| push vs walk | 0.593 | 0.529 +/- 0.032 | **0.677** |

Idle/Push/Walk is only marginally above chance, and push vs walk is **not decodable**: a model that only sees the
timestamp beats every EEG model (push always precedes walk in both protocols).

The 2025 notebooks reported 35-50 % three-class accuracy with random window splits and test-set model selection.
The audit (`legacy_audit.html`) measured the same kind of models at 98 % median balanced accuracy with a random 5-fold
split of windows versus 36 % when train and test come from different blocks, and 35 % leave-one-subject-out. That gap
is leakage, not signal.

## 2. Push vs walk, controlled (`results/pushwalk_*.log`)

* Decoding (strict, leave-one-subject-out): EEG AUC 0.514, eyes-proxy 0.504, **time-only 0.811**.
  With raw features EEG reaches 0.573 (subject-level 0.568, 8/10 above 0.5), still below time-only.
* Scenario 2 crossed-order blocks (5 participants, 16 recordings): the order-invariant push-vs-walk effect has a
  maximum |t| of 2.35 and a minimum family-wise corrected p of 0.603. The only strong feature is an *order* effect
  (parietal beta2, max |t| 6.56, p_FWER 0.062), i.e. time, not intent.
* The minimum detectable effect (80 % power) is about 0.2-0.3 dB at subject level, so small effects remain possible.

Verdict: no push vs walk signature can be claimed from these recordings.

## 3. The brain switch (rest vs active command)

Pre-QA results (`results/ensemble.txt`, `nested.txt`, `confirm.txt`), AUC of the 8 s causally smoothed probability:

| Evaluation | Model | AUC | Control |
|---|---|---|---|
| within recording | log band power, causal re-centering | 0.640 | null 0.471 +/- 0.027, p = 0.032 |
| new session, same person | fusion, 4 s windows | 0.693 (xsess) | null 0.498 +/- 0.034 |
| new person (LOSO) | Riemann fusion + LR | 0.671 (subject-level 0.688, 9/10 > 0.5) | null 0.489 +/- 0.040 |
| new person | ShallowConvNet / EEGNet | 0.666 / 0.631 | |
| new person, nested | selection inside the loop | 0.670 (subject-level 0.685, 9/10 > 0.5) | |
| new person | ensemble (50/50, fixed a priori) | **0.707** (subject-level 0.708, 10/10 > 0.5) | nulls 0.514 and 0.538 |
| new person + 3-5 min warm-up | fusion re-fit | 0.709 vs 0.662 | the gain also appears under the null (null 0.536 vs 0.497) |
| nested, 13 classical candidates | `classical2.py` | 0.656 | below the 0.671 baseline; best plain candidate 0.675, i.e. no real gain |
| bagged ShallowConvNet (5 members) | `deep2.py` | 0.677 (ensemble 0.707) | no gain over the single net; null runs not completed |

Figure: `docs/figures/auc_summary.png`.

### Corrections found by the QA review (`results/code_review.md`)

| Finding | Effect on Riemann LOSO AUC |
|---|---|
| **Critical:** in scenario 1 the 120 s eyes-closed baseline biases the causal reference; the eyes-open baseline then looks "rest-like" and the drift-null cannot reproduce it (the transition is locked to protocol time) | scenario 1 scores 0.800, scenario 2 0.574; scoring without the eyes-open baseline 0.638; restarting the reference 0.606 |
| **High:** absolute eigenvalue floor of 1e-12 on covariances stored in V^2 (47-80 % of eigenvalues clipped) | 0.671 -> 0.641 with a scale-invariant floor |
| **High:** the null shifted the eyes-closed exclusion too (nulls about 0.03 too low); the online deep branch uses a different window grid from offline; bad channels are detected with look-ahead and not interpolated online | parity claim holds for the Riemann branch only |
| **Medium:** the onset metric counts "already on" as detected; the decoder crashes on chunks of 1200 samples or more; 63 configurations were screened before the nested step | optimism 0.01-0.03 |

Net effect: the pre-QA numbers are upper bounds. Realistic estimates are about 0.60-0.64 for the Riemann model, with a
genuine effect remaining in scenario 1 after removing the eyes-open baseline (0.70-0.72). The review found **no
train/test label leakage** in the LOSO / cross-session / cycle splits.
The corrected pipeline was not re-run end to end; the corrections are quantified, not applied. Figure:
`docs/figures/qa_corrections.png`.

## 4. Is the switch neural? (`results/confounds_*`, `results/maps_*`)

* The posterior beta/alpha increase at rest -> task is **neural**: +0.33 dB, 26 of 28 recordings, p < 1e-4, in both
  scenarios; 89 % survives regression of eye and muscle proxies and it is +0.44 dB on blink-free frames.
  Switching back decreases it. The two jumps have opposite signs, which slow drift cannot produce.
* But the **decoder is entangled with eye behaviour**: blink rate drops from 9.9 to 7.8 per minute at task onset
  (mostly scenario 1); a model on eye proxies alone reaches AUC 0.640; the EEG with the proxies regressed out
  reaches 0.593 (paired p = 0.001 against the base). Removing frontal channels barely changes the score
  (no Fp/F7/F8: 0.674; occipital only: 0.644; centro-parietal only: 0.638).
* Across 28 recordings the neural effect does not scale with the eye or neck-EMG changes
  (`results/confounds_coupling.png`: rho = -0.08 blink rate, +0.08 VEOG, +0.22 neck EMG; temporal EMG +0.44, p = 0.02).
* Interpretation: a "visual disengagement / internal attention" state change, not the textbook motor-cortex ERD, and
  not a pure mental command.

## 5. Real-time replay (`results/replay_S05.txt`)

Participant S05 held out of training, 100 ms chunks, ensemble decoder, CPU:

| Recording | AUC | state accuracy | false activations / min rest | onsets detected | ms / window |
|---|---|---|---|---|---|
| S05-3 | 0.888 | 0.82 | 1.00 | 4/4 | 18.6 |
| S05-4 | 0.828 | 0.77 | 1.50 | 4/4 | 17.7 |
| S05-5 | 0.695 | 0.67 | 0.00 | 2/2 | 17.3 |
| S05-6 | 0.411 | 0.53 | 1.50 | 2/2 | 16.8 |

The Riemann branch matched the offline features to about 1e-14. Per the QA review this holds only for recordings
without bad channels (23/28), and the deep branch is computed differently online. One of four recordings is below
chance, in line with the large between-recording spread.

## 6. Limitations

* 10 participants, 28 recordings: subject-level conclusions are fragile.
* The rest-vs-command decoder partly reads eye/attention behaviour; it is not a clean motor-intent signal.
* No EOG/EMG channels; artefact control relies on proxies derived from the EEG.
* The corrections of section 3 are quantified but the full pipeline has not been re-run with them.
* Push vs walk and three-class intent are not decodable beyond what elapsed time explains.

## 7. Next steps

* Apply the review's fixes (reset the reference at the eyes-open onset, scale-invariant eigenvalue floor, causal
  bad-channel handling online) and re-run LOSO and the nulls.
* Collect a protocol with randomised block order and no eyes-closed baseline, with recorded EOG.
* Per-user warm-up with proper nulls for the calibration gain.
* Complete the drift-null runs of the bagged deep branch.
