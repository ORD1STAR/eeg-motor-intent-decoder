# Methodology

The goal of the project is to decode motor *intent* (idle, push, walk) from a 30-channel EEG headset, in a way that
survives an honest evaluation and can run online. The first phase (June 2025) produced scores that turned out to
come from leakage (see `legacy_audit.html`). The second phase (October 2026, package `eegbci`) rebuilt everything
around evaluation rigour.

## 1. Data and labels

* 10 participants, 28 usable recordings, 30 channels, 500 Hz (`data/README.md`).
* Two cue protocols (scenario 1: 480 s with an eyes-closed and an eyes-open baseline; scenario 2: 500 s with
  alternating push/walk blocks). Both are encoded in `eegbci.config.PROTOCOLS`.
* Three label schemes are evaluated: `cmd3` (idle / push / walk), `task_rest` (**brain switch**: rest vs any command)
  and `push_walk`.
* A 1 s guard is dropped after each block start and before each block end (reaction time). The scenario-1
  eyes-closed baseline is excluded from labels.

## 2. Causal signal processing

Everything that feeds a model is **causal**, so offline scores equal what an online decoder would produce.

| Step | Module | Detail |
|---|---|---|
| Loading | `data.py` | EDF via MNE, byte-identical duplicates removed by content hash, protocol detected from annotations |
| Filtering | `preprocessing.py` | Butterworth SOS with carried state (no zero-phase filtering); bad channels interpolated |
| Windows | `windows.py` | 2 s or 4 s windows every 0.5 s; peak-to-peak rejection at 150 uV; cached per recording |
| Features | `features.py` | filter-bank OAS covariances (4-8, 8-12, 12-16, 16-24, 24-30 Hz); **causal Riemannian re-centering** (exponentially weighted reference, tau = 60 s); tangent-space vectors and log band power |
| Smoothing | `realtime.py`, `ensemble.py` | causal EMA (8 s) on the probability; adaptive threshold = running mean of the smoothed probability |

Models: logistic regression on a fusion of log band power (30 channels, 5 bands) and tangent-space features
(12 parieto-occipital channels, 4 bands); EEGNet and ShallowConvNet on causally Euclidean-aligned 2 s epochs
(`deep.py`); an equal-weight ensemble of the Riemannian model and ShallowConvNet, fixed *a priori* (`ensemble.py`);
bagged ShallowConvNets with inner-subject early stopping (`deep2.py`).

## 3. Evaluation rules (why the numbers can be trusted, and where they cannot)

1. **No random window splits.** Neighbouring windows overlap and EEG drifts, so a random split measures *when* a window
   was recorded. Splits used (`evaluation.py`): leave-one-cycle-out within a recording, leave-one-recording-out within a
   person (cross-session), leave-one-subject-out (LOSO). A block is never split between train and test.
2. **Beat the drift-null, not just chance.** The identical pipeline is re-run with each recording's protocol timeline
   circularly shifted. That keeps drift and autocorrelation but breaks the link to the task. A result counts only if it
   beats that distribution (`evaluation.null_distribution`, nulls retrain the whole pipeline).
3. **Beat a time-only model.** A classifier that sees only the timestamp (`evaluation.time_features`) is a mandatory
   control. In both protocols push precedes walk, so any push-vs-walk score has to beat it.
4. **Nested selection.** The reported configuration is chosen by an inner LOSO on 9 people and tested on the 10th
   (`benchmark.py nested`, `classical2.py nested`), so selection optimism is included in the number.
5. **Pre-registered analysis plans.** `results/deep2_plan.txt` was written before any result of that experiment existed.
6. **Independent review.** `results/code_review.md` is a line-by-line QA of the pipeline (leakage, causality,
   numerical issues) and quantifies each finding. See `evaluation.md` for what it changed.

Metrics: AUC of the smoothed P(active) per recording (chance 0.5), reported as the mean over recordings and at subject
level; balanced accuracy with a fixed and with an adaptive threshold.

## 4. Validity analyses (is it neural?)

* `neuro.py`: transition-locked band power, baseline-corrected on the preceding seconds. A real state change gives
  opposite-signed responses at onset and offset, which slow drift cannot produce.
* `confounds.py`, `confounds_decode.py`: blink detection from Fp1/Fp2 (no EOG channel exists), VEOG/HEOG/EMG proxies,
  their coupling to the neural effect, and decoding ablations (no frontal channels, blink gating, proxy regression).
* `brainonly.py`, `maps.py`: multitaper channel x frequency x time maps per cue change, with channel exclusion, blink
  masking and label-free regression of the proxies; cluster permutation tests in MNE.
* `pushwalk.py`: is there a push != walk signature once time, drift and eyes are controlled? Matched onsets
  (scenario 1), crossed-order blocks (scenario 2), decoding with time-only and eyes-only controls.
* `decision.py`: causal on/off rules (EMA + adaptive threshold, fixed threshold, HMM filter, CUSUM) compared with
  hyper-parameters chosen on the other 9 subjects only.

## 5. Reproducing the experiments

See the README (section "Reproduce") for the exact commands. Every step logs to `results/`; the files committed there
are the original outputs of the runs reported in `evaluation.md`.
