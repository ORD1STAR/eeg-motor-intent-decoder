# Results

Original outputs of the runs reported in `docs/evaluation.md`, archived verbatim except for anonymisation (participant
names replaced by `S01`-`S10`, local paths removed). Large derived arrays (`*.npz`, `*.joblib`, model weights) are not
committed; every step rebuilds them. Paths written inside the older logs refer to the original layout: `eegbci/results`
is `results/`, `eegbci/tests` is `tests/`, `all_data` is `$EEGBCI_DATA_DIR`.

| Files | Produced by | Content |
|---|---|---|
| `round1.txt`, `round3.txt`, `round4.txt`, `confirm.txt`, `explore_log.jsonl` | `explore.py` | feature / alignment / model screening and drift-null confirmation |
| `nested.txt`, `nested_loso.json` | `benchmark.py nested` | nested leave-one-subject-out (the number to quote) |
| `deep_*`, `ensemble*`, `deep2_*` | `deep.py`, `ensemble.py`, `deep2.py` | EEGNet, ShallowConvNet, ensemble, bagged nets, drift-nulls |
| `classical2_*` | `classical2.py` | 13 classical candidates, nested selection |
| `calibration.*` | `explore.py calibration` | per-user warm-up simulation |
| `confounds_*` | `confounds.py`, `confounds_decode.py` | blinks, EMG proxies, coupling, decoding ablations |
| `maps_*` | `maps.py` | onset-locked channel x frequency x time maps and cluster statistics |
| `pushwalk_*` | `pushwalk.py` | push vs walk tests |
| `replay_S05.*` | `realtime.py replay S05` | online replay of a held-out participant |
| `deep2_plan.txt` | written before the deep2 runs | analysis plan |
| `code_review.md`, `FINDINGS_phase1_switch.md` | QA review, archived summary | corrections and phase-1 summary |

Figures: `fig_topomaps.png`, `maps_topo_*.png` (the `*_raw` variants are not committed), `confounds_*.png`.
