# Reviewer-to-experiment map

Each reviewer's concern from the first round, the experiment or artifact
we built to answer it, and what it showed. All results are three-seed (42, 1337, 2026).

| Reviewer concern | How we tested it | What we found |
|---|---|---|
| Is the TLA gain just a different recipe or head? | `TLA` vs `C1` (same recipe, same head, no module) | No. TLA − C1 = +0.08, −0.15, −0.81 Macro-F1 points; no interval excludes zero. |
| Is the TLA gain just extra parameters? | `TLA` vs `C2` (exactly 3,545,089 added parameters in both) | TLA is below C2 on all three seeds (−0.13, −0.22, −0.15). |
| Would a simple lightweight attention do? | `C4` (ECA, 1,542 parameters) vs `C1` and `TLA` | C4 is below C1 on all three seeds; no lightweight gain either. |
| Is the HSV gain confounded by head or augmentation? | `R0`–`R3` share one head and hue jitter 0.0 | With the confounds removed there is no HSV gain (mean R1 − R0 = −0.01, R2 − R0 = −0.06, R3 − R0 = −0.40). |
| The bootstrap procedure is unclear | `07_paired_bootstrap.py`: 10,000 paired image-level resamples, fixed 7-label Macro-F1, percentile interval | Fully specified in the Methods; all 24 intervals in `tables/bootstrap_per_seed.tex`. |
| Hyperparameters are scattered | `10_make_training_settings_table.py`, generated from all 24 run configs | `tables/training_settings.tex`. |
| Robustness vs field generalization | `08_eval_robustness.py`: 18 controlled corruptions, all 24 runs | No add-on improves robustness over its matched baseline; the recipe changes mean perturbed Macro-F1 by ~6 points. Described as benchmark robustness, not field validation. |
| Might the HSV branch start too small to matter? | `init_scale_diagnostics.json` + `13_summarize_init_scale.py` | Measured before the first optimizer step; see `tables/init_scale_summary.csv`. |
| Reproducibility and data availability | Frozen HF release, manifest SHA-256, per-run provenance, export bundle | Dataset and code public; every table regenerates from stored run outputs. |

## How we read the results

The primary architectural comparison is **C1 → TLA**, interpreted together with C2 and C4.
The complementary chromatic comparison is **R0 → R1/R2/R3**. We never rank a token-family
model against an HSV-family model, because the two families use different training
recipes; R0 vs C1 is reported only as a recipe effect.

Which framing the paper takes was decided by the rule in `FRAMING_DECISION_RULE.md`,
committed before seeds 1337 and 2026 were scored. It selected Framing B.
