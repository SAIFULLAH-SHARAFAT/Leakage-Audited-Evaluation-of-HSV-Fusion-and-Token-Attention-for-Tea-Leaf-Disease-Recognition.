# Reproducibility notes

1. We lock the dataset by manifest SHA-256 and, after download, by the resolved Hugging Face commit SHA.
2. Every run command is generated from `configs/experiments.yaml`.
3. We use seeds 42, 1337, and 2026.
4. Each run stores its own config, exact predictions, test IDs, metrics, and source hashes.
5. The paper's runs live under `results/v2/` (campaign `tea_leaf_v2`). The earlier
   `results/fresh/` runs belong to the superseded v1 protocol and are not evidence.
6. `scripts/fd_scripts/` is design work and is **not** part of the paper's evidence.
7. If a software or library update changes the numbers, we report the captured
   environment rather than silently changing the recipe.

## Where each seed was trained

All 24 runs used the same code and the same configuration apart from the seed. The
three seeds were trained on different machines:

| Seed | torch | timm | GPU |
|---|---|---|---|
| 42 | 2.14.0+cu130 | 1.0.29 | NVIDIA RTX A4000 |
| 1337 | 2.5.1+cu121 | 1.0.29 | NVIDIA RTX A6000 |
| 2026 | 2.5.1+cu121 | 1.0.29 | RTX 6000 Ada (R0–R3, C1, C2); RTX A5000 (TLA); RTX 3090 (C4) |

Deterministic algorithms make each run reproducible on its own hardware and software;
they do not make runs bit-identical across GPUs or torch versions. Differences of this
kind act as seed-level noise, and within a seed every arm shares the same data order,
initialisation and augmentation stream.

**Source hashes differ across machines only in line endings.** Seed 42 recorded the
SHA-256 of CRLF files (Windows); seeds 1337 and 2026 recorded LF files (Linux). Every
recorded hash traces to the same commits — `cac6fc3` (swin trainer), `7a43c0a` (token
trainer), `4d16751` (training contracts), `546d17d` (experiment config) — in one form
or the other. Normalise line endings before comparing hashes across machines.

**Resumed runs:** R0, R1 and R2 at seed 1337, and C2 at seed 2026, were resumed from a
checkpoint. The v2 resume path restores Python, NumPy, torch and CUDA RNG state and the
train-loader generator, and logs each resume in `config/resume_history.jsonl`.

## Robustness evaluation

Robustness (`scripts/08_eval_robustness.py`) was run on different devices per seed: seed 42
on CPU (FP32), seeds 1337 and 2026 on GPU (RTX A4000, FP16 autocast). The per-seed outputs
are merged by `scripts/21_merge_robustness.py`, which records the device for each run.

Every run first recomputes its unperturbed test Macro-F1 from `ema_model_weights_only.pth`
and compares it with `metrics/test_results.json`. 23 of 24 runs match exactly. `C1_s2026`
does not: `Healthy leaf/healthy_00791.jpg` has exactly equal stored probabilities for
Healthy leaf and Helopeltis (top-2 margin 0.0). The training machine resolved the tie to the
true class, and the evaluation hardware resolves it the other way, which lowers Macro-F1
from 0.951524 to 0.950253. All other stored and recomputed probabilities agree to a median of
1e-5. The contract therefore accepts at most two flips with stored margin ≤ 1e-3; a
deliberately mismatched checkpoint (`C1_s1337` weights against `C1_s2026` outputs) flips 19
confident images and is rejected. For `C1_s2026`, retention uses the recomputed clean score.
