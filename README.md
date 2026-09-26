# Leakage-Audited Evaluation of HSV Fusion and Token Attention for Tea Leaf Disease Recognition

Repository: <https://github.com/SAIFULLAH-SHARAFAT/Leakage-Audited-Evaluation-of-HSV-Fusion-and-Token-Attention-for-Tea-Leaf-Disease-Recognition.>

This repository holds our code, our dataset audit, and the full evidence trail for a study
on classifying tea leaf diseases and pests from photographs of single leaves. There are
seven classes — Brown Blight, Gray Blight, Green mirid bug, Healthy leaf, Helopeltis, Red
spider, and Tea algal leaf spot — and one backbone throughout, Swin-S at 224×224.

We ask two questions and keep them deliberately separate:

1. **Does token-level attention added at the last Swin stage actually help, or does it just
   add parameters?** We compare the module (`TLA`) against three controls: an identical
   model with no module at all (`C1`), a non-attention MLP matched to `TLA`'s parameter
   count *exactly* (`C2`), and a very cheap channel-attention comparator (`C4`).
2. **Does giving the network an explicit HSV colour signal help?** We compare three HSV
   branch variants (`R1`, `R2`, `R3`) against a matched RGB baseline (`R0`).

Eight models, three seeds (42, 1337, 2026), twenty-four training runs.

### What we found

Neither add-on gives a detectable improvement under this protocol. Under matched controls,
`TLA` does not detectably improve on `C1` on any seed (95.11 ± 0.71 vs 95.41 ± 0.26 test
Macro-F1) and scores below the parameter-matched MLP `C2` on all three seeds. No HSV variant
shows a detectable improvement over `R0` (94.28 ± 0.47). None of the 24 predeclared
paired-bootstrap intervals excludes zero — which means no detectable difference at this
test-set size, not proof of equivalence — and no add-on shows a detectable robustness
improvement under 18 synthetic corruptions. The largest effect in the study is the training recipe itself: the
same network (`R0` vs `C1`) differs by about 1.1 Macro-F1 points clean and 6 points under
perturbation. For the hardest pair, Brown Blight versus Tea algal leaf spot, lesion hue
differs by only about 3.5°, and under a linear probe the RGB backbone features separate the
pair far better than colour features do. We deploy `C1`, the model without an add-on, in our
server-assisted prototype. We committed a decision rule for how to frame the paper before scoring
the last two seeds; it selected the controlled-evaluation framing (see
`docs/FRAMING_DECISION_RULE.md`).

### How we tried to make the comparisons trustworthy

Most of the machinery here exists to make the comparisons trustworthy rather than to make
the numbers large:

- **The dataset is frozen and cryptographically locked.** Every script that touches data
  re-checks a SHA-256 of the split manifest and refuses to run if it has moved.
- **The comparisons are declared up front**, in `configs/experiments.yaml`, before any
  training run of this campaign started.
- **The controls are parameter-matched by construction, and the match is asserted at
  runtime.** `TLA` and `C2` both hold exactly 3,545,089 active parameters. If a flag is
  silently ignored and the architecture comes out different, the run aborts.
- **Tables and figures are generated, not typed.** Every table in the paper is produced
  from the run artifacts by a script, so it can't drift away from the JSON it came from.
- **Every run records its own provenance** — the exact argv, the SHA-256 of the training
  script and config that produced it, the dataset lock, the Python and platform strings.
- **We fold corrections in openly instead of overwriting.** This is campaign **v2**; the
  next section says exactly what changed and why, and the v1 partial results stay on disk as
  a labelled historical artifact rather than being deleted or quietly merged in.

---

## What changed in v2

A review of the v1 pipeline found problems in two places: in the training code, and in the
scripts that are supposed to catch problems before or after training. Because some of the
training fixes change the trained weights, we started a fresh campaign (`tea_leaf_v2`, new
output directory) rather than mixing old and new runs. The declared architectures and the
predeclared comparisons did not change.

**Training-protocol fixes (these change trained weights, so every v2 run uses them):**

| Area | What changed |
|---|---|
| Token-family features | The token modules now read the Stage-4 tokens *after* Swin's final LayerNorm, as the standard Swin classifier does. v1 read the pre-norm features, which left the final norm disconnected from the loss. |
| Head initialisation | The shared classifier head is initialised from its own seeded stream, so building an optional module first no longer changes the head's starting weights (`training_contracts.matched_initialization`). |
| Weight decay | Norm parameters, biases and gate parameters are exempt from weight decay (`training_contracts.optimizer_groups`), so decay no longer pulls a gate towards σ(0) = 0.5. |
| Gradient accumulation | A final, incomplete accumulation group is weighted by its true size (`training_contracts.accumulation_weight`). |
| AMP step accounting | The LR scheduler and EMA advance only on optimizer steps that AMP did not skip. |
| Resume | Checkpoints store and restore the Python, NumPy, torch and CUDA RNG states and the train-loader generator; each resume is logged. |

**Validation and evidence-generation fixes (these do not touch training):**

| Area | What we added or fixed |
|---|---|
| `02_preflight.py` | Gates on CUDA/dependency availability, compiles every script to catch syntax errors before launch, checks the matrix is exactly 8 experiments × 3 seeds = 24 runs, and requires `ImageHash` because the derivative audit needs it. |
| `03_capture_environment.py` | Also records the Git commit SHA, the working-tree dirty/clean status, and the SHA-256 of `experiments.yaml` and of the dataset manifest. |
| `04_validate_model_contracts.py` | Adds a forward-equivalence test (copy weights between `R0` and `C1` and confirm identical output, not just identical parameter count), a gradient-flow check on `C1`'s final Swin `LayerNorm`, and a sanity guard on `C4`. |
| `src/training_contracts.py` | Checkpoint-selection metadata separately reports the optimizer-step and skipped-update counts at the selected checkpoint and at the end of the run. |
| `src/train_token_models.py` | Bare `assert` statements (strippable under `python -O`) replaced with explicit exceptions. |
| `07_paired_bootstrap.py` | Drops an unsupported claim that unstratified resampling is conservative for minority classes. The procedure is unchanged. |
| `08_eval_robustness.py` | Reads the matrix through `campaign.py`, validates the frozen test index before scoring, uses strict checkpoint loading, and requires each run to reproduce its stored clean test score before any corruption is scored (see "Clean-score contract" below). Test files are sorted by name, so the loader matches the frozen index on Windows too. Results are saved after every run. |
| `11_export_reproducibility_bundle.py` | Includes `PACKAGE_MANIFEST.json` and the derivative-leakage audit output when they exist. |
| `12_pairwise_error_analysis.py` | Refuses to compare runs from different campaigns, requires unique and identical image-ID sets, re-checks label alignment, and states that its McNemar p-value is not corrected for multiplicity. |
| `13_summarize_init_scale.py` *(new)* | Aggregates each HSV run's `init_scale_diagnostics.json` across seeds. |
| `14_pipeline.py` *(new)* | Runs the campaign in a fixed, safe stage order (`prepare → smoke → seed42 → remaining → analyze → export`). |
| `15_package_manifest.py` *(new)* | Maintenance only: regenerates `PACKAGE_MANIFEST.json`, a content-hash inventory of the code. |
| `16_audit_derivative_perceptual_leakage.py` *(new)* | Re-checks every training derivative against validation and test at the frozen pHash + embedding operating point. Any linked pair is a stop condition. |
| `17`–`21` *(new)* | Post-hoc failure analysis and the manuscript helpers, described below. |

All output paths changed too. v1 wrote everything under `results/fresh/`; v2's
`configs/experiments.yaml` sets:

```yaml
campaign_id: tea_leaf_v2
results_dir: results/v2/runs
ckpt_dir: results/v2/ckpt
tables_dir: results/v2/tables
figures_dir: results/v2/figures
```

`results/fresh/` (one seed of `R0` complete, `C1` mid-training) is kept as a historical
record of the v1 pipeline. **It is not v2 evidence**: it predates the fixes above, its
`run_provenance.json` does not carry the `tea_leaf_v2` campaign ID, and we never cite or
average it with anything under `results/v2/`.

---

## What's in the box

```text
configs/
  experiments.yaml            the eight experiments, two recipes, campaign_id "tea_leaf_v2",
                               output paths, and the paired comparisons
  future_designs.yaml         a separate, unrun design campaign (see "Future designs")
data/
  DATASET_CARD.md             layout notes and the HF dataset card front-matter
  Tea_leaf_dataset/           the images themselves (not in git; downloaded in step 1)
  manifests/                  the audit trail: split manifest, fingerprint, pruning records
docs/
  DATASET_CARD.md             what the frozen build contains
  DATA_PROVENANCE.md          where the images came from, and what the audit does not claim
  FRAMING_DECISION_RULE.md    the rule we committed before scoring seeds 1337/2026, and its outcome
  REVIEWER_EXPERIMENT_MAP.md  which artifact answers which reviewer question, and the answer
  REVIEW_OF_PROPOSED_FIXES.md our dated review of the v1 fix list (the input to v2)
  reproducibility.md          machines, resumed runs, robustness devices, and the tie case
  FUTURE_*.md                 design notes for the shared-colour-branch follow-up
scripts/
  00_ .. 16_*.py              the pipeline, in the order we ran it (15 is maintenance-only)
  17_, 18_*.py                post-hoc failure analysis — exploratory, not predeclared
  19_, 20_, 21_*.py           manuscript tables, table embedding, robustness merge
  dataset_cleaning (1..5).py  archival record of how the frozen dataset was built
  fd_scripts/                 the future-design campaign, isolated so it can't run by accident
src/
  train_swin_three_models.py  the HSV family trainer (R0–R3) and the shared utility layer
  train_token_models.py       the token family trainer (C1, TLA, C2, C4)
  training_contracts.py       shared protocol pieces used by both trainers
  campaign.py                 shared experiment-matrix and path resolution
results/v2/                   everything the campaign writes (runs/, tables/, figures/, analysis/, audits/)
PACKAGE_MANIFEST.json         generated by 15_package_manifest.py
```

The manuscript sources (the revised paper, supplement and response letter) are not in this
repository while the paper is under review; they will be added after acceptance. Every
table in them is generated by the scripts here, and `20_embed_tables.py` writes those tables
into the manuscript file.

The two trainers are large, self-contained scripts. `train_token_models.py` imports its
seeding, EMA, checkpointing, and HSV conversion helpers from `train_swin_three_models.py`,
and both import shared logic from `training_contracts.py` and `campaign.py` — all four
files must stay side by side in `src/`.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. fetch the frozen dataset and record the Hugging Face commit it came from
python scripts/00_fetch_frozen_dataset.py --revision main --overwrite

# 2. check that what landed on disk is exactly the audited partition
python scripts/01_verify_frozen_dataset.py

# 3. re-check that no training derivative perceptually leaks into val/test
python scripts/16_audit_derivative_perceptual_leakage.py

# 4. check dependencies, the dataset lock, and that the run matrix is well formed
python scripts/02_preflight.py
python scripts/03_capture_environment.py
python scripts/04_validate_model_contracts.py

# 5. one run, to prove the loop works end to end
python scripts/05_run_experiments.py --only C1 --seeds 42
```

Steps 2–4 are cheap and start no training. Step 5 takes hours on one GPU. If any of steps
2–4 fail, or if step 3 reports a linked pair, stop there — they fail for a reason. Steps
1–5 can also be run as a unit with `14_pipeline.py --prepare` / `--smoke`.

`00_fetch_frozen_dataset.py` is the only supported way to obtain the data. It lays down the
`data/Tea_leaf_dataset/<split>/<class>/` image tree and `data/manifests/` exactly as the rest
of the pipeline expects them, and records the resolved Hugging Face commit. Loading the
dataset through the `datasets` library instead returns an in-memory object, not that
on-disk layout, so nothing downstream would run. If you already have the exact release in
`data/Tea_leaf_dataset/` and `data/manifests/`, skip step 1 and go straight to verification.

---

## The dataset

The images come from **TeaLeafBD** (Mendeley Data, DOI `10.17632/744vznw5k2.4`). We do not
use the raw release: we use a cleaned, leakage-audited partition that we published as a
frozen snapshot at
[`saifullah03/tea-leaf-disease-dataset`](https://huggingface.co/datasets/saifullah03/tea-leaf-disease-dataset).

| Split | Images | Notes |
|---|---:|---|
| train | 6,090 | 3,971 retained originals + 2,119 training-only augmented derivatives |
| validation | 816 | originals only |
| test | 808 | originals only |
| **total** | **7,714** | 5,595 originals |

Per class:

| Class | train | val | test | total |
|---|---:|---:|---:|---:|
| Brown Blight | 858 | 83 | 86 | 1,027 |
| Gray Blight | 896 | 163 | 160 | 1,219 |
| Green mirid bug | 897 | 185 | 179 | 1,261 |
| Healthy leaf | 893 | 147 | 141 | 1,181 |
| Helopeltis | 863 | 86 | 89 | 1,038 |
| Red spider | 835 | 76 | 75 | 986 |
| Tea algal leaf spot | 848 | 76 | 78 | 1,002 |

### What the audit checks

Leaf photographs are taken in bursts: several frames of the same physical leaf, seconds
apart, under the same light. A file-level random split scatters those frames across
train/val/test, and the model is then scored on near-identical views of leaves it trained
on. Byte hashing does not catch that, and neither does filename matching. So we ran four
checks at build time, and all four can be re-checked from this repository:

1. **Exact duplicates** — MD5 of every file; zero MD5 groups may span splits.
2. **Source families** — an `_aug…` suffix is stripped to recover the canonical source
   stem; no canonical family may appear in more than one split.
3. **Derivative confinement** — every augmented image must live in `train`.
4. **Perceptual near-duplicates** — two images are linked only when *both* a 64-bit pHash
   (Hamming distance ≤ 8) and a ResNet-50 embedding cosine similarity (≥ 0.92) agree.
   We found 249 such pairs; for every cross-partition pair we deleted the evaluation-side
   image, removing **79 images** (35 from validation, 44 from test). The training split was
   left untouched.

A fifth check runs at campaign time rather than only at build time:

5. **Derivatives against evaluation images** — `16_audit_derivative_perceptual_leakage.py`
   re-applies the exact operating point stored in `final_fingerprint.json`, this time between
   every training derivative and every validation/test image of the same class. Checks 1–4
   screened originals; an augmented derivative can resemble an evaluation image more closely
   than its source does, so this closes that gap. For the frozen build it screened 2,119
   derivatives against 1,624 evaluation images and found **no linked pair**. It never deletes
   or modifies an image; a linked pair would be a stop condition.

We want to be precise about what this establishes. The released partition contains no
byte-identical cross-split duplicates, no filename-derived family overlap, no derivatives
outside training, and no cross-partition pairs above the stated two-signal threshold. It
does not prove that no semantic near-duplicate of any kind exists anywhere.
`docs/DATA_PROVENANCE.md` states the same limit.

`data/manifests/` carries the whole record: `final_split_manifest.csv` (one row per
instance, with MD5, canonical id, and derived flag), `final_fingerprint.json` (counts and
every threshold used), `pruned_images.csv` (what was removed and why, with the measured
distances), `final_linked_pairs.csv`, `aug_params.csv` (augmentation parameters recovered by
measuring the derivatives against their sources), and `montages/` (the visual evidence we
used to choose the thresholds).

### The lock

The manifest is pinned by SHA-256:

```text
d407fdb133ebc95c3b0bc6f815280786bcdef37cd6c3c009192198fd21943d3d
```

`scripts/01_verify_frozen_dataset.py` hashes the manifest file's raw bytes (deliberately not
a pandas round-trip, which would make the hash depend on your pandas version), compares it
with the value recorded inside `final_fingerprint.json`, walks the image tree, and re-runs
the leakage audits. `--verify-md5` additionally re-hashes all 7,714 files.

```bash
python scripts/01_verify_frozen_dataset.py                # fast structural + leakage check
python scripts/01_verify_frozen_dataset.py --verify-md5   # plus per-file MD5
```

A passing run prints split totals `{'train': 6090, 'val': 816, 'test': 808}` and zeros for
cross-split exact-MD5 groups, cross-split canonical families, and derived images outside
train. `05_run_experiments.py` re-checks the same lock before any job launches, and each
trainer re-asserts the per-class file counts itself when `--strict_final_counts` is set —
which the campaign config sets for every run.

---

## The experiments

### Token family — the primary question

All four share the same backbone, the same 768→384→7 classifier head, and the same `token`
recipe. They differ only in what is inserted at Stage 4.

| ID | What it is | Key flags | Active module params |
|---|---|---|---:|
| `C1` | Recipe-matched Swin-S control, no token module | `--arch rgb` | 0 |
| `TLA` | Stage-4 token-level attention | `--arch tcca` | 3,545,089 |
| `C2` | Parameter-matched non-attention MLP | `--arch adapter --adapter_expansion 3` | 3,545,089 |
| `C4` | Lightweight ECA-style comparator | `--arch eca` | 1,542 |

(`tcca` is the historical code name of the token module; in the paper and here it is `TLA`.)

### HSV family — the complementary question

| ID | What it is | Key flags |
|---|---|---|
| `R0` | RGB Swin-S baseline with the matched MLP head | — |
| `R1` | HSV with sin/cos hue, per-channel (vector) gate | `--use_hsv --gate_vector` |
| `R2` | Raw 3-channel HSV, scalar gate | `--use_hsv --hsv_raw` |
| `R3` | HSV with sin/cos hue, scalar gate | `--use_hsv` |

### The predeclared comparisons

These eight pairs are written into `configs/experiments.yaml` and are what
`07_paired_bootstrap.py --auto` evaluates:

| A | B | What it isolates |
|---|---|---|
| `TLA` | `C1` | token attention vs. a recipe- and head-matched no-token control |
| `C2` | `C1` | parameter-matched MLP vs. no-token control |
| `C4` | `C1` | lightweight ECA vs. no-token control |
| `TLA` | `C2` | attention *mechanism* vs. equal *capacity* |
| `TLA` | `C4` | full attention vs. a cheap attention comparator |
| `R1` | `R0` | HSV sin/cos, vector gate vs. matched RGB baseline |
| `R2` | `R0` | raw HSV, scalar gate vs. matched RGB baseline |
| `R3` | `R0` | HSV sin/cos, scalar gate vs. matched RGB baseline |

We evaluate these eight comparisons (and any ad hoc pair run through
`12_pairwise_error_analysis.py`) without a multiple-comparisons correction, so any single
interval or p-value is evidence about that one pair, not one of eight independent
confirmations at the nominal level.

### How to read the results

- The primary architectural question is **`TLA` vs `C1`**, read together with `C2` and `C4`.
- **`TLA` vs `C2`** is the mechanism-versus-capacity test. It means something only because the
  two parameter counts are identical, and because `04_validate_model_contracts.py` confirms
  `R0` and `C1` are forward-equivalent under shared weights.
- The HSV question is **`R1`/`R2`/`R3` vs `R0`**, and nothing else.
- **Do not rank a token model against an HSV model.** They are trained under different
  recipes — different learning rate, different batch/accumulation split, different gate
  warm-up. That is why `06_aggregate_results.py` emits two separate LaTeX tables,
  `token_primary_results.tex` and `hsv_complementary_results.tex`, and puts the combined
  view in `all_results_supplement.tex`. `R0` vs `C1` is a *recipe* comparison, and we report
  it only as that.
- The bootstrap intervals describe **test-sample** uncertainty for a fixed checkpoint. The
  across-seed standard deviations in `summary.csv` are the training-variance evidence. They
  answer different questions; don't substitute one for the other.
- The bootstrap resampling is **unstratified** — see `07_paired_bootstrap.py` below.
- The corruption suite is a **controlled** robustness probe. It is not field validation.

---

## How the models work

### The token modules (Stage 4, 49 tokens × 768 dims)

The backbone is wrapped by `SwinWithHooks`, which returns the Stage-4 tokens after Swin's
final LayerNorm. Only these tokens are used: the classifier mean-pools them *after* the
module has run. (v1 also had a Stage-3 branch, but it ran after Stage 4 had already been
computed, so it could not affect classification; we removed it rather than count it as
capacity.) At 224×224 input the Stage-4 map is 7×7, the same as the window size, so the
backbone's own Stage-4 blocks already attend over all 49 tokens.

**`TLA` (`Stage4TCCAFusion`).** Two linear projections produce a key/value source from the
same Stage-4 tokens, an 8-head `MultiheadAttention` attends from the tokens to that source,
the result goes through a LayerNorm residual, and the whole thing is folded back in through
a sigmoid gate:

```text
c        = W_proj · (W_color · t)
attended = MultiheadAttention(Q = t, K = c, V = c)      # 8 heads, head dim 96
h        = LayerNorm(t + attended)
out      = t + α · sigmoid(gate) · h
```

The gate parameter is initialised at −2.0, so `sigmoid(gate) ≈ 0.12` at the start: the
module begins as a small perturbation of the baseline and has to earn its influence.

**`C2` (`Stage4TokenMLPAdapter`).** Same insertion point, same gated residual, same gate
initialisation, same warm-up. The one thing removed is token mixing — it is per-token
channel mixing, so tokens never exchange information. The parameter match is algebraic
rather than approximate:

```text
TLA:      6·D² + 8·D + 1
adapter:  2r·D² + (r+5)·D + 1
2r = 6  ⇒  r = 3,  and (r+5)·D = 8·D
D = 768 ⇒ both equal 3,545,089
```

**`C4` (`Stage4TokenECA`).** ECA-Net style: mean over tokens, a single `Conv1d` with an
adaptively chosen kernel (k = 5 at D = 768) over the channel axis, sigmoid channel weights,
LayerNorm, same gated residual. 1,542 parameters. It is deliberately *not* matched — the
point is to ask whether something nearly free does the same job.

### The HSV branch

Here fusion happens on the pooled 768-d feature vector, not on tokens. The normalised input
is de-normalised back to `[0, 1]`, converted to HSV in forced float32 (AMP is switched off
for the conversion), and — unless `--hsv_raw` — the hue is encoded as
`[sin 2πH, cos 2πH, S, V]` so that the wraparound at 0°/360° is continuous. Two stride-2
convolutions and global average pooling produce a 128-d colour embedding, a linear layer
projects it to 768, and a gate MLP that sees both the RGB feature and the colour embedding
decides how much to let in:

```text
fused = feat + (α · gate) · hsv_feat
```

with the gate's output bias initialised to −2.0 again. `--gate_vector` makes the gate
per-channel instead of scalar.

### Two fairness controls worth knowing about

- **Shared head.** In v1, the RGB baseline used a linear head and the HSV variants a
  two-layer MLP, which confounded any HSV gain with head capacity. Every model in v2 — both
  families — uses the same `Dropout → Linear(768,384) → GELU → Dropout → Linear(384,7)` head,
  initialised from the same seeded stream.
- **Architecture-independent hue jitter.** `hue_jitter` is `0.0` for every experiment and
  never depends on whether HSV is in use. Changing colour augmentation alongside a colour
  branch would make the comparison meaningless.

### Gate warm-up

`α` ramps linearly from `1/gate_warmup_epochs` to 1.0 over the first epochs — 5 for the HSV
family, 10 for the token family — so a new module cannot destabilise a pretrained backbone
early on. **Evaluation always uses α = 1.0**, including validation during the warm-up
epochs.

---

## Training recipe

`scripts/10_make_training_settings_table.py` regenerates this table as LaTeX from the
recorded configs of all 24 runs, and fails if any two runs of a family disagree.

| Setting | HSV family (R0–R3) | Token family (C1/TLA/C2/C4) |
|---|---|---|
| Backbone | `swin_small_patch4_window7_224.ms_in1k`, ImageNet-1k pretrained | same |
| Classifier head | MLP 768–384–7 | same |
| Input size | 224 | 224 |
| Epoch budget | 100 | 100 |
| Physical batch | 64 | 32 |
| Gradient accumulation | 1 | 2 |
| Effective batch | 64 | 64 |
| Optimizer | AdamW, β = (0.9, 0.999) | same |
| Learning rate | 5e-4 | 1e-4 |
| Weight decay | 0.05; norms, biases and gates exempt | same |
| LR schedule | 5-epoch linear warm-up from 1e-6, then cosine to 1e-6, stepped per optimizer update | same |
| DropPath | 0.2 | 0.2 |
| Label smoothing | 0.1 | 0.1 |
| Gradient clipping | 1.0 | 1.0 |
| Mixed precision | AMP (FP16) | AMP (FP16) |
| EMA decay | 0.9998 | 0.9998 |
| Gate warm-up | 5 epochs | 10 epochs |
| Hue jitter | 0.0 | 0.0 |
| DataLoader workers | 0 | 0 |
| Checkpoint criterion | validation macro-F1 | validation macro-F1 |
| Early-stopping patience | 25 epochs | 25 epochs |
| Seeds | 42, 1337, 2026 | 42, 1337, 2026 |

Training augmentation is `RandomResizedCrop(224, scale=(0.85, 1.0), bicubic)` →
`RandomHorizontalFlip(0.5)` → `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
hue=0.0)` → normalise with ImageNet statistics. Evaluation is `Resize(255, bicubic)` →
`CenterCrop(224)` → normalise. No TTA, no MixUp/CutMix, no RandAugment, no random erasing.

Model selection is strictly on validation macro-F1, evaluated under EMA weights. The final
test pass reloads the best checkpoint and overlays the EMA shadow onto it. The selection
metadata records the optimizer-step and skipped-update counts at the selected epoch and at
the end of the run, so you can tell how much optimisation the reported weights actually saw.

---

## Running the campaign

There are two ways to run this: orchestrated through `14_pipeline.py`, or manually through
the numbered scripts. We describe the manual route first because it is what the pipeline
calls under the hood; the orchestrated route is the recommended default.

### The manual route, script by script

#### `00_fetch_frozen_dataset.py` — get the data, record which version

```bash
python scripts/00_fetch_frozen_dataset.py --revision main --overwrite
```

Resolves the Hugging Face revision to a concrete commit SHA, downloads that snapshot,
exposes it at `data/Tea_leaf_dataset/` and `data/manifests/` (by symlink; pass `--copy` if
symlinks are awkward on your system), and writes `data/hf_dataset_lock.json` with the
resolved SHA. Report that SHA, not the branch name — branches move.

#### `01_verify_frozen_dataset.py` — prove the data is the audited partition

Described above. Exits non-zero on any mismatch.

#### `16_audit_derivative_perceptual_leakage.py` — re-verify derivative confinement

```bash
python scripts/16_audit_derivative_perceptual_leakage.py
```

Non-destructive. Re-runs the frozen pHash + embedding operating point between the training
derivatives and the validation/test images, and writes
`results/v2/audits/derivative_eval_perceptual_summary.json` and the pair table. A non-empty
result is a stop condition — treat it like a failing `01_verify_frozen_dataset.py`.

#### `02_preflight.py` — cheap checks before you burn GPU hours

Imports every required package and prints its version (including `ImageHash`), compiles
every script to catch syntax errors, confirms the run matrix resolves to exactly
8 experiments × 3 seeds = 24 jobs, then runs the dataset verification,
`05_run_experiments.py --list`, and `05_run_experiments.py --all --dry-run --skip-done`. It
starts no training.

#### `03_capture_environment.py` — write down what you ran it on

Writes `results/v2/environment.json`: Python, platform, torch / torchvision / timm /
numpy / pandas / scikit-learn / Pillow versions, CUDA and cuDNN versions, GPU names, a full
`pip freeze`, the current Git commit SHA and whether the working tree is clean, and the
SHA-256 of `configs/experiments.yaml` and of the dataset manifest.

#### `04_validate_model_contracts.py` — build the architectures and check the algebra

Instantiates R0, R1, C1, TLA, C2 and C4 without touching the network or the data, prints
their parameter counts, and enforces:

- **R0 and C1 have identical total parameter counts** — they are the same RGB Swin-S with
  the same head.
- **R0 and C1 are forward-equivalent** — weights are copied between the two models and a
  fixed input must give identical outputs.
- **TLA and C2 both land on exactly** 6·768² + 8·768 + 1 = 3,545,089 active parameters.
- **C1's final Swin `LayerNorm` receives a finite, non-zero gradient** — a check against a
  detached backbone tail in the no-module control.
- **A sanity guard on `C4`**, given how easy a misconfiguration is to miss on an
  architecture this small.

It exits non-zero on any failure.

#### `05_run_experiments.py` — the training driver

```bash
python scripts/05_run_experiments.py --list                 # what exists
python scripts/05_run_experiments.py --all --dry-run        # print every command, run none
python scripts/05_run_experiments.py --only TLA --seeds 42  # one run
python scripts/05_run_experiments.py --family token         # one family, all seeds
python scripts/05_run_experiments.py --all --seeds 42       # the seed-42 campaign
python scripts/05_run_experiments.py --all --seeds 1337 2026
python scripts/05_run_experiments.py --all --skip-done      # resume a partial campaign
```

It verifies the dataset lock first and refuses to launch on a mismatch. For each job it
builds the full command line from `configs/experiments.yaml`, runs it as a subprocess,
verifies the declared parameter budget against what the run reported, and writes
`config/run_provenance.json` with the `tea_leaf_v2` campaign ID.

`--skip-done` treats a run as finished when `metrics/test_results.json` exists. Every
training command is launched with `--auto_resume`, so an interrupted run picks up from
`results/v2/ckpt/<RUN>/last_checkpoint.pth` — we relied on this for runs trained on
session-limited machines.

#### `06_aggregate_results.py` — turn run directories into tables

```bash
python scripts/06_aggregate_results.py
python scripts/06_aggregate_results.py --family token
```

Walks `results/v2/runs/`, reads each run's `config.json` and `test_results.json`, and
writes to `results/v2/tables/`:

- `per_run.csv` — one row per (experiment, seed)
- `summary.csv` — mean ± sd across seeds
- `per_class_f1.csv` and `per_class_f1.tex` (F1 in %)
- `token_primary_results.tex`, `hsv_complementary_results.tex` — the two main tables
- `all_results_supplement.tex` — the combined view, for a supplement only

It warns when an experiment has fewer than three seeds.

#### `07_paired_bootstrap.py` — how much of a difference is sampling noise?

```bash
python scripts/07_paired_bootstrap.py --auto --n-boot 10000
python scripts/07_paired_bootstrap.py --a results/v2/runs/TLA_s42 \
                                      --b results/v2/runs/C1_s42
```

10,000 resamples; the resampling unit is the individual test image; the same index vector is
applied to both models in each replicate, so the difference is genuinely paired; macro-F1 is
recomputed from scratch per replicate against a fixed 7-label set; the interval is the
2.5th–97.5th percentile of the difference.

Resampling is **unstratified**: per-class counts vary across replicates. We make no
directional claim about what that does to minority-class intervals; if per-class behaviour
matters for a comparison, read `per_class_f1.csv` alongside the bootstrap output.

The two runs are aligned by **image ID**, not by assumed ordering: each run saves
`raw_outputs/test_ids.json`, and the script reorders B to match A and then verifies the
ground-truth vectors agree. It refuses to run if the IDs are missing or differ.

Writes `bootstrap.csv` and `bootstrap_summary.csv`. The fraction of replicates with Δ > 0 is
reported as a descriptive fraction, not a p-value.

#### `08_eval_robustness.py` — controlled corruptions

```bash
python scripts/08_eval_robustness.py --planned --seeds 42 --num-workers 0 \
       --out results/v2/tables/robustness_parts/robustness_s42.csv
```

Evaluates 18 deterministic corruptions plus the unperturbed condition: brightness ±0.20,
contrast ×0.75 / ×1.25, saturation ×0.5 / ×1.5, hue ±0.05, Gaussian noise σ = 0.03 / 0.08,
Gaussian blur radius 1.0 / 2.5, JPEG quality 60 / 30, left and right shadow ramps, and warm
and cool white balance. Noise is seeded from a hash of the image path, so every model sees
identical corruptions. The unperturbed path is exactly the training-time evaluation
transform.

**Clean-score contract.** Before any corruption is scored, the script recomputes each run's
unperturbed macro-F1 from `ema_model_weights_only.pth` and requires it to equal the stored
test score. The one accepted exception is a near tie: at most two images may resolve to a
different class, and only if their stored top-2 probability margin is ≤ 1e-3. A wrong
checkpoint, preprocessing or class order flips many confident images and still stops the
run. Across our 24 runs, 23 reproduce their stored score exactly. The exception is `C1_s2026`:
`Healthy leaf/healthy_00791.jpg` has exactly equal stored probabilities for Healthy leaf and
Helopeltis, and other hardware breaks the tie the other way. The contract result and device
are recorded per run.

We ran one seed at a time into `results/v2/tables/robustness_parts/` and merged them with
`21_merge_robustness.py`. This is a stress test under synthetic perturbations. It says
nothing about a different farm, a different camera, or a different season.

#### `09_make_figures.py` — figures from the tables, never from memory

```bash
python scripts/09_make_figures.py --format pdf --dpi 300
```

Produces, for whichever input tables exist: class distribution per partition, the offline
augmentation inventory, per-family macro-F1 with seed spread (separate figures per family,
so the plot can't imply a cross-family ranking), macro-F1 by perturbation family,
unperturbed vs. mean-stress retention, and the paired-bootstrap deltas with their intervals.
Output goes to `results/v2/figures/`.

#### `10_make_training_settings_table.py`

Regenerates the consolidated hyperparameter table as LaTeX from the recorded configs of all
completed runs, and refuses to write it if two runs of the same family disagree.

#### `13_summarize_init_scale.py` — aggregate the HSV scale diagnostics

```bash
python scripts/13_summarize_init_scale.py
```

Collects every HSV run's `init_scale_diagnostics.json` across seeds into one summary table,
so the question "did the branch start too small to matter?" has a campaign-level answer.

#### `11_export_reproducibility_bundle.py`

```bash
python scripts/11_export_reproducibility_bundle.py
```

Packages a shareable bundle: code, configs, docs, manifests, the dataset lock, the
environment capture, the generated tables, and every run's config / metrics / logs / raw
predictions and IDs, plus `PACKAGE_MANIFEST.json` and the derivative-audit output. Image
data and model checkpoints are deliberately excluded.

#### `12_pairwise_error_analysis.py` — per-image disagreement

```bash
python scripts/12_pairwise_error_analysis.py \
  --a results/v2/runs/TLA_s42 \
  --b results/v2/runs/C1_s42 \
  --out results/v2/tables/error_TLA_vs_C1_s42.json
```

Aligns two runs by image ID and reports how many images each got right that the other
missed, plus an exact McNemar two-sided p-value. It refuses to compare runs from different
campaigns and states in its output that the p-value is uncorrected for multiplicity. We ran
it for TLA vs C1 and TLA vs C2 on each seed.

#### `15_package_manifest.py` — maintenance only

```bash
python scripts/15_package_manifest.py --write   # regenerate PACKAGE_MANIFEST.json
python scripts/15_package_manifest.py           # verify current files against it
```

Run `--write` only after intentional code changes are finished, and run it without `--write`
afterwards to confirm nothing has drifted.

### Post-hoc failure analysis — exploratory, not predeclared

These two scripts look *into* the finished runs to explain what went wrong. They are not
among the predeclared analyses, and `14_pipeline.py` does not run them. Both are read-only
on runs and images and write under `results/v2/analysis/`. Image copies go in an `images/`
subfolder that is gitignored, in line with our rule that image data is not versioned here.

#### `17_consensus_failures.py` — the images every arm gets wrong

```bash
python scripts/17_consensus_failures.py --seed 42          # every completed arm
python scripts/17_consensus_failures.py --min-arms 5       # wrong in at least 5 arms
```

Finds the test images that every arm misclassifies and copies them into
`consensus_failures_s<SEED>/images/<true>__as__<predicted>/`, with a one-page
`contact_sheet.png`. `consensus_failures_s<SEED>.csv` records every arm's prediction and confidence,
whether they agree on the wrong label (a tie is reported as a tie), whether an image is a
surviving near-duplicate of another test image, and `expert_label` / `expert_notes` columns
for the label review.

Across the three seeds, 9, 13 and 12 test images are misclassified by all eight arms. Eight
images are misclassified by all 24 runs, and for 7 of those every run predicts the same
wrong class — more consistent with a labelling question or genuine ambiguity than with a
capacity gap. We reviewed all 17 images that all eight arms miss on at least one seed. The
review confirmed none of their dataset labels: 5 are suspected mislabels (each looks like
the class the models predict) and 12 are ambiguous; of the eight missed by all 24 runs, 2
are suspected mislabels and 6 ambiguous. That review is provisional and visual, not expert
ground truth, and we have changed no label.

#### `18_confusable_pair_analysis.py` — why two classes are confused

```bash
python scripts/18_confusable_pair_analysis.py --seed 42                   # Brown Blight vs Tea algal leaf spot
python scripts/18_confusable_pair_analysis.py --seed 42 --with-models     # adds embedding probes + gates (CPU)
python scripts/18_confusable_pair_analysis.py --pair "Gray Blight" "Red spider"
```

Answers, for one pair of classes: how often each arm confuses them; whether colour is
discriminative in these photographs at all (colour-only classifiers, fixed in advance,
fitted on training originals and scored on test, using whole-image colour and colour inside
a heuristic lesion mask); how much of each image is lesion; whether the networks fail where
colour fails; and, with `--with-models`, whether the trained HSV branch's pooled embedding
carries the signal. The lesion mask is a colour heuristic, not a segmentation;
`images/mask_overlay.png` shows it on the confused images so it can be checked by eye.
`--with-models` runs on CPU by default so a training job keeps the GPU.

What we found for Brown Blight vs Tea algal leaf spot (probes on seed 42; everything else
holds on all three seeds):

| Evidence | Value |
|---|---|
| Heuristic lesion area, share of the **whole image** (median) | 4.6% vs 2.6% |
| Heuristic lesion area, share of the **leaf** (median) | 16.6% vs 10.4% |
| Leaf share of the image — framing check (median) | 27.9% vs 27.9% (p = 0.81) |
| Lesion hue / saturation (median) | 36.2° vs 39.7° / 0.50 vs 0.49 |
| Colour-only AUC, whole image → lesion colour → lesion colour + extent | 0.64 → 0.71 → 0.82 |
| Linear probe on the RGB backbone feature (R0–R3) | 0.97–0.98 |
| Linear probe on the trained HSV-branch embedding (R1–R3) | 0.61–0.66 |
| Majority-confused test images that colour also gets wrong (seed 42) | 7 of 8 |
| Pair confusions over three seeds, R0 vs R1 / R2 / R3 | 32 vs 36 / 32 / 39 |

The two lesion types are nearly the same colour in these photographs, and they cover a few
percent of each image. The R1–R3 HSV branch ends in global average pooling over the whole
image, so its embedding retains about as much pair information as the image's mean colour.
What separates these classes is lesion extent and texture, which the RGB backbone already
captures.

The two area rows use different denominators on purpose. Share of the whole image is what a
globally pooled branch receives, so it is the right figure for the dilution argument. Share
of the leaf is the right figure for lesion burden. The leaf fills only about 28% of each
image, and equally so in both classes, so the extent difference is not a framing artefact.

**The lesion mask is a heuristic, and our visual review of the overlay found real
limitations**, which checks on the same images confirmed. It substantially under-segments
dark brown and black lesions (on `brown_blight_00221` and `00222` it catches roughly 35–45%
of dark tissue). It also selects leaf margins, veins, tears and hole boundaries: about
15–16% of the mask in both classes, and two thirds of it on `UNADJUSTEDNONRAW_thumb_168`.
And it picks up occasional debris on the paper (9% of the mask beside `brown_blight_00464`).
Because margin contamination is the same in both classes it adds noise rather than bias, and
the dark-lesion misses fall mostly on Brown Blight, so they work against the extent
difference, not for it. Even so, we treat every lesion figure as heuristic-selected area,
not validated lesion severity. Quantitative accuracy needs expert reference masks.

A seed-42 observation that the HSV gate opened widest on the classes where colour lowered
F1 **did not replicate** on seed 1337, and we do not report it as a finding. What holds
across seeds is only that the widest-opening gate falls on Tea algal leaf spot or Brown
Blight in 8 of 9 HSV runs.

### Manuscript helpers

#### `19_make_revision_tables.py` — the remaining manuscript tables

```bash
python scripts/19_make_revision_tables.py
```

Reads existing artifacts only, with no weights and no GPU, and writes to `results/v2/tables/`:
`bootstrap_per_seed.tex` (all 24 predeclared intervals), `model_complexity.tex` (the eight
arms), `consensus_failures.tex` (images wrong in all 24 runs, with the provisional visual
note), `pair_confusion.tex` (Brown Blight / Tea algal leaf spot confusions per arm), and,
once the merged robustness files exist, `robustness_summary.tex` and
`robustness_grouped.tex`. It stops if any predeclared interval excludes zero, because the
bootstrap caption would then be wrong.

#### `20_embed_tables.py` — make the manuscript compile on its own

```bash
python scripts/20_embed_tables.py            # default: revision.tex
```

Replaces each `\gentable{NAME}` line in the manuscript with the contents of
`results/v2/tables/NAME.tex`, between `% >>> generated` / `% <<<` markers, so the file
compiles on Overleaf without a separate `tables/` folder. Re-running it refreshes the text
between the markers, so we run it again whenever `06`, `10` or `19` regenerate a table.

#### `21_merge_robustness.py` — combine per-seed robustness runs

```bash
python scripts/21_merge_robustness.py
```

`08` writes `robustness_summary.csv` next to its `--out` file, so running one seed at a time
into `robustness_parts/` leaves only the last seed's summary there. This script rebuilds
`robustness.csv`, `robustness_summary.csv` (with the worst condition, the clean-score
contract result and the device per run) and `robustness_grouped.csv` in
`results/v2/tables/` from the per-seed `robustness_s<seed>.csv` files. It refuses to run
unless all 24 runs are present with all 19 conditions.

### The orchestrated route: `14_pipeline.py`

```bash
python scripts/14_pipeline.py --prepare
python scripts/14_pipeline.py --smoke
# Inspect R0_s42 and C1_s42 before continuing.
python scripts/14_pipeline.py --seed42
python scripts/14_pipeline.py --remaining
python scripts/14_pipeline.py --analyze
python scripts/14_pipeline.py --export
```

Each stage wraps a fixed set of the numbered scripts, and the stages must be run in order —
`--seed42` refuses to run before `--smoke` has produced `R0_s42` and `C1_s42`, and so on:

| Stage | Runs |
|---|---|
| `--prepare` | `00`, `01`, `16`, `02`, `03`, `04` |
| `--smoke` | `05` for `R0_s42` and `C1_s42` only |
| `--seed42` | `05` for the remaining six experiments at seed 42 |
| `--remaining` | `05` for all eight experiments at seeds 1337 and 2026 |
| `--analyze` | `06`, `10`, `13`, `07`, `08`, `09` |
| `--export` | `15`, `11` |

Inspecting `R0_s42` and `C1_s42` before continuing is not optional ceremony: it is the point
in the campaign where a smoke-test failure is cheapest to catch. The post-hoc scripts
(`17`, `18`) and the manuscript helpers (`19`–`21`) are run by hand.

---

## What a run writes

```text
results/v2/
  ckpt/<RUN>/                       live checkpoints while training (gitignored)
  runs/<EXP>_s<SEED>/
    config/
      config.json                   every effective setting for this run
      env.json                      torch / timm / CUDA versions on the training machine
      classes.json, class_to_idx.json
      run_provenance.json           argv, source SHA-256s, dataset lock, campaign_id,
                                     timestamp, optimizer-step / skipped-update counts
      resume_history.jsonl          only for resumed runs
    logs/
      train_log.csv                 one row per epoch
      gate_stats_log.csv            HSV runs only: per-class gate statistics per epoch
    metrics/
      test_results.json             acc1, macro/micro/weighted F1, macro-AUC, params, GFLOPs
      per_class_metrics.json        per-class precision / recall / F1
      classification_report.txt
      confusion_matrix.npy
      init_scale_diagnostics.json   R1–R3 only: pre-optimization HSV/RGB scale measurement
    raw_outputs/
      test_ids.json                 808 stable image IDs, in evaluation order
      test_predictions.npy          argmax predictions
      test_targets.npy              ground truth
      test_probabilities.npy        float64 softmax, (808, 7)
      gate_stats_test.json          HSV runs only: per-class gate statistics on test
    figures/confusion_matrix.png
    model/                          weights (gitignored; ema_model_weights_only.pth is what we evaluate)
  tables/                           everything 06, 07, 10, 12, 13, 19 and 21 generate
  figures/                          everything 09 generates
  analysis/                         17 and 18 outputs
  audits/                           16 outputs
```

`test_ids.json` is the piece that makes the downstream analysis honest. It holds entries like
`Brown Blight/UNADJUSTEDNONRAW_thumb_107.jpg`, aligned index-for-index with the prediction
arrays, and is checked for length and uniqueness. Without it, a "paired" bootstrap would just
be an assumption that two harnesses enumerated the test set in the same order.

`init_scale_diagnostics.json` deserves a note: for the HSV runs it records, on the first
validation batch and **before the first optimizer update**, the RMS of the RGB feature, the
RMS of the projected HSV contribution, the raw sigmoid gate statistics, and the resulting
change in logits. It answers the reasonable objection that an HSV branch might simply start
too small to matter. It is written once and never overwritten on resume.

---

## Campaign status (v2)

**The campaign is complete.** All 24 training runs are finished (8 experiments × 3 seeds),
each verified against its stored predictions, test IDs, configuration and campaign ID. Where
each seed was trained, and which runs were resumed, is recorded in `docs/reproducibility.md`.

| Experiment | seed 42 | seed 1337 | seed 2026 |
|---|---|---|---|
| R0 | ✅ | ✅ | ✅ |
| R1 | ✅ | ✅ | ✅ |
| R2 | ✅ | ✅ | ✅ |
| R3 | ✅ | ✅ | ✅ |
| C1 | ✅ | ✅ | ✅ |
| TLA | ✅ | ✅ | ✅ |
| C2 | ✅ | ✅ | ✅ |
| C4 | ✅ | ✅ | ✅ |

| Analysis | Status |
|---|---|
| Aggregation, paired bootstrap, McNemar, settings table, init-scale summary, figures | done, all three seeds |
| Framing decision (`docs/FRAMING_DECISION_RULE.md`) | **Framing B**: TLA vs C1 favours TLA on 0 of 3 seeds, TLA is below C2 on all three, and none of the 24 predeclared intervals excludes zero |
| Consensus failures and the Brown Blight / Tea algal leaf spot analysis | done, all three seeds (backbone probes on seed 42) |
| Controlled robustness (`08`, merged by `21`) | done, 24 runs × 18 corruptions; no add-on shows a detectable robustness improvement over its matched baseline |
| Derivative-vs-evaluation leakage audit (`16`) | done, 0 linked pairs |

---

## Reproducibility notes

We take determinism seriously but do not promise it absolutely:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set at the top of both trainers, before torch is
  imported, because deterministic cuBLAS GEMMs require it.
- Every run seeds Python, NumPy and torch, sets `cudnn.deterministic = True`,
  `cudnn.benchmark = False`, disables TF32 on both matmul and cuDNN, and calls
  `torch.use_deterministic_algorithms(True)`. If an op has no deterministic kernel, that call
  degrades to a warning and the run continues with partial determinism.
- Both recipes load data in the main process (`num_workers: 0`), because worker parallelism
  is a determinism risk.
- Each run's `run_provenance.json` records the exact argv, the SHA-256 of the training script
  and of `configs/experiments.yaml` at launch time, the dataset manifest SHA, the Hugging Face
  dataset lock, the campaign ID (`tea_leaf_v2`), the Python version and platform string, a
  UTC timestamp, and the optimizer-step and skipped-update counts.
- `PACKAGE_MANIFEST.json` gives a content-hash inventory of the code itself, independent of
  Git.

Our three seeds were trained on different machines (seed 42: torch 2.14.0 on an RTX A4000;
seeds 1337 and 2026: torch 2.5.1 on RTX A6000 / RTX 6000 Ada, with two seed-2026 arms on an
RTX A5000 and an RTX 3090; timm 1.0.29 throughout). Determinism holds within a machine, not
across machines, so these differences act as seed-level noise. `docs/reproducibility.md`
has the full table, the resumed runs, and why recorded source hashes differ only in line
endings.

If a library upgrade changes the numbers, report the captured environment rather than
quietly adjusting the recipe. Two standing rules we hold ourselves to, because they are easy
to violate without noticing:

- **No new architectures after looking at test results.** The matrix in
  `configs/experiments.yaml` is the predeclared campaign.
- **No change to model or hyperparameter design after examining final test outcomes.** The
  training-recipe table exists to be checked against, not adjusted toward, a result we have
  already seen.

---

## How the frozen dataset was built

`scripts/dataset_cleaning (1..5).py` are the archival record of the pipeline that produced
the frozen release. They are not part of the experiment pipeline and are not run again — the
released snapshot is the source of truth — but we keep them so the build is inspectable.

1. **`(2)` build_clean_dataset** — index the source 70/15/15 split, verify every image is
   readable, MD5 everything, and delete any training image that duplicates or shares a
   filename family with a validation/test image. Filename-based and hash-based only; the
   script says so explicitly.
2. **`(3)` build_dedup_split** — the perceptual audit. pHash plus ResNet-50 embeddings, with
   an `audit` mode that sweeps thresholds and writes montages for visual inspection before any
   file is touched, and a `build` mode that re-splits over duplicate clusters.
3. **`(4)` finalize_dataset** — the mode we actually used: *prune* rather than re-split.
   Delete the evaluation-side member of every cross-partition near-duplicate pair, leaving
   training untouched.
4. **`(5)` recover_aug_params** — recover the offline augmentation parameters by measuring
   each derivative against its source (ORB plus a partial affine fit for rotation and zoom, a
   least-squares pixel fit for brightness and contrast, a residual sweep for blur sigma),
   because the generator settings were not recorded at the time.
5. **`(1)` build_tea_leaf_dataset** — the consolidated eleven-stage build that produced the
   final manifests, the fingerprint, and the montages shipped in `data/manifests/`.

---

## Future designs

`scripts/fd_scripts/` and `configs/future_designs.yaml` hold a *separate, unrun* campaign: a
shared-weight chromatic branch that would reuse the RGB Swin's large attention and MLP
matrices and add only a small HSV patch stem plus per-block coupling vectors.

| ID | Design | Extra params | Extra compute |
|---|---|---:|---:|
| F0 | shared-weight dual trajectory, independent self-attention | 6,240 | +100% |
| F1 | asymmetric shared-KV colour branch, a/b/c coupling | 60,384 | +84% |
| F2 | minimal a/c coupling | 24,288 | +84% |
| F3 | confidence-mixture readout on F1 | 0 over F1 | +84% |
| F4 | attention-ablation control (one attention pass; parameter-matched, not compute-matched) | 60,384 | +66% |
| F5 | information-null compute-matched control (F1's graph, branch fed `[R,G,B,Y]`) | 60,384 | +84% |

"Lightweight" is true of parameters only: every design runs a second trajectory through the
shared weights. F1 vs F5 is the primary comparison, because it is the only one that
separates a chromatic benefit from a compute benefit.

**Status: NO-GO.** `07_power_analysis.py` shows that on the current test set the contestable
headroom is 2.47 macro-F1 points, while the median minimum detectable effect is 2.00 points —
above our preregistered threshold of half the headroom. No design can show a credible gain on
this test set, and the three-seed v2 results (no colour benefit anywhere, colour-only
features weaker than the backbone) give no reason to expect one.
`docs/FUTURE_COLOR-BRANCH-DESIGN.md` and `docs/FUTURE_SHARED_COLOR_BRANCH.md` carry the
reasoning, the three go/no-go gates, the parameter and compute accounting, and the list of
ablations a future paper would owe. Any future design is a new predeclared campaign; it
cannot join this one, whose test results we have inspected.

The tree is deliberately isolated so that `05_run_experiments.py --all` can never touch it,
and its outputs go to `results/future_designs/` — not `results/v2/`. **Nothing in it is part
of the current evidence.** The shared-KV Swin block adapter is a reference contract rather
than a working implementation, and timm's Swin internals must be inspected and pinned before
anyone writes it.

The tree is **design-only, not an executable pipeline**, and that is enforced rather than
merely documented. Nine files are self-contained, runnable tools — the parameter accounting,
the shared-colour components and their unit tests, the shared-KV reference contract, the
confidence-mixture demo, the attention-ablation copy control, the timm Swin API inspector,
the power analysis (`07`), and the parameter/compute accounting (`08`). Every other filename
in the directory is a thin delegation to `_entry.py`, which forwards dataset preparation to
the real `scripts/` implementation, prints the declared F0–F5 designs for `--list` /
`--dry-run`, runs the component contracts for `--prepare`, and otherwise exits with "Future
training/analysis is not implemented." No file under `fd_scripts/` can read
`configs/experiments.yaml` or write to `results/v2/`.

---

## Citing this work

If you use this code or the audited partition, please cite both this repository and the
underlying TeaLeafBD release. `CITATION.cff` has the machine-readable version.

Authors: MD Shaifullah Sharafat, Md Nahin Alam, Mehrab Karim Opee, Nilavro Das Kabya
(Electrical and Computer Engineering, North South University), Mohammad Aminul Islam (Plant
Pathology, Habiganj Agricultural University), and Riasat Khan (Electrical and Computer
Engineering, North South University).

## Licence

The code in this repository is MIT licensed — see `LICENSE`. The TeaLeafBD image data is
**not** covered by that licence and remains governed by the terms of its original Mendeley
Data release.
