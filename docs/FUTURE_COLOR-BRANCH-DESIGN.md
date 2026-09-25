# Future color-branch designs

We keep here the research designs we worked out after reading:

- `2608.12385v2` (Decode-Branch Transformer)
- `2609.13285v2` (Grouped Value Attention)

We keep them separate so `scripts/05_run_experiments.py --all` can never run
them accidentally. Nothing from this folder should be cited until a new frozen
experiment matrix is declared and all required multi-seed controls are run.

**Status: design-only, and currently NO-GO.** No design below may be trained until
every gate in [Go / no-go](#go--no-go) passes. On the current test set the power
gate already fails (see below). Because the current campaign's test results have
been inspected, any future design is a new predeclared campaign; it cannot be
added to the `tea_leaf_v2` paper.

## Design ladder

Extra parameters and compute are measured against the RGB Swin-S baseline
(8.740 GMACs at 224x224). Regenerate every number here with
`python scripts/fd_scripts/08_cost_accounting.py`; it asserts the parameter counts
against `configs/future_designs.yaml`.

| ID | Design | Extra params | Attention passes / block | Extra compute |
|---|---|---:|---:|---:|
| F0 | shared-weight dual trajectory, independent self-attention | 6,240 | 2 | +100.1% |
| F1 | asymmetric shared-KV colour branch, a/b/c coupling | 60,384 | 2 | +84.3% |
| F2 | minimal asymmetric shared-KV branch, a/c coupling | 24,288 | 2 | +84.2% |
| F3 | confidence-mixture readout on top of F1 | 0 over F1 | 2 | +84.3% |
| F4 | attention-ablation gated-copy control | 60,384 | 1 | +65.9% |
| F5 | information-null compute-matched control | 60,384 | 2 | +84.3% |

1. **F0** — the same Swin stage weights evaluated for RGB and HSV trajectories,
   each with its own self-attention. Establishes whether weight sharing itself is
   viable.
2. **F1** — target design. The RGB path is structurally independent for fixed
   weights; the colour path forms its own query and reads the RGB keys and values.
   All large Q/K/V/O/MLP matrices are shared.
3. **F2** — F1 without the MLP-intermediate b-vector. A new simplified design, not a
   faithful copy of Decode-Branch.
4. **F3** — F1 with a confidence-mixture readout, testing both the original clipped
   collision gate and a 7-class-normalised collision gate. Precondition below.
5. **F4 — attention-ablation control.** F1 with the branch's attention sub-layer
   replaced by `g * primary attention output`; the MLP sub-layer and the b/c
   couplings are unchanged. `g` replaces the query coupling `a` (both D-sized), so
   F4 is **parameter-matched to F1 but runs one attention pass per block instead of
   two**. It asks whether the branch's own attention pass matters. It is *not*
   compute-matched; earlier versions of this document called it that, and the file
   implementing it was renamed from `04_compute_matched_copy_control.py` to
   `04_attention_ablation_copy_control.py` accordingly.
6. **F5 — information-null compute-matched control.** F1's exact graph, couplings,
   stem and readout, with the branch stem fed `[R, G, B, Y]` (`rgb_luma_rep`)
   instead of sin/cos HSV. Four input channels keep the stem at 6,240 parameters, so
   F5 matches F1 in both parameters and compute while carrying no information the
   primary does not already see. **F1 vs F5 is the primary comparison**: it is the
   only one that separates a benefit of the chromatic re-parameterisation from a
   benefit of 84% more compute. Without F5, a gain for F1 is uninterpretable.

The preregistered comparisons are listed under `future_campaign` in
`configs/future_designs.yaml`.

## Parameter accounting for Swin-S

Dims/depths: `[96,192,384,768]`, `[2,2,18,2]`; `sum depth*dim = 9,024`.

- faithful a/b/c vectors with MLP ratio 4: `6 * 9,024 = 54,144`
- minimal a/c vectors: `2 * 9,024 = 18,048`
- 4-channel -> 96 patch stem, 4x4 with bias: `4*96*16 + 96 = 6,240`
- faithful total before optional readout: 60,384 (~0.060M)
- minimal a/c total before optional readout: 24,288 (~0.024M)

Run `python scripts/fd_scripts/00_parameter_accounting.py` to regenerate these
figures rather than copying them by hand.

**Parameters are not the cost.** Every design except the readout-only F3 sends a
second trajectory through the shared weights, so the colour branch roughly doubles
compute while adding almost no parameters. Never quote the 0.024–0.060M figure
without the +66–100% compute figure next to it.

## Evidence from the `tea_leaf_v2` runs (seed 42, with the three-seed check noted)

We record this here because a future design has to answer it, not route around it.
Produced by `scripts/18_confusable_pair_analysis.py`, `scripts/07_paired_bootstrap.py`
and `scripts/fd_scripts/07_power_analysis.py`; exploratory except for the bootstrap.

- **No HSV arm beats the RGB baseline.** On seed 42 all three lost (macro-F1 −0.80,
  −0.64, −1.23); over three seeds the mean differences are −0.01, −0.06 and −0.40, and
  none of the nine per-seed intervals excludes zero.
- **Gate behaviour — corrected after seeds 1337 and 2026.** On seed 42 the per-class
  gate correlated negatively with per-class ΔF1 (−0.66 for R2, −0.68 for R3). **This did
  not replicate:** on seed 1337 the correlation is positive for all three HSV arms, so it
  is not a finding and must not be cited. What does hold across seeds: the class with the
  widest mean test gate is Tea algal leaf spot or Brown Blight in 8 of 9 HSV arm-seed
  runs (the exception is R2 at seed 1337, Gray Blight), while gate magnitudes are
  unstable across seeds (largest class mean from 0.04 to 1.00).
- **Brown Blight vs Tea algal leaf spot, the dominant consensus confusion:**
  heuristic lesion area is a median 4.6% and 2.6% of the whole image (16.6% and
  10.4% of the leaf; the leaf fills ~28% of the image in both classes), and lesion
  hue differs by only ~3.5° (36.2° vs 39.7°) at the same saturation (0.50 vs
  0.49). The lesion mask under-segments dark lesions and picks up leaf margins and
  hole edges, so these are heuristic areas, not validated severity. A colour-only
  classifier on whole-image statistics reaches test AUC 0.64; on lesion-masked
  colour 0.71; lesion colour plus lesion extent and count 0.82. A linear probe on
  the RGB backbone feature reaches **0.97–0.98**.
- **The trained R1–R3 HSV embedding carries almost none of it:** probe AUC
  0.61–0.66, the same as whole-image mean colour. The branch ends in global average
  pooling over the whole image, so a lesion signal covering a few percent of pixels
  is diluted into the leaf and background. (Whole-image share is the relevant
  denominator here, because it is what that pooling averages over.)

What this means for the designs here, stated both ways:

- *In their favour:* F0–F5 feed colour through a spatial trajectory of Swin tokens,
  not a globally pooled vector, so they do not repeat the specific dilution failure
  of R1–R3.
- *Against them:* even lesion-localised colour is a weak discriminator for the
  hardest pair (AUC 0.71), and the RGB features already separate it far better.
  The room for a colour-specific gain on this dataset is small.

## Go / no-go

All three gates are declared in `configs/future_designs.yaml` under `go_no_go`.

1. **Robustness.** Run `08_eval_robustness.py --planned` on the finished current
   campaign. Go only if at least one HSV arm differs from R0 under the colour
   perturbations by more than its paired-bootstrap interval, in either direction.
   **Status:** the three-seed robustness pass is done. Under hue shift, saturation and
   white balance the HSV arms lie within 0.8 macro-F1 points of R0, inside the seed
   spread. We did not bootstrap the perturbed conditions, so this gate is formally
   open, but the evidence points the same way as the power gate.
2. **Power.** Run `07_power_analysis.py` on the intended evaluation set. Go only if
   the median minimum detectable effect at 80% power is at most half the contestable
   headroom. **Current test set, seed 42: headroom 2.47 macro-F1 points, median
   MDE80 2.00 points against a threshold of 1.23 — NO-GO.** The tightest pair (TLA
   vs C1) needs 1.59 points. No design can show a credible gain on this test set;
   a harder or shifted evaluation set is required.
3. **Evidence.** The design must address the findings above.

### F3 precondition

A mixture of two heads can only recover images that exactly one head gets right.
Before crediting F3, report `prediction_diversity(primary, colour head)` from
`01_shared_color_components.py` on validation. For reference, the independently
trained seed-42 arms — different architectures, different weights — already have a
median error correlation of phi = 0.62. A colour branch sharing 99.9% of its
weights with the primary will be more correlated than that, not less.

## Important conceptual caveat

The primary RGB **forward graph** does not read the colour branch. That does not
mean a jointly trained RGB predictor has identical learned weights to a
separately trained baseline, because branch-loss gradients can update shared
weights. A future paper should distinguish structural forward independence from
training-time co-adaptation, and compare joint training against a frozen-primary
variant.

## Implementation status

The shared-KV Swin design is intentionally kept as a contract/reference
implementation because block-level Swin attention integration is
timm-version-sensitive. Run `05_inspect_timm_swin_api.py` in the future
environment before implementing the adapter, then freeze that timm version in a
separate campaign. F5 must be implemented alongside F1.
