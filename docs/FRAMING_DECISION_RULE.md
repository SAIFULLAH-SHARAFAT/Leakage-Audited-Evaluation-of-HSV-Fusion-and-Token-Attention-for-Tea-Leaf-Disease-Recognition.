# Framing Decision Rule

Written and committed **before** any 3-seed aggregate, paired bootstrap, or score
from seeds 1337 and 2026 was computed or viewed. Those 16 runs were checked for
integrity only, with every check reporting PASS/FAIL and no metric printed.

In the interest of full disclosure: seed 42 **had** already been inspected when this
rule was written. On seed 42, TLA vs C1 was +0.08 Macro-F1 points (net one image of
808), C2 scored above TLA, and all three HSV variants scored below R0. The rule is
therefore pre-specified with respect to seeds 1337 and 2026, not with respect to
seed 42.

Both framings are kept, and this rule decides between them mechanically.

## Primary quantities (fixed in advance)

- Metric: test Macro-F1 over the fixed 7-class label set.
- Paired bootstrap: `scripts/07_paired_bootstrap.py`, B = 10,000, resampling unit =
  test image, one shared index vector per replicate, percentile 95% interval,
  unstratified, computed separately per seed.
- A comparison "favours X" on a seed when that seed's pointwise 95% interval for
  (X − Y) lies entirely above zero.

## Framing A — "token attention, narrowly claimed"

Chosen **only if both** hold:

1. **TLA vs C1** favours TLA on at least 2 of the 3 seeds; and
2. the three-seed mean Macro-F1 of **TLA is at least that of C2** (the
   parameter-matched, non-attention control).

Under A, TLA is the primary contribution, claimed only as far as the controls allow:
an improvement over a recipe- and head-matched control that is not explained by
added capacity.

## Framing B — "a leakage-audited, controlled evaluation"

Chosen **otherwise**. Under B, the contribution is:

- the integrity-audited TeaLeafBD protocol;
- the matched-control finding on whether Stage-4 token attention (TLA vs C1, C2,
  C4) and post-pooling HSV fusion (R1–R3 vs R0) help — reported as measured,
  including a null result;
- the mechanistic account of why the HSV branch fails where it does (global-pooling
  dilution, gate opening on the classes where colour hurts);
- evidence on the nature of the residual errors (consensus failures and the expert
  review).

## Secondary rule — HSV claims (applies under either framing)

An HSV variant may be described as improving on R0 **only if** R_k vs R0 favours R_k
on at least 2 of 3 seeds. Otherwise it is reported as no detectable improvement, and
any class-level pattern as exploratory.

## What is not allowed

- Choosing the framing on any criterion other than the two conditions above.
- Selecting seeds, arms, metrics or intervals after seeing the 3-seed results.
- Comparing across the two recipe families (HSV family vs token family).

## Outcome

Recorded after running `07_paired_bootstrap.py --auto --n-boot 10000` on all 24 runs
(outputs `results/v2/tables/bootstrap.csv`, `bootstrap_summary.csv`).

| Condition | Result | Met? |
|---|---|---|
| TLA vs C1 favours TLA on ≥ 2 of 3 seeds | 0 of 3 (Δ = +0.0008, −0.0015, −0.0081; no interval excludes 0) | **No** |
| Three-seed mean TLA ≥ C2 | TLA 0.9511 < C2 0.9528; TLA below C2 on all three seeds | **No** |
| HSV: R_k vs R0 favours R_k on ≥ 2 of 3 seeds | R1 0/3, R2 0/3, R3 0/3 | **No** |

**Decision: Framing B.** HSV variants are reported as giving no detectable improvement
over R0. None of the 24 predeclared per-seed intervals excludes zero.

Exploratory checks made alongside, which bound what the manuscript may claim:

- On seed 42 the per-class gate correlated negatively with per-class ΔF1 (r ≈ −0.66 to
  −0.68 for R2/R3). **This did not replicate**: on seed 1337 it is positive for all
  three HSV arms. It must not be reported as a finding.
- What does replicate: the widest-opening gate falls on Tea algal leaf spot or Brown
  Blight in 8 of 9 HSV arm-seed combinations; no HSV variant improves Brown Blight on the
  three-seed mean.
- 8 test images are misclassified by all 24 runs, 7 of them with the identical wrong
  label in all 24; all 8 are in the expert-reviewed set, whose reviewer confirmed none of
  their dataset labels.
