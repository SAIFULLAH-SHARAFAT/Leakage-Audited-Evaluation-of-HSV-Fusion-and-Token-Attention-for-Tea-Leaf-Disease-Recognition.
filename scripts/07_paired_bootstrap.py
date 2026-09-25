#!/usr/bin/env python3
"""
07_paired_bootstrap.py
======================

Paired bootstrap comparison between two runs on the same test set.

Procedure, as reported in the manuscript:
    B                 10,000 resamples (configurable)
    resampling unit   the individual test image
    pairing           the same index vector is applied to both models within
                      each replicate, so the difference is paired
    metric            Macro-F1 recomputed from scratch on each replicate
    interval          percentile, 2.5th to 97.5th
    stratification    none

Resampling is unstratified, so class representation varies across replicates.
No claim is made that these intervals are uniformly more or less conservative
than a stratified bootstrap; they are reported as pointwise descriptive
uncertainty for the finite test sample.

The bootstrap quantifies test-sample uncertainty for a FIXED pair of
checkpoints. It does not estimate training-seed variance; for that, compare the
across-seed standard deviations in the configured tables_dir/summary.csv.

USAGE
-----
    # one comparison
    python scripts/07_paired_bootstrap.py --a results/v2/runs/TLA_s42 \\
                                          --b results/v2/runs/C1_s42

    # every seed of one experiment against its family baseline
    python scripts/07_paired_bootstrap.py --experiment TLA --baseline C1

    # all non-baseline experiments against their family baselines
    python scripts/07_paired_bootstrap.py --auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from campaign import location, matrix, require_campaign
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]


def load_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_campaign(run_dir)
    raw=run_dir/"raw_outputs"
    y=np.load(raw/"test_targets.npy")
    p=np.load(raw/"test_predictions.npy")
    ids_path=raw/"test_ids.json"
    if not ids_path.exists():
        raise SystemExit(
            f"{ids_path} missing. Current campaign runs must save image IDs; "
            "label-vector equality alone is insufficient for paired resampling."
        )
    ids=np.asarray(json.load(open(ids_path)),dtype=object)
    if len(ids)!=len(y) or len(set(ids.tolist()))!=len(ids):
        raise SystemExit(f"Invalid test IDs in {ids_path}")
    return ids,y,p


LABELS = np.arange(7)

def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    # Fixed label set keeps the Macro-F1 denominator identical in every
    # bootstrap replicate, even if a class is absent by chance.
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def paired_bootstrap(dir_a: Path, dir_b: Path, n_boot: int, seed: int,
                     metric: str = "macro_f1") -> Dict:
    ida, ya, pa = load_run(dir_a)
    idb, yb, pb = load_run(dir_b)
    if set(ida.tolist()) != set(idb.tolist()):
        raise SystemExit(f"{dir_a.name} and {dir_b.name} do not contain the same test image IDs")
    # Reorder B to A's exact image order before drawing paired indices.
    pos={sid:i for i,sid in enumerate(idb.tolist())}
    order=np.asarray([pos[sid] for sid in ida.tolist()],dtype=int)
    yb,pb=yb[order],pb[order]
    if not np.array_equal(ya,yb):
        raise SystemExit("Ground-truth labels disagree after image-ID alignment")

    fn = macro_f1 if metric == "macro_f1" else (
        lambda y, p: float(accuracy_score(y, p)))
    n = len(ya)
    observed = fn(ya, pa) - fn(yb, pb)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = fn(ya[idx], pa[idx]) - fn(yb[idx], pb[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "a": dir_a.name, "b": dir_b.name, "metric": metric,
        "observed_delta": observed,
        "ci_low": float(lo), "ci_high": float(hi),
        "bootstrap_fraction_delta_positive": float((deltas > 0).mean()),
        "pointwise_ci_excludes_zero": bool(lo > 0 or hi < 0),
        "interpretation": "descriptive pointwise interval; not multiplicity-adjusted; fraction is not a p-value",
        "n_boot": n_boot, "n_test": int(n),
        "resampling_unit": "test_image",
        "ci_type": "percentile",
        "stratified": False,
        "rng_seed": seed,
    }


def family_baselines() -> Dict[str, str]:
    cfg = matrix()
    base = {v["family"]: k for k, v in cfg["experiments"].items()
            if v.get("role") == "baseline" and "family" in v}
    return {k: base.get(v.get("family"), "")
            for k, v in cfg["experiments"].items()
            if v.get("role") != "baseline"}


def runs_for(runs_dir: Path, exp_id: str) -> List[Path]:
    return sorted(d for d in runs_dir.iterdir()
                  if d.name.rsplit("_s", 1)[0] == exp_id
                  and (d / "raw_outputs" / "test_targets.npy").exists()
                  and (d / "raw_outputs" / "test_ids.json").exists())


def seed_of(d: Path) -> Optional[str]:
    parts = d.name.rsplit("_s", 1)
    return parts[1] if len(parts) == 2 else None


def compare_experiment(runs_dir: Path, exp: str, baseline: str,
                       n_boot: int, seed: int) -> List[Dict]:
    a_runs = {seed_of(d): d for d in runs_for(runs_dir, exp)}
    b_runs = {seed_of(d): d for d in runs_for(runs_dir, baseline)}
    shared = sorted(set(a_runs) & set(b_runs))
    if not shared:
        print(f"  no shared seeds between {exp} and {baseline}; skipped")
        return []
    out = []
    for s in shared:
        r = paired_bootstrap(a_runs[s], b_runs[s], n_boot, seed)
        r["experiment"], r["baseline"], r["seed"] = exp, baseline, s
        out.append(r)
        star = "  CI excludes 0" if r["pointwise_ci_excludes_zero"] else ""
        print(f"  seed {s}: delta {r['observed_delta']:+.4f}  "
              f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
              f"Bootstrap fraction >0 {r['bootstrap_fraction_delta_positive']:.3f}{star}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(location("results_dir")))
    ap.add_argument("--out", default=str(location("tables_dir") / "bootstrap.csv"))
    ap.add_argument("--a", help="Run directory A")
    ap.add_argument("--b", help="Run directory B")
    ap.add_argument("--experiment", help="Experiment ID for A")
    ap.add_argument("--baseline", help="Experiment ID for B")
    ap.add_argument("--auto", action="store_true",
                    help="Compare every experiment against its family baseline")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    results: List[Dict] = []

    if args.a and args.b:
        r = paired_bootstrap(Path(args.a), Path(args.b), args.n_boot, args.seed)
        print(json.dumps(r, indent=2))
        results.append(r)
    elif args.experiment and args.baseline:
        print(f"{args.experiment} vs {args.baseline}")
        results += compare_experiment(runs_dir, args.experiment, args.baseline,
                                      args.n_boot, args.seed)
    elif args.auto:
        planned = matrix()["paired_comparisons"]

        if planned:
            for item in planned:
                exp, base = item["a"], item["b"]
                print(f"\n{exp} vs {base}  --  {item.get('purpose','')}")
                results += compare_experiment(
                    runs_dir, exp, base, args.n_boot, args.seed
                )
        else:
            for exp, base in family_baselines().items():
                if not base:
                    continue
                print(f"\n{exp} vs {base}")
                results += compare_experiment(runs_dir, exp, base,
                                              args.n_boot, args.seed)
    else:
        raise SystemExit("choose --a/--b, --experiment/--baseline, or --auto")

    if not results:
        return
    df = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if "experiment" in df.columns:
        print("\nSUMMARY ACROSS SEEDS")
        g = df.groupby(["experiment", "baseline"])
        summ = pd.DataFrame({
            "n_seeds": g.size(),
            "mean_delta": g["observed_delta"].mean(),
            "sd_delta": g["observed_delta"].std(ddof=1),
            "positive_seeds": g["observed_delta"].apply(lambda s: int((s > 0).sum())),
            "seeds_pointwise_ci_excludes_0": g["pointwise_ci_excludes_zero"].sum(),
        }).reset_index()
        print(summ.round(4).to_string(index=False))
        summ.to_csv(Path(args.out).with_name("bootstrap_summary.csv"), index=False)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
