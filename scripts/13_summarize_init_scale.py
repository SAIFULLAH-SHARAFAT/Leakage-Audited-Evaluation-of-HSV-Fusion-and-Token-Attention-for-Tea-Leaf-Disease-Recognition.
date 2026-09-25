#!/usr/bin/env python3
"""Aggregate pre-optimization HSV initialization-scale diagnostics for the current campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from campaign import location, matrix, require_campaign

METRICS = [
    "rgb_feature_rms",
    "hsv_embedding_rms",
    "hsv_projected_rms",
    "hsv_projected_to_rgb_rms",
    "gate_mean",
    "gate_std",
    "gate_min",
    "gate_max",
    "full_gate_delta_rms",
    "full_gate_delta_to_rgb_rms",
    "first_epoch_gate_alpha",
    "first_epoch_delta_rms",
    "first_epoch_delta_to_rgb_rms",
    "feature_cosine_mean",
    "full_gate_logit_delta_rms",
]


def expected_runs() -> list[tuple[str, int]]:
    cfg = matrix()
    out: list[tuple[str, int]] = []
    for exp, spec in cfg["experiments"].items():
        # Current HSV diagnostics are meaningful only for non-baseline HSV arms.
        if spec.get("family") != "hsv" or spec.get("role") == "baseline":
            continue
        seeds = spec.get("seeds", cfg["defaults"]["seeds"])
        out.extend((exp, int(s)) for s in seeds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-missing", action="store_true",
                    help="Write partial summaries instead of failing on missing planned diagnostics")
    args = ap.parse_args()

    runs = location("results_dir")
    out_dir = location("tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for exp, seed in expected_runs():
        run = runs / f"{exp}_s{seed}"
        diag_path = run / "metrics" / "init_scale_diagnostics.json"
        if not diag_path.exists():
            missing.append(str(diag_path))
            continue
        cfg = require_campaign(run)
        d = json.loads(diag_path.read_text())
        if d.get("measured_before_optimizer_updates") is not True:
            raise ValueError(f"{diag_path}: diagnostic was not marked pre-optimization")
        if d.get("data_split") != "validation":
            raise ValueError(f"{diag_path}: expected validation split diagnostic")
        if int(d.get("seed", -1)) != seed:
            raise ValueError(f"{diag_path}: seed mismatch")
        row = {
            "campaign_id": cfg["campaign_id"],
            "experiment": exp,
            "seed": seed,
            "hsv_use_sincos": bool(d.get("hsv_use_sincos")),
            "gate_vector": bool(d.get("gate_vector")),
            "n_images": int(d.get("n_images", 0)),
        }
        for key in METRICS:
            if key not in d:
                raise ValueError(f"{diag_path}: missing field {key}")
            row[key] = float(d[key])
        rows.append(row)

    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing planned initialization diagnostics:\n  " + "\n  ".join(missing)
        )
    if not rows:
        raise SystemExit("No initialization-scale diagnostics found")

    per_run = pd.DataFrame(rows).sort_values(["experiment", "seed"]).reset_index(drop=True)
    per_run_path = out_dir / "init_scale_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    agg = per_run.groupby("experiment")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    summary_path = out_dir / "init_scale_summary.csv"
    agg.to_csv(summary_path, index=False)

    print(per_run.round(6).to_string(index=False))
    print("\nSUMMARY")
    print(agg.round(6).to_string(index=False))
    if missing:
        print(f"\nWARNING: partial summary; {len(missing)} planned diagnostic(s) missing")
    print(f"\nwrote {per_run_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
