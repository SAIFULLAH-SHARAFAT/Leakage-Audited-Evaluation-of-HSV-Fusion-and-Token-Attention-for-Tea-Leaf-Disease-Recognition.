#!/usr/bin/env python3
"""Merge per-seed robustness outputs into the three-seed robustness tables.

08_eval_robustness.py writes robustness_summary.csv and robustness_grouped.csv next to
its --out file, so running one seed at a time into robustness_parts/ leaves only the
last seed's summary there. The per-condition files robustness_s<seed>.csv are complete,
so this script rebuilds everything from them, with the same formulas as 08:

  tables/robustness.csv          every run x condition
  tables/robustness_summary.csv  per run: unperturbed, mean/worst stress, drop, retention
  tables/robustness_grouped.csv  per experiment x perturbation group: mean, std
"""
from __future__ import annotations

import argparse

import pandas as pd

from campaign import location, matrix

TABLES = location("tables_dir")
PARTS = TABLES / "robustness_parts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge whatever seeds exist (for checking only; writes to robustness_parts/)")
    args = ap.parse_args()

    # robustness_s<digits>.csv only: robustness_summary.csv also matches "robustness_s*".
    files = sorted(f for f in PARTS.glob("robustness_s*.csv") if f.stem[len("robustness_s"):].isdigit())
    if not files:
        raise SystemExit(f"no robustness_s*.csv in {PARTS}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    cfg = matrix()["defaults"]
    expected = {f"{e}_s{s}" for e in cfg["robustness_experiments"] for s in cfg["robustness_seeds"]}
    runs = set(df["run"])
    n_cond = df.groupby("run")["perturbation"].nunique()
    if df.duplicated(["run", "perturbation"]).any():
        raise SystemExit("duplicate run x perturbation rows; a seed was written twice")
    if (n_cond != 19).any():
        raise SystemExit(f"runs without all 19 conditions: {sorted(n_cond[n_cond != 19].index)}")
    if runs != expected and not args.allow_partial:
        raise SystemExit(f"expected {len(expected)} runs, found {len(runs)}; "
                         f"missing {sorted(expected - runs)}")

    rows = []
    for run, g in df.groupby("run"):
        clean = float(g.loc[g.perturbation == "unperturbed", "macro_f1"].iloc[0])
        stress = g[g.perturbation != "unperturbed"]["macro_f1"]
        rows.append({"run": run, "experiment": run.rsplit("_s", 1)[0],
                     "seed": run.rsplit("_s", 1)[-1], "unperturbed": clean,
                     "mean_stress": float(stress.mean()), "worst_stress": float(stress.min()),
                     "worst_condition": g.loc[stress.idxmin(), "perturbation"],
                     "mean_drop": clean - float(stress.mean()),
                     "mean_retention": float((stress / clean).mean()),
                     # seeds 42 and 1337 predate these columns; both ran the strict check
                     "clean_contract": g["clean_contract"].iloc[0] if "clean_contract" in g and pd.notna(g["clean_contract"].iloc[0]) else "exact",
                     "device": g["device"].iloc[0] if "device" in g and pd.notna(g["device"].iloc[0]) else ("cpu" if run.endswith("_s42") else "cuda")})
    summary = pd.DataFrame(rows)
    grouped = df.groupby(["experiment", "group"])["macro_f1"].agg(["mean", "std"]).reset_index()

    out = PARTS / "merged_check" if args.allow_partial else TABLES
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    summary.to_csv(out / "robustness_summary.csv", index=False)
    grouped.to_csv(out / "robustness_grouped.csv", index=False)
    print(f"merged {len(runs)} runs from {len(files)} file(s) into {out}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
