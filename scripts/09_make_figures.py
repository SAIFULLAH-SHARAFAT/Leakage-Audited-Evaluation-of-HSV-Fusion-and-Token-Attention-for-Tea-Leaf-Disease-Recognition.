#!/usr/bin/env python3
"""
09_make_figures.py
==================

Generate every figure from the result tables, so figures cannot drift out of
sync with the numbers they depict.

Figures produced (only those whose input tables exist):

    fig_dataset_distribution   class counts per partition
    fig_augmentation_inventory  derivatives per class and operation
    fig_main_results            Macro-F1 per experiment with seed spread
    fig_robustness_groups       Macro-F1 per perturbation family
    fig_robustness_retention    unperturbed vs mean stress, per model
    fig_bootstrap_deltas        paired-bootstrap deltas with 95% intervals

USAGE
-----
    python scripts/09_make_figures.py
    python scripts/09_make_figures.py --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from campaign import location, matrix

REPO = Path(__file__).resolve().parents[1]
TABLES = location("tables_dir")
FIGS = location("figures_dir")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def save(fig, name: str, fmt: str, dpi: int) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{name}.{fmt}"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print("wrote", out)


def read(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


# =============================================================================
# FIGURES
# =============================================================================

def fig_dataset(manifest: pd.DataFrame, fmt: str, dpi: int) -> None:
    counts = (manifest.groupby(["class_name", "split"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["train", "val", "test"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(counts))
    w = 0.27
    for k, (col, color) in enumerate(zip(["train", "val", "test"],
                                         ["#3b6ea5", "#8ab17d", "#e07a5f"])):
        ax.bar(idx + (k - 1) * w, counts[col], w, label=col, color=color)
    ax.set_xticks(idx)
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("Class distribution by partition")
    ax.legend(frameon=False)
    save(fig, "fig_dataset_distribution", fmt, dpi)


def fig_augmentation(manifest: pd.DataFrame, fmt: str, dpi: int) -> None:
    import re
    tr = manifest[manifest.split == "train"].copy()
    tr["op"] = (tr.filename.str.extract(r"_aug[_-]?([a-z]+)", flags=re.I)[0]
                .fillna("original").str.lower())
    piv = (tr.groupby(["class_name", "op"]).size().unstack(fill_value=0))
    ops = [c for c in ["flip", "zoom", "rotation", "brightness", "contrast", "blur"]
           if c in piv.columns]
    if not ops:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(piv))
    cmap = plt.get_cmap("Blues")
    for i, op in enumerate(ops):
        ax.bar(piv.index, piv[op], bottom=bottom, label=op,
               color=cmap(0.35 + 0.1 * i))
        bottom += piv[op].to_numpy()
    ax.set_xticks(range(len(piv.index)))
    ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylabel("Derived images")
    ax.set_title("Offline augmentation inventory (training partition)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, "fig_augmentation_inventory", fmt, dpi)


def fig_main(per_run: pd.DataFrame, fmt: str, dpi: int) -> None:
    # Do not visually rank experiments trained under different recipes.
    experiments = matrix()["experiments"]
    family_map = {family: [k for k,v in experiments.items() if v["family"] == family]
                  for family in dict.fromkeys(v["family"] for v in experiments.values())}
    for family, wanted in family_map.items():
        sub = per_run[per_run.experiment.isin(wanted)].copy()
        if sub.empty:
            continue
        present = [x for x in wanted if x in set(sub.experiment)]
        fig, ax = plt.subplots(figsize=(7, max(3.0, 0.55 * len(present) + 1.4)))
        for i, exp in enumerate(present):
            vals = sub.loc[sub.experiment == exp, "macro_f1"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.errorbar(vals.mean(), i, xerr=sd, fmt="o", capsize=3, markersize=6)
            ax.scatter(vals, np.full(len(vals), i, dtype=float), s=14, zorder=1)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present)
        ax.set_xlabel("Macro-F1")
        ax.set_title(f"{family.capitalize()} family: test Macro-F1 (mean ± sd)")
        save(fig, f"fig_main_results_{family}", fmt, dpi)


def fig_robustness_groups(grouped: pd.DataFrame, fmt: str, dpi: int) -> None:
    piv = grouped.pivot(index="group", columns="experiment", values="mean")
    order = [g for g in ["unperturbed", "brightness/contrast", "saturation",
                         "hue shift", "gaussian noise", "blur",
                         "jpeg compression", "shadow", "white balance"]
             if g in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(piv))
    n = len(piv.columns)
    w = 0.8 / max(n, 1)
    cmap = plt.get_cmap("tab10")
    for k, col in enumerate(piv.columns):
        ax.bar(idx + (k - (n - 1) / 2) * w, piv[col], w, label=col, color=cmap(k))
    ax.set_xticks(idx)
    ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(bottom=max(0.0, float(np.nanmin(piv.to_numpy())) - 0.05))
    ax.set_title("Macro-F1 by controlled perturbation family")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fig_robustness_groups", fmt, dpi)


def fig_retention(summary: pd.DataFrame, fmt: str, dpi: int) -> None:
    g = summary.groupby("experiment")[["unperturbed", "mean_stress"]].mean()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(g["unperturbed"], g["mean_stress"], s=55, color="#3b6ea5")
    for name, r in g.iterrows():
        ax.annotate(name, (r["unperturbed"], r["mean_stress"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    lo = float(min(g.min())) - 0.02
    hi = float(max(g.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Unperturbed Macro-F1")
    ax.set_ylabel("Mean stress Macro-F1")
    ax.set_title("Unperturbed vs corrupted performance")
    save(fig, "fig_robustness_retention", fmt, dpi)


def fig_bootstrap(boot: pd.DataFrame, fmt: str, dpi: int) -> None:
    boot = boot.copy()
    boot["label"] = boot["experiment"] + " vs " + boot["baseline"] + \
                    " (s" + boot["seed"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(boot) + 1.6))
    y = np.arange(len(boot))
    ax.errorbar(boot["observed_delta"], y,
                xerr=[boot["observed_delta"] - boot["ci_low"],
                      boot["ci_high"] - boot["observed_delta"]],
                fmt="o", color="#3b6ea5", capsize=3, markersize=5)
    ax.axvline(0, color="grey", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(boot["label"], fontsize=8)
    ax.set_xlabel("$\\Delta$ Macro-F1")
    ax.set_title("Paired bootstrap, 95% percentile intervals")
    save(fig, "fig_bootstrap_deltas", fmt, dpi)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--manifest",
                    default=str(REPO / "data" / "manifests" / "final_split_manifest.csv"))
    args = ap.parse_args()
    fmt, dpi = args.format, args.dpi

    made = 0
    manifest = read(Path(args.manifest))
    if manifest is not None:
        fig_dataset(manifest, fmt, dpi); made += 1
        fig_augmentation(manifest, fmt, dpi); made += 1
    else:
        print(f"[skip] manifest not found at {args.manifest}")

    per_run = read(TABLES / "per_run.csv")
    if per_run is not None:
        fig_main(per_run, fmt, dpi); made += 1
    else:
        print("[skip] run scripts/06_aggregate_results.py first")

    grouped = read(TABLES / "robustness_grouped.csv")
    if grouped is not None:
        fig_robustness_groups(grouped, fmt, dpi); made += 1

    rsum = read(TABLES / "robustness_summary.csv")
    if rsum is not None:
        fig_retention(rsum, fmt, dpi); made += 1

    boot = read(TABLES / "bootstrap.csv")
    if boot is not None and "experiment" in boot.columns:
        fig_bootstrap(boot, fmt, dpi); made += 1

    print(f"\n{made} figure(s) written to {FIGS}")


if __name__ == "__main__":
    main()
