#!/usr/bin/env python3
"""
06_aggregate_results.py
=======================

Build every results table directly from the run artifacts.

No number in the paper is transcribed by hand. Each run directory is read for
its config and test metrics, and the per-seed and aggregated tables are written
as CSV and as LaTeX ready to \\input{} into the manuscript.

Emits under the tables_dir declared in configs/experiments.yaml:
    per_run.csv          one row per (experiment, seed)
    summary.csv          mean +/- sd over seeds
    per_class_f1.csv     per-class F1, mean +/- sd
    main_results.tex     main comparison table
    per_class_f1.tex     per-class table

USAGE
-----
    python scripts/06_aggregate_results.py
    python scripts/06_aggregate_results.py --family token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from campaign import matrix, require_campaign

REPO = Path(__file__).resolve().parents[1]

CLASSES = ["Brown Blight", "Gray Blight", "Green mirid bug", "Healthy leaf",
           "Helopeltis", "Red spider", "Tea algal leaf spot"]


def load_labels(config: dict) -> dict:
    return matrix()["experiments"]


def collect(runs_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    if not runs_dir.exists():
        raise SystemExit(f"no results directory at {runs_dir}")
    for run in sorted(runs_dir.iterdir()):
        mp = run / "metrics" / "test_results.json"
        cp = run / "config" / "config.json"
        if not (mp.exists() and cp.exists()):
            continue
        m, c = json.load(open(mp)), require_campaign(run)
        if m.get("campaign_id") != c["campaign_id"] or m.get("best_epoch") is None:
            raise ValueError(f"Incomplete or inconsistent selection metadata: {mp}")
        exp_id = run.name.rsplit("_s", 1)[0]
        row = {
            "experiment": exp_id,
            "run": run.name,
            "seed": c.get("seed"),
            "campaign_id": c["campaign_id"],
            "arch": c.get("arch", c.get("model_name", c.get("model", ""))),
            "use_hsv": c.get("use_hsv", c.get("use_hsv_branch", False)),
            "lr": c.get("lr"),
            "batch_size": c.get("batch_size"),
            "acc1": m.get("acc1"),
            "macro_f1": m.get("macro_f1"),
            "macro_auc": None if m.get("macro_auc") in ("N/A", None) else m.get("macro_auc"),
            "params_m": m.get("params_m"),
            "gflops": m.get("gflops"),
            "token_module_params": m.get("token_module_params"),
            "best_epoch": m.get("best_epoch"),
        }
        pc = run / "metrics" / "per_class_metrics.json"
        if pc.exists():
            d = json.load(open(pc))
            # Schema written by the training scripts:
            #   {"<class name>": {"f1": .., "precision": .., "recall": ..}, ...}
            if isinstance(d, dict) and d and all(isinstance(v, dict) for v in d.values()):
                for cname, m_ in d.items():
                    if "f1" in m_:
                        row[f"f1::{cname}"] = m_["f1"]
            # Tolerated alternatives
            elif isinstance(d, dict) and isinstance(d.get("f1"), dict):
                for k, v in d["f1"].items():
                    row[f"f1::{k}"] = v
            elif isinstance(d, dict) and isinstance(d.get("f1"), list) \
                    and len(d["f1"]) == len(CLASSES):
                for k, v in zip(CLASSES, d["f1"]):
                    row[f"f1::{k}"] = v
        rows.append(row)
    if not rows:
        raise SystemExit(f"no completed runs found under {runs_dir}")
    return pd.DataFrame(rows)


def config_order() -> list:
    return list(matrix()["experiments"])


def order_by_config(df: pd.DataFrame, col: str = "experiment") -> pd.DataFrame:
    order = config_order()
    if not order:
        return df
    rank = {k: i for i, k in enumerate(order)}
    return (df.assign(_r=df[col].map(lambda x: rank.get(x, 10_000)))
              .sort_values(["_r", col]).drop(columns="_r").reset_index(drop=True))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["acc1", "macro_f1", "macro_auc"]
    g = df.groupby("experiment")
    out = pd.DataFrame({
        "n_seeds": g["seed"].nunique(),
        "params_m": g["params_m"].first(),
        "gflops": g["gflops"].first(),
    })
    for m in metrics:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_std"] = g[m].std(ddof=1)
    return out.reset_index()


def per_class(df: pd.DataFrame, pct: bool = False) -> Optional[pd.DataFrame]:
    cols = [c for c in df.columns if c.startswith("f1::")]
    if not cols:
        return None
    g = df.groupby("experiment")[cols]
    mean, std = g.mean(), g.std(ddof=1)
    scale, nd = (100.0, 2) if pct else (1.0, 4)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        name = c.split("::", 1)[1]
        out[name] = [f"{m*scale:.{nd}f} ± {s*scale:.{nd}f}" if not np.isnan(s)
                     else f"{m*scale:.{nd}f}" for m, s in zip(mean[c], std[c])]
    return out.reset_index()


def tex(text: str) -> str:
    """Escape a config label for LaTeX and use the manuscript's terminology."""
    text = str(text).replace("sin-cos", "sin--cos")
    for ch, rep in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(ch, rep)
    return text


def fmt(mean: float, std: float, pct: bool = True, nd: int = 2) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    scale = 100.0 if pct else 1.0
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"${mean * scale:.{nd}f}$"
    return f"${mean * scale:.{nd}f} \\pm {std * scale:.{nd}f}$"


CAPTIONS = {
    "tab:token_results": (
        "Primary comparison (token family): test-set performance of the "
        "recipe-matched Swin-S control (C1), Stage-4 token-level attention (TLA), "
        "the parameter-matched non-attention control (C2), and the lightweight ECA "
        "comparator (C4). All four share one training recipe and classifier head. "
        "Mean $\\pm$ standard deviation over seeds 42, 1337 and 2026."),
    "tab:hsv_results": (
        "Complementary comparison (HSV family): test-set performance of the matched "
        "RGB baseline (R0) and three HSV fusion variants (R1--R3), which share one "
        "training recipe and classifier head. Mean $\\pm$ standard deviation over seeds "
        "42, 1337 and 2026. The two families use different recipes, so values are not "
        "comparable across Tables~\\ref{tab:token_results} and~\\ref{tab:hsv_results}."),
    "tab:all_results_supplement": (
        "All eight arms under the audited TeaLeafBD protocol, mean $\\pm$ standard "
        "deviation over three seeds. R0--R3 and C1/TLA/C2/C4 use different training "
        "recipes; inferential comparisons are made only within each family."),
}


def latex_main(summary: pd.DataFrame, labels: dict, out: Path,
               label: str = "tab:all_results_supplement") -> None:
    lines = [
        "% generated by scripts/06_aggregate_results.py -- do not edit by hand",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{CAPTIONS[label]}}}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{ID} & \\textbf{Configuration} & \\textbf{Seeds} & "
        "\\textbf{Acc@1 (\\%)} & \\textbf{Macro-F1 (\\%)} & "
        "\\textbf{Macro-AUC (\\%)} & \\textbf{Params / GFLOPs} \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        name = tex(labels.get(r["experiment"], {}).get("label", r["experiment"]))
        lines.append(
            f"{r['experiment']} & {name} & {int(r['n_seeds'])} & "
            f"{fmt(r['acc1_mean'], r['acc1_std'])} & "
            f"{fmt(r['macro_f1_mean'], r['macro_f1_std'])} & "
            f"{fmt(r['macro_auc_mean'], r['macro_auc_std'])} & "
            f"{r['params_m']:.2f}M / {r['gflops']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    out.write_text("\n".join(lines) + "\n")


def latex_per_class(pc: pd.DataFrame, labels: dict, out: Path) -> None:
    cls = [c for c in pc.columns if c != "experiment"]
    lines = [
        "% generated by scripts/06_aggregate_results.py -- do not edit by hand",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Per-class test F1 (\\%), mean $\\pm$ standard deviation over seeds "
        "42, 1337 and 2026. R0--R3 and C1/TLA/C2/C4 use different training recipes; "
        "compare arms within a family only.}",
        "\\label{tab:per_class_f1}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(pc) + "}",
        "\\toprule",
        "\\textbf{Class} & " + " & ".join(
            f"\\textbf{{{r['experiment']}}}" for _, r in pc.iterrows()) + " \\\\",
        "\\midrule",
    ]
    pm = " \\pm "
    for c in cls:
        vals = " & ".join("$" + str(pc.loc[i, c]).replace(" ± ", pm) + "$"
                          for i in pc.index)
        lines.append(f"{c} & {vals} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg0=matrix()
    ap.add_argument("--runs-dir", default=str(REPO / cfg0["defaults"]["results_dir"]))
    ap.add_argument("--out-dir", default=str(REPO / cfg0["defaults"]["tables_dir"]))
    ap.add_argument("--family", help="Restrict to one family from the config")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels({})

    df = collect(Path(args.runs_dir))
    if args.family:
        keep = {k for k, v in labels.items() if v.get("family") == args.family}
        df = df[df["experiment"].isin(keep)]

    df.to_csv(out / "per_run.csv", index=False)
    print("\nPER-RUN")
    print(df[["run", "seed", "arch", "lr", "acc1", "macro_f1",
              "params_m"]].round(4).to_string(index=False))

    summary = order_by_config(summarise(df))
    summary.to_csv(out / "summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.round(4).to_string(index=False))

    # Combined table is useful for the supplement only. Main-paper tables are
    # split by predeclared family so models trained under different recipes are
    # not visually ranked against each other.
    latex_main(summary, labels, out / "all_results_supplement.tex")
    print(f"\nwrote {out/'all_results_supplement.tex'}")
    for fam, name, label in [("token", "token_primary_results.tex", "tab:token_results"),
                             ("hsv", "hsv_complementary_results.tex", "tab:hsv_results")]:
        ids={k for k,v in labels.items() if v.get("family")==fam}
        sf=summary[summary["experiment"].isin(ids)].copy()
        if len(sf):
            latex_main(sf, labels, out/name, label)
            print(f"wrote {out/name}")

    pc = per_class(df)
    if pc is not None:
        pc = order_by_config(pc)
        pc.to_csv(out / "per_class_f1.csv", index=False)
        latex_per_class(order_by_config(per_class(df, pct=True)), labels,
                        out / "per_class_f1.tex")
        print(f"wrote {out/'per_class_f1.tex'}")
    else:
        print("per-class metrics not found in run artifacts; skipped")

    incomplete = summary[summary["n_seeds"] < 3]["experiment"].tolist()
    if incomplete:
        print(f"\nNOTE: fewer than 3 seeds for {incomplete}. "
              f"Label these as screening results in the manuscript.")


if __name__ == "__main__":
    main()
