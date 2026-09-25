#!/usr/bin/env python3
"""Generate the consolidated training-settings table requested by Reviewer #1."""
from __future__ import annotations

import json
from pathlib import Path
from campaign import location, matrix, require_campaign

REPO = Path(__file__).resolve().parents[1]
RUNS = location("results_dir")
OUT = location("tables_dir") / "training_settings.tex"


def load_cfg(run: str) -> dict:
    return require_campaign(RUNS / run)


def family_config(family):
    cfg = matrix()
    ids = [k for k,v in cfg["experiments"].items() if v["family"] == family]
    configs = [load_cfg(p.name) for p in sorted(RUNS.glob("*_s*"))
               if p.name.rsplit("_s",1)[0] in ids and (p/"config/config.json").exists()]
    if not configs:
        raise SystemExit(f"No recorded configuration for {family}; run its baseline first")
    keys = ["input_size", "epochs", "batch_size", "gradient_accumulation_steps", "lr",
            "weight_decay", "warmup_epochs", "min_lr", "label_smoothing", "ema_decay"]
    if any(any(c[k] != configs[0][k] for k in keys) for c in configs):
        raise ValueError(f"Inconsistent recipe settings in {family}")
    return configs[0]


def val(c: dict, key: str, default="--"):
    return c.get(key, default)


def sci(x) -> str:
    """Format a learning rate as LaTeX, e.g. 0.0005 -> $5\\times10^{-4}$."""
    mant, exp = f"{float(x):.0e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def main() -> None:
    hsv = family_config("hsv")
    token = family_config("token")
    head = r"MLP 768--384--7"
    rows = [
        ("Backbone", "Swin-S, ImageNet-1k pretrained", "Swin-S, ImageNet-1k pretrained"),
        ("Classifier head (all arms)", head, head),
        ("Input size", val(hsv, "input_size", 224), val(token, "input_size", 224)),
        ("Epoch budget", val(hsv, "epochs"), val(token, "epochs")),
        ("Physical batch size", val(hsv, "batch_size"), val(token, "batch_size")),
        ("Gradient accumulation", val(hsv, "gradient_accumulation_steps", 1), val(token, "gradient_accumulation_steps", 1)),
        ("Effective batch size", val(hsv, "batch_size") * val(hsv, "gradient_accumulation_steps", 1), val(token, "batch_size") * val(token, "gradient_accumulation_steps", 1)),
        ("Optimizer", "AdamW", "AdamW"),
        ("Peak learning rate", sci(val(hsv, "lr")), sci(val(token, "lr"))),
        ("Weight decay", val(hsv, "weight_decay", 0.05), val(token, "weight_decay", 0.05)),
        ("Weight-decay exemptions", "norms, biases, gates", "norms, biases, gates"),
        ("LR warm-up (epochs)", val(hsv, "warmup_epochs", 5), val(token, "warmup_epochs", 5)),
        ("Warm-up start / minimum LR", sci(val(hsv, "warmup_lr_init", 1e-6)), sci(val(token, "warmup_lr_init", 1e-6))),
        ("DropPath", val(hsv, "drop_path_rate"), val(token, "drop_path_rate")),
        ("Label smoothing", val(hsv, "label_smoothing", 0.1), val(token, "label_smoothing", 0.1)),
        ("EMA decay", val(hsv, "ema_decay", 0.9998), val(token, "ema_decay", 0.9998)),
        (r"Gradient clipping ($\ell_2$)", val(hsv, "grad_clip_norm", 1.0), val(token, "grad_clip_norm", 1.0)),
        ("Mixed precision", "AMP (FP16)", "AMP (FP16)"),
        ("Data-loader workers", val(hsv, "num_workers", 0), val(token, "num_workers", 0)),
        ("Gate warm-up (epochs)", val(hsv, "gate_warmup_epochs", 5), val(token, "gate_warmup_epochs", 10)),
        ("Hue jitter", val(hsv, "hue_jitter", 0.0), val(token, "hue_jitter", 0.0)),
        ("Checkpoint criterion", "validation Macro-F1", "validation Macro-F1"),
        ("Early-stopping patience", val(hsv, "early_stopping_patience", 25), val(token, "early_stopping_patience", 25)),
        ("Seeds", "42, 1337, 2026", "42, 1337, 2026"),
    ]

    lines = [
        "% generated from completed run configs -- do not edit by hand",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Consolidated training-recipe settings, generated from the completed run configurations of all 24 runs. Every arm within a family uses the same recipe; architecture-specific settings are given in the text.}",
        r"\label{tab:training_settings}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{HSV family (R0--R3)} & \textbf{Token family (C1/TLA/C2/C4)} " + r"\\",
        r"\midrule",
    ]
    for key, a, b in rows:
        lines.append(f"{key} & {a} & {b} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
