#!/usr/bin/env python3
"""Explicit safe entry points for v2 verification, staged training, analysis and export."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

S = Path(__file__).resolve().parent


def run(name: str, *args: str) -> None:
    print("\n$", sys.executable, str(S / name), *args)
    subprocess.run([sys.executable, str(S / name), *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true",
                    help="Dataset, derivative audit, model contracts and environment capture")
    ap.add_argument("--smoke", action="store_true",
                    help="Run only the two baseline harness checks: R0_s42 then C1_s42")
    ap.add_argument("--seed42", action="store_true",
                    help="Run the remaining six seed-42 models after smoke baselines pass")
    ap.add_argument("--remaining", action="store_true",
                    help="Run all models for seeds 1337 and 2026")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if not any(vars(a).values()):
        ap.print_help()
        return

    if a.prepare:
        run("01_verify_frozen_dataset.py", "--verify-md5")
        run("02_preflight.py")
        run("04_validate_model_contracts.py")
        run("16_audit_derivative_perceptual_leakage.py")
        run("03_capture_environment.py")

    if a.smoke:
        run("05_run_experiments.py", "--only", "R0", "--seeds", "42", "--skip-done")
        run("05_run_experiments.py", "--only", "C1", "--seeds", "42", "--skip-done")

    if a.seed42:
        run("05_run_experiments.py", "--only", "R1", "R2", "R3", "TLA", "C2", "C4",
            "--seeds", "42", "--skip-done")

    if a.remaining:
        run("05_run_experiments.py", "--all", "--seeds", "1337", "2026", "--skip-done")

    if a.analyze:
        run("06_aggregate_results.py")
        run("10_make_training_settings_table.py")
        run("13_summarize_init_scale.py")
        run("07_paired_bootstrap.py", "--auto", "--n-boot", "10000")
        run("08_eval_robustness.py", "--planned")
        run("09_make_figures.py", "--format", "pdf")

    if a.export:
        run("15_package_manifest.py")
        run("11_export_reproducibility_bundle.py")


if __name__ == "__main__":
    main()
