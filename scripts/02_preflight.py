#!/usr/bin/env python3
"""Fail-loud preflight before any v2 GPU training is launched."""
from __future__ import annotations

import importlib
import py_compile
import subprocess
import sys
from pathlib import Path

from campaign import matrix

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    needed = [
        "torch", "torchvision", "timm", "numpy", "pandas", "sklearn",
        "scipy", "yaml", "PIL", "tqdm", "matplotlib", "imagehash",
        "huggingface_hub",
    ]
    failed = []
    for name in needed:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
    if failed:
        raise SystemExit("Missing/broken dependencies: " + repr(failed))

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Refusing to launch the GPU campaign.")
    print(f"[OK] CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")

    # Compile current-campaign source and top-level scripts. Future-design scripts
    # are intentionally outside this campaign's executable contract.
    current_py = sorted((REPO / "src").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    for p in current_py:
        py_compile.compile(str(p), doraise=True)
    print(f"[OK] syntax: {len(current_py)} Python files compiled")

    cfg = matrix()
    defaults = cfg["defaults"]
    if defaults["campaign_id"] == "tea_leaf_fresh_v1":
        raise SystemExit("Old v1 campaign ID detected; final runs must use the new campaign")
    results_dir = (REPO / defaults["results_dir"]).resolve()
    protected = (REPO / "results" / "fresh").resolve()
    if results_dir == protected or protected in results_dir.parents:
        raise SystemExit("Configured results_dir points into preserved results/fresh v1 outputs")

    exps = cfg["experiments"]
    seeds = defaults["seeds"]
    jobs = sum(len(spec.get("seeds", seeds)) for spec in exps.values())
    if set(exps) != {"R0", "R1", "R2", "R3", "C1", "TLA", "C2", "C4"}:
        raise SystemExit(f"Unexpected experiment matrix: {list(exps)}")
    if jobs != 24:
        raise SystemExit(f"Expected 24 planned training jobs, found {jobs}")
    print(f"[OK] experiment matrix: {len(exps)} models x 3 seeds = {jobs} runs")

    for required in (
        "01_verify_frozen_dataset.py", "04_validate_model_contracts.py",
        "05_run_experiments.py", "13_summarize_init_scale.py",
        "16_audit_derivative_perceptual_leakage.py",
    ):
        if not (REPO / "scripts" / required).exists():
            raise SystemExit(f"Required current-campaign script missing: scripts/{required}")

    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "01_verify_frozen_dataset.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "05_run_experiments.py"), "--list"],
        check=True,
    )
    # --skip-done is required once any run has completed: the launcher treats a
    # finished run directory as immutable and exits non-zero without it. It also
    # re-verifies the recorded parameter budget of every completed run, so the
    # dry run checks both the pending command lines and the finished budgets.
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "05_run_experiments.py"),
         "--all", "--dry-run", "--skip-done"],
        check=True,
    )
    print("PASS: preflight complete. No training was started.")


if __name__ == "__main__":
    main()
