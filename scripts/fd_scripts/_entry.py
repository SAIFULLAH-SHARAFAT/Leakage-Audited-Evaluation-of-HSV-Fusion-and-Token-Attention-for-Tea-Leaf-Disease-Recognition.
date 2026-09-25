"""Future designs are component prototypes, not an executable training campaign."""
import argparse
from pathlib import Path
import runpy
import sys
import yaml

REPO = Path(__file__).resolve().parents[2]


def main(name):
    # Dataset preparation is shared, never copied into an accidental scripts/data tree.
    if name in ("00_fetch_frozen_dataset.py", "01_verify_frozen_dataset.py"):
        sys.path.insert(0, str(REPO/"scripts"))
        runpy.run_path(str(REPO/"scripts"/name), run_name="__main__")
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO/"configs/future_designs.yaml").read_text())
    if args.list or args.dry_run:
        print("Future campaign: design-only; no training jobs launched or executable recipes declared")
        for key, value in cfg["future_campaign"]["designs"].items():
            print(key, value["label"])
        return
    if name in ("02_preflight.py", "04_validate_model_contracts.py") or args.prepare:
        runpy.run_path(str(Path(__file__).parent/"06_validate_future_components.py"), run_name="__main__")
        return
    raise SystemExit("Future training/analysis is not implemented. Use --list or --dry-run to inspect designs. "
                     "The current campaign is scripts/05_run_experiments.py; no fresh runs are routed here.")
