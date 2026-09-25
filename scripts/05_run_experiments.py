#!/usr/bin/env python3
"""
05_run_experiments.py
=====================

Run experiments declared in configs/experiments.yaml.

Every run writes under the results_dir declared in configs/experiments.yaml.
Each run contains config, logs, metrics, raw predictions and model artifacts, so downstream
aggregation never depends on anything typed by hand.

USAGE
-----
    # what would run, without running it
    python scripts/05_run_experiments.py --list
    python scripts/05_run_experiments.py --only C1 --dry-run

    # a single run
    python scripts/05_run_experiments.py --only C1 --seeds 42

    # a family
    python scripts/05_run_experiments.py --family token

    # everything not already finished
    python scripts/05_run_experiments.py --all --skip-done

NOTES
-----
* --skip-done treats a run as finished only when a valid COMPLETE.json and all
  required artifact files exist.
* Session-limited environments (Kaggle, Colab) should invoke one or two runs
  per session; --auto_resume in the training scripts recovers an interrupted
  run from its checkpoint directory.
* Parameter-count expectations in the YAML are checked after each run. A
  mismatch means the architecture was not built as declared, which is the
  failure mode a silently ignored flag produces.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required:  pip install pyyaml")

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("experiments") or not cfg.get("paired_comparisons"):
        raise ValueError("Executable campaign needs experiments and paired_comparisons")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def completed_run(run_dir: Path, expected_campaign: str) -> bool:
    marker = run_dir / "COMPLETE.json"
    if not marker.exists():
        return False

    required = [
        run_dir / "config" / "config.json",
        run_dir / "config" / "run_provenance.json",
        run_dir / "metrics" / "test_results.json",
        run_dir / "metrics" / "per_class_metrics.json",
        run_dir / "raw_outputs" / "test_predictions.npy",
        run_dir / "raw_outputs" / "test_targets.npy",
        run_dir / "raw_outputs" / "test_probabilities.npy",
        run_dir / "raw_outputs" / "test_ids.json",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"{run_dir.name}: COMPLETE.json exists but required artifacts are missing:\n"
            + "\n".join(missing)
        )

    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise RuntimeError(f"{run_dir.name}: invalid completion marker status: {meta.get('status')}")

    if meta.get("campaign_id") != expected_campaign:
        raise RuntimeError(
            f"{run_dir.name}: completion marker campaign "
            f"'{meta.get('campaign_id')}' != '{expected_campaign}'"
        )

    return True


def verify_dataset_fingerprint(cfg: dict) -> None:
    """Verify the immutable manifest artifact and frozen dataset metadata."""
    defaults = cfg["defaults"]
    expected = defaults.get("expected_manifest_sha256")

    fp = REPO / defaults["manifest_fingerprint"]
    mp = REPO / defaults["manifest_csv"]

    if not fp.exists() or not mp.exists():
        raise SystemExit(
            "Frozen dataset manifests missing. Run scripts/00 and 01 first."
        )

    meta = json.load(open(fp))

    # Hash the frozen manifest artifact directly. Re-serializing through pandas
    # is intentionally avoided because CSV output can vary across pandas versions.
    got = _sha256_file(mp)

    if got != expected or meta.get("manifest_sha256") != expected:
        raise SystemExit(
            "DATASET MANIFEST LOCK MISMATCH. Refusing to launch experiments.\n"
            f" expected: {expected}\n"
            f" raw manifest: {got}\n"
            f" json: {meta.get('manifest_sha256')}"
        )

    got_counts = (
        meta.get("n_train"),
        meta.get("n_val"),
        meta.get("n_test"),
        meta.get("n_total"),
    )

    if got_counts != (6090, 816, 808, 7714):
        raise SystemExit(
            f"Dataset count fingerprint mismatch: {got_counts}"
        )

    print(f"Dataset manifest lock verified: {got}")


def write_run_provenance(cfg: dict, exp_id: str, seed: int, run_name: str, cmd: List[str]) -> None:
    """Attach the exact command, hashes, dataset lock, and HF snapshot ID to each run."""
    import datetime
    import platform
    defaults = cfg["defaults"]
    run_dir = REPO / defaults["results_dir"] / run_name
    cdir = run_dir / "config"
    cdir.mkdir(parents=True, exist_ok=True)
    exp = cfg["experiments"][exp_id]
    recipe = cfg["recipes"][exp["recipe"]]
    train_script = REPO / recipe["script"]
    sources = {
        "train_script": {"path": str(train_script.relative_to(REPO)), "sha256": _sha256_file(train_script)},
        "experiment_yaml": {"path": str(Path(cfg.get("_config_path", CONFIG)).relative_to(REPO)), "sha256": _sha256_file(Path(cfg.get("_config_path", CONFIG)))},
    }
    shared = REPO / "src/train_swin_three_models.py"
    if shared.exists():
        sources["shared_swin_script"] = {"path": "src/train_swin_three_models.py", "sha256": _sha256_file(shared)}
    helper = REPO / "src/training_contracts.py"
    sources["training_contracts"] = {"path": "src/training_contracts.py", "sha256": _sha256_file(helper)}
    hf_lock = REPO / "data" / "hf_dataset_lock.json"
    provenance = {
        "campaign_id": defaults.get("campaign_id"),
        "experiment": exp_id,
        "seed": seed,
        "run_name": run_name,
        "command": cmd,
        "dataset_manifest_sha256": defaults.get("expected_manifest_sha256"),
        "hf_dataset_lock": json.load(open(hf_lock)) if hf_lock.exists() else {
            "status": "local_manifest_verified",
            "resolved_commit_sha": None,
            "manifest_sha256": _sha256_file(REPO / defaults["manifest_csv"]),
            "note": "Local dataset; historical remote commit not inferred"
        },
        "sources": sources,
        "python": sys.version,
        "platform": platform.platform(),
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    dest = cdir / "run_provenance.json"
    if dest.exists():
        original = json.loads(dest.read_text())
        for key in ("campaign_id", "sources", "command", "dataset_manifest_sha256", "hf_dataset_lock"):
            if original.get(key) != provenance.get(key):
                raise RuntimeError(f"Resume provenance changed: {key}. Use a new run directory.")
    else:
        dest.write_text(json.dumps(provenance, indent=2))
    with (cdir / "invocations.jsonl").open("a") as f:
        f.write(json.dumps({"recorded_utc": provenance["recorded_utc"], "command": cmd}) + "\n")


def flag_args(d: Dict) -> List[str]:
    """Turn a dict into CLI flags. True -> bare flag, False -> omitted."""
    out: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def build_command(cfg: dict, exp_id: str, seed: int) -> tuple[str, List[str]]:
    defaults = cfg["defaults"]
    exp = cfg["experiments"][exp_id]
    recipe = cfg["recipes"][exp["recipe"]]

    run_name = f"{exp_id}_s{seed}"
    run_dir = REPO / defaults["results_dir"]
    ckpt_dir = REPO / defaults["ckpt_dir"] / run_name
    protected = (REPO / "results/fresh").resolve()
    if run_dir.resolve().is_relative_to(protected) or ckpt_dir.resolve().is_relative_to(protected):
        raise RuntimeError("results/fresh is the preserved v1 campaign; use a new campaign path")

    args: Dict = {}
    args.update(recipe["args"])
    args.update(exp.get("args", {}))

    cmd = [
        sys.executable, str(REPO / recipe["script"]),
        "--exp_name", run_name,
        "--run_dir", str(run_dir),
        "--data_root", str(REPO / defaults["data_root"]),
        "--ckpt_temp_dir", str(ckpt_dir),
        "--seed", str(seed),
        "--campaign_id", defaults["campaign_id"],
        "--auto_resume"
    ]
    cmd += flag_args(args)
    return run_name, cmd


def check_params(run_dir: Path, exp: dict, exp_id: str) -> None:
    mp = run_dir / "metrics" / "test_results.json"
    if not mp.exists():
        return
    m = json.load(open(mp))

    # Optional total-model guard.
    got_total = m.get("params_m")
    want_total = exp.get("expect_params_m")
    if want_total is not None and got_total is not None:
        if abs(float(got_total) - float(want_total)) > 0.05:
            raise SystemExit(
                f"TOTAL PARAMETER MISMATCH for {exp_id}: built {got_total}M, "
                f"declared {want_total}M. Refusing to continue."
            )
        print(f"   total parameter count verified: {got_total}M")

    # Reviewer controls use the ACTIVE token-module parameter budget.
    want_token = exp.get("expect_token_params")
    if want_token is not None:
        got_token = m.get("token_module_params")
        if got_token is None:
            raise SystemExit(
                f"Missing token_module_params in {mp} for {exp_id}. "
                "Refusing to accept an unverifiable control run."
            )
        if int(got_token) != int(want_token):
            raise SystemExit(
                f"TOKEN PARAMETER MISMATCH for {exp_id}: built {got_token:,}, "
                f"expected {int(want_token):,}. Refusing to continue."
            )
        print(f"   active token-module budget verified: {int(got_token):,}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--only", nargs="+", help="Experiment IDs, e.g. C1 C2")
    ap.add_argument("--family", help="Run a whole family (hsv | token)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", help="Override declared seeds")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    verify_dataset_fingerprint(cfg)
    exps = cfg["experiments"]

    if args.list:
        print(f"{'ID':<6} {'family':<7} {'recipe':<6} {'seeds':<18} label")
        for k, v in exps.items():
            seeds = v.get("seeds", cfg["defaults"]["seeds"])
            print(
                f"{k:<6} {v.get('family',''):<7} {v['recipe']:<6} "
                f"{str(seeds):<18} {v['label']}"
            )
        return

    if args.only:
        selected = [e for e in args.only if e in exps]
        missing = [e for e in args.only if e not in exps]
        if missing:
            raise SystemExit(f"unknown experiment(s): {missing}")
    elif args.family:
        selected = [k for k, v in exps.items() if v.get("family") == args.family]
    elif args.all:
        selected = list(exps)
    else:
        raise SystemExit("choose --only, --family, --all or --list")

    jobs = []
    for exp_id in selected:
        seeds = args.seeds or exps[exp_id].get("seeds", cfg["defaults"]["seeds"])
        for s in seeds:
            jobs.append((exp_id, s))

    print(f"{len(jobs)} run(s) selected\n")
    for exp_id, seed in jobs:
        run_name, cmd = build_command(cfg, exp_id, seed)
        run_dir = REPO / cfg["defaults"]["results_dir"] / run_name

        if completed_run(run_dir, cfg["defaults"]["campaign_id"]):
            if args.skip_done:
                check_params(run_dir, exps[exp_id], exp_id)
                print(f"[skip] {run_name} already complete")
                continue
            raise SystemExit(
                f"Completed run is immutable: {run_dir}; use --skip-done"
            )

        print("=" * 78)
        print(f"RUN {run_name}  --  {exps[exp_id]['label']}")
        print("=" * 78)
        print(" ".join(cmd) + "\n")
        if args.dry_run:
            continue

        write_run_provenance(cfg, exp_id, seed, run_name, cmd)
        subprocess.run(cmd, check=True)
        check_params(run_dir, exps[exp_id], exp_id)


if __name__ == "__main__":
    main()