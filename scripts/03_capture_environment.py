#!/usr/bin/env python3
"""Capture the exact software, hardware, Git, config and dataset state for the campaign."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from campaign import matrix, location

REPO = Path(__file__).resolve().parents[1]


def version(name: str) -> str:
    try:
        m = __import__(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"unavailable: {e}"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def main() -> None:
    import torch

    cfg = matrix()
    defaults = cfg["defaults"]
    config_path = REPO / "configs" / "experiments.yaml"
    manifest_path = REPO / defaults["manifest_csv"]

    info = {
        "campaign_id": defaults["campaign_id"],
        "python": sys.version,
        "platform": platform.platform(),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "timm": version("timm"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "sklearn": version("sklearn"),
        "scipy": version("scipy"),
        "pillow": version("PIL"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_sha": command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command(["git", "status", "--porcelain"]),
        "experiments_yaml_sha256": sha256(config_path),
        "manifest_sha256": sha256(manifest_path),
        "expected_manifest_sha256": defaults.get("expected_manifest_sha256"),
    }
    try:
        info["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except Exception as e:
        info["pip_freeze"] = []
        info["pip_freeze_error"] = str(e)

    out = location("results_dir").parent / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "pip_freeze"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
