"""Single source of paths and declared analysis comparisons."""
import json
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]


def matrix():
    with (REPO / "configs/experiments.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg.get("experiments"):
        raise ValueError("Experiment matrix is missing or empty")
    if not cfg.get("paired_comparisons"):
        raise ValueError("Predeclared paired_comparisons must not be empty")
    return cfg


def location(key):
    return REPO / matrix()["defaults"][key]


def require_campaign(run):
    run = Path(run)
    cfg = json.loads((run / "config/config.json").read_text())
    expected = matrix()["defaults"]["campaign_id"]
    if cfg.get("campaign_id") != expected:
        raise ValueError(f"{run.name}: cannot mix campaign {cfg.get('campaign_id')} with {expected}")
    return cfg
