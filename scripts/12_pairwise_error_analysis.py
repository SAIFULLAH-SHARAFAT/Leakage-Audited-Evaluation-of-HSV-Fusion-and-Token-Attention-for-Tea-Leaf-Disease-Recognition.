#!/usr/bin/env python3
"""Image-ID-aligned paired error-overlap analysis for two completed campaign runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from campaign import matrix, require_campaign


def load(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    run = Path(run)
    cfg = require_campaign(run)
    raw = run / "raw_outputs"
    required = {
        "ids": raw / "test_ids.json",
        "targets": raw / "test_targets.npy",
        "predictions": raw / "test_predictions.npy",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run.name}: missing raw outputs: {missing}")

    ids = np.asarray(json.loads(required["ids"].read_text()), dtype=object)
    y = np.load(required["targets"])
    p = np.load(required["predictions"])
    if not (ids.ndim == y.ndim == p.ndim == 1):
        raise ValueError(f"{run.name}: IDs/targets/predictions must all be 1-D")
    if not (len(ids) == len(y) == len(p)):
        raise ValueError(f"{run.name}: raw output lengths disagree")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{run.name}: duplicate test image IDs")
    return ids, y, p, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ra, rb = Path(args.a), Path(args.b)
    ia, ya, pa, ca = load(ra)
    ib, yb, pb, cb = load(rb)

    if ca["campaign_id"] != cb["campaign_id"]:
        raise ValueError("Cannot compare runs from different campaigns")
    if set(ia.tolist()) != set(ib.tolist()):
        only_a = sorted(set(ia.tolist()) - set(ib.tolist()))[:5]
        only_b = sorted(set(ib.tolist()) - set(ia.tolist()))[:5]
        raise ValueError(
            f"Test image-ID sets differ; only A examples={only_a}, only B examples={only_b}"
        )

    pos = {sid: i for i, sid in enumerate(ib.tolist())}
    order = np.asarray([pos[sid] for sid in ia.tolist()], dtype=np.int64)
    yb, pb = yb[order], pb[order]
    if not np.array_equal(ya, yb):
        raise ValueError("Ground-truth labels disagree after exact image-ID alignment")

    a_correct = pa == ya
    b_correct = pb == ya
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    both = int((a_correct & b_correct).sum())
    neither = int((~a_correct & ~b_correct).sum())
    discordant = a_only + b_only
    pvalue = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0

    declared = {
        (x["a"], x["b"]): x.get("purpose", "")
        for x in matrix().get("paired_comparisons", [])
    }
    exp_a = ra.name.rsplit("_s", 1)[0]
    exp_b = rb.name.rsplit("_s", 1)[0]
    purpose = declared.get((exp_a, exp_b), declared.get((exp_b, exp_a), ""))

    out = {
        "campaign_id": ca["campaign_id"],
        "a": ra.name,
        "b": rb.name,
        "purpose": purpose,
        "n_test": int(len(ya)),
        "both_correct": both,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": neither,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": pvalue,
        "multiplicity_note": (
            "This is one pairwise exact McNemar test. If multiple pairwise tests are used "
            "for confirmatory significance claims, define the family and apply an explicit "
            "multiple-testing procedure in the statistical analysis plan."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
