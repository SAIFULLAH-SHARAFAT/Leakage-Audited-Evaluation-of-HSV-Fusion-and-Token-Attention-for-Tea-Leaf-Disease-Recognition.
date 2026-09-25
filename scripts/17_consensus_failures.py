#!/usr/bin/env python3
"""
17_consensus_failures.py
========================

Collect the test images that every completed arm misclassifies, for expert review.

POST-HOC AND EXPLORATORY. This is not one of the predeclared analyses and its
output is not part of the campaign evidence. It reads completed run artifacts and
the frozen test images and never modifies either.

An image that six independently trained, architecturally different models all get
wrong -- and that they all get wrong *the same way* -- is more likely a labelling
or genuine-ambiguity case than a capacity gap. This script gathers those images,
records what every arm predicted, flags images that are near-duplicates of each
other inside the test split, and lays out a review sheet with blank expert columns.

Writes to <results_dir parent>/analysis/consensus_failures_s<SEED>/:

    consensus_failures_s<SEED>.csv   one row per image: true label, modal wrong label,
                             agreement, every arm's prediction and confidence,
                             source prefix, near-duplicate partner, and blank
                             expert_label / expert_notes columns to fill in
    summary.json             arms used, counts, unanimity, class and pair tallies
    images/<true>__as__<pred>/<file>   copies of the images            (gitignored)
    images/contact_sheet.png           one-page grid for visual review (gitignored)

Image copies stay out of git, in line with the repository's policy of not
versioning image data.

USAGE
-----
    python scripts/17_consensus_failures.py                 # seed 42, every completed arm
    python scripts/17_consensus_failures.py --seed 1337
    python scripts/17_consensus_failures.py --min-arms 5    # wrong in at least 5 arms
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from campaign import location, matrix, require_campaign

REPO = Path(__file__).resolve().parents[1]


def completed_runs(runs_dir: Path, seed: int) -> list[Path]:
    runs = sorted(d for d in runs_dir.glob(f"*_s{seed}")
                  if (d / "metrics" / "test_results.json").exists())
    if len(runs) < 2:
        raise SystemExit(f"Need at least two completed runs for seed {seed}; found {len(runs)}")
    return runs


def load_arm(run: Path) -> dict:
    require_campaign(run)
    raw = run / "raw_outputs"
    return {
        "name": run.name.rsplit("_s", 1)[0],
        "ids": json.loads((raw / "test_ids.json").read_text()),
        "y": np.load(raw / "test_targets.npy"),
        "pred": np.load(raw / "test_predictions.npy"),
        "prob": np.load(raw / "test_probabilities.npy"),
        "classes": json.loads((run / "config" / "classes.json").read_text()),
    }


def surviving_test_pairs(manifest_csv: Path, pairs_csv: Path) -> dict[str, str]:
    """Map each test image to its near-duplicate partner, for pairs where both survive."""
    test = set()
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                test.add(f"{r['class_name']}/{r['filename']}")
    partner: dict[str, str] = {}
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split_i"] != "test" or r["split_j"] != "test":
                continue
            a, b = f"{r['class_name']}/{r['file_i']}", f"{r['class_name']}/{r['file_j']}"
            if a in test and b in test:
                partner[a], partner[b] = b, a
    return partner


def source_prefix(image_id: str) -> str:
    stem = Path(image_id).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def contact_sheet(rows: list[dict], image_root: Path, out: Path, thumb: int = 256) -> None:
    cols = 4
    caption = 44
    rows_n = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        with Image.open(image_root / r["image_id"]) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            x, y = (k % cols) * thumb, (k // cols) * (thumb + caption)
            sheet.paste(im, (x + (thumb - im.width) // 2, y))
        draw.text((x + 4, y + thumb + 2), f"#{r['rank']} {Path(r['image_id']).name}"[:40], fill="black")
        draw.text((x + 4, y + thumb + 16), f"true: {r['true_label']}"[:40], fill="darkgreen")
        wrong_as = r["modal_wrong_label"].replace(" (tie)", "").replace(" / ", "+")
        draw.text((x + 4, y + thumb + 30),
                  f"pred: {wrong_as} ({r['agreement']})"[:42], fill="darkred")
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-arms", type=int, default=None,
                    help="Minimum number of arms that must be wrong (default: all arms)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = matrix()
    defaults = cfg["defaults"]
    runs = completed_runs(location("results_dir"), args.seed)
    arms = [load_arm(r) for r in runs]

    ref = arms[0]
    for a in arms[1:]:
        if a["ids"] != ref["ids"]:
            raise SystemExit(f"{a['name']} evaluated a different test ordering than {ref['name']}")
        if not np.array_equal(a["y"], ref["y"]):
            raise SystemExit(f"{a['name']} has different test targets than {ref['name']}")
        if a["classes"] != ref["classes"]:
            raise SystemExit(f"{a['name']} has a different class list than {ref['name']}")

    classes, ids, y = ref["classes"], ref["ids"], ref["y"]
    n_arms = len(arms)
    min_arms = args.min_arms or n_arms
    if not 1 <= min_arms <= n_arms:
        raise SystemExit(f"--min-arms must be between 1 and {n_arms}")

    pred = np.stack([a["pred"] for a in arms])            # arms x images
    prob = np.stack([a["prob"] for a in arms])            # arms x images x classes
    wrong = pred != y[None, :]
    n_wrong = wrong.sum(0)
    selected = np.where(n_wrong >= min_arms)[0]

    partner = surviving_test_pairs(REPO / defaults["manifest_csv"],
                                   REPO / "data" / "manifests" / "final_linked_pairs.csv")

    rows = []
    for i in selected:
        wrong_preds = pred[wrong[:, i], i]
        # Rank by count, then class index, so ties resolve deterministically; a
        # tie is reported as such rather than silently choosing one label.
        ranked = sorted(Counter(wrong_preds.tolist()).items(), key=lambda kv: (-kv[1], kv[0]))
        agree = ranked[0][1]
        tied = [c for c, k in ranked if k == agree]
        modal = tied[0]
        modal_label = (" / ".join(classes[c] for c in tied) + " (tie)") if len(tied) > 1 \
            else classes[modal]
        row = {
            "image_id": ids[i],
            "true_label": classes[y[i]],
            "modal_wrong_label": modal_label,
            "agreement": f"{agree}/{int(n_wrong[i])}",
            "unanimous_same_wrong_label": bool(agree == n_arms),
            "n_arms_wrong": int(n_wrong[i]),
            "n_arms": n_arms,
            "p_true_mean": round(float(prob[:, i, y[i]].mean()), 4),
            "p_modal_mean": round(float(prob[:, i, modal].mean()), 4),
            "source_prefix": source_prefix(ids[i]),
            "near_duplicate_partner": partner.get(ids[i], ""),
        }
        for a_idx, a in enumerate(arms):
            row[f"{a['name']}_pred"] = classes[pred[a_idx, i]]
            row[f"{a['name']}_p_pred"] = round(float(prob[a_idx, i, pred[a_idx, i]]), 4)
        row["expert_label"] = ""
        row["expert_notes"] = ""
        rows.append(row)

    rows.sort(key=lambda r: (-r["n_arms_wrong"], -r["p_modal_mean"]))
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank

    out = Path(args.out_dir) if args.out_dir else (
        location("results_dir").parent / "analysis" / f"consensus_failures_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    images = out / "images"
    if images.exists():
        shutil.rmtree(images)          # this directory is owned by this script
    image_root = REPO / defaults["data_root"] / "test"
    for r in rows:
        wrong_as = r["modal_wrong_label"].replace(" (tie)", "").replace(" / ", "+")
        dst = images / f"{r['true_label']}__as__{wrong_as}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_root / r["image_id"], dst / Path(r["image_id"]).name)
    if rows:
        contact_sheet(rows, image_root, images / "contact_sheet.png")

    fields = ["rank"] + [k for k in rows[0] if k != "rank"] if rows else ["rank"]
    with (out / f"consensus_failures_s{args.seed}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    in_pairs = sorted({r["image_id"] for r in rows if r["near_duplicate_partner"] in
                       {x["image_id"] for x in rows}})
    summary = {
        "campaign_id": defaults["campaign_id"],
        "seed": args.seed,
        "arms": [a["name"] for a in arms],
        "min_arms_wrong": min_arms,
        "n_test": int(len(y)),
        "n_selected": len(rows),
        "n_unanimous_same_wrong_label": int(sum(r["unanimous_same_wrong_label"] for r in rows)),
        "wrong_in_at_least_one_arm": int((n_wrong >= 1).sum()),
        "oracle_accuracy_ceiling": round(float((n_wrong < n_arms).mean()), 6),
        "best_arm_accuracy": round(float(max((a["pred"] == y).mean() for a in arms)), 6),
        "by_true_class": dict(Counter(r["true_label"] for r in rows)),
        "by_confusion": dict(Counter(f"{r['true_label']} -> {r['modal_wrong_label']}" for r in rows)),
        "selected_images_that_are_near_duplicates_of_each_other": in_pairs,
        "note": "Post-hoc exploratory analysis; not a predeclared comparison.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"arms ({n_arms}): {', '.join(summary['arms'])}")
    print(f"images wrong in >= {min_arms} arms: {len(rows)} of {len(y)}; "
          f"unanimous on the same wrong label: {summary['n_unanimous_same_wrong_label']}")
    print(f"oracle ceiling {summary['oracle_accuracy_ceiling']*100:.2f}% | "
          f"best arm {summary['best_arm_accuracy']*100:.2f}%")
    for pair, k in sorted(summary["by_confusion"].items(), key=lambda t: -t[1]):
        print(f"  {k:2d}  {pair}")
    if in_pairs:
        print(f"near-duplicates of each other inside this set: {in_pairs}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
