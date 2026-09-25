#!/usr/bin/env python3
"""
18_confusable_pair_analysis.py
==============================

Why do the models confuse two specific classes, and would colour have helped?

POST-HOC AND EXPLORATORY. Not a predeclared analysis; its output is not part of
the campaign evidence. Reads completed run artifacts, the frozen manifest and the
frozen images, and never modifies any of them.

Default pair: Brown Blight vs Tea algal leaf spot -- the dominant confusion among
the images every arm misclassifies, and the pair on which the HSV gate opened
widest while colour lowered F1.

Every question is answered with a number, and every feature set and model is
fixed here in advance, so no result is chosen after seeing test performance.

  Q1  How often does each arm confuse the pair, in each direction, and does
      adding HSV (R1-R3 vs R0) or token attention (TLA vs C1) change that?
  Q2  Is colour discriminative for this pair in these photographs? Colour-only
      classifiers are fitted on the training ORIGINALS and scored on the test
      images, for five feature sets:
        global_mean    mean hue (sin/cos), saturation, value over the whole image --
                       a proxy for what a globally pooled colour branch receives
        global_hist    whole-image hue / saturation / value histograms
        lesion_colour  hue / saturation / value inside a heuristic lesion mask
        lesion_morph   lesion pixel fraction and connected-component count / size
        lesion_all     lesion_colour + lesion_morph
  Q3  How much of each image is lesion? A small fraction means global average
      pooling -- which is how the R1-R3 HSV branch ends -- dilutes it.
  Q4  Are the networks' pair confusions the images that colour cannot separate?
  Q5  (--with-models) Does the trained HSV branch's pooled embedding carry the
      pair signal? Linear probes on the HSV embedding of R1-R3 and on the RGB
      backbone feature of every HSV-family arm, plus each image's gate value.

The lesion mask is a heuristic, not a segmentation: chromatic warm-hue pixels
(red / orange / brown / tan), with leaf green and the low-saturation paper
background excluded. images/mask_overlay.png draws it on the pair's confused test
images so its behaviour can be checked by eye before any number is trusted.

Known limitations, from a visual review of that overlay and checked numerically:
it substantially under-segments dark brown / black lesions; it also selects leaf
margins, veins, tears and hole boundaries (on one image most of the mask is the
leaf edge); and it picks up occasional debris on the paper. Treat every lesion
figure as heuristic-selected area, not validated lesion severity -- measuring
accuracy requires expert reference masks.

Lesion area is reported against TWO denominators, both named in every column:
  lesion_frac_of_image  share of the whole 224x224 crop, paper included -- the
                        right denominator for what a globally pooled branch sees
  lesion_frac_of_leaf   share of the leaf (largest non-paper object, holes filled)
                        -- the right denominator for lesion burden
The leaf fills only part of the crop, so the two differ several-fold.
leaf_frac_of_image is reported too, to check that framing does not differ by
class. mask_frac_on_margin / mask_frac_on_paper quantify the contamination above.
These are reported quantities; the five feature sets are fixed and unchanged.

Writes to <results_dir parent>/analysis/pair_<A>_vs_<B>_s<SEED>/:
    summary.json             every statistic
    per_image.csv            per-image lesion features; test rows add colour-model
                             probabilities, every arm's prediction and gate values
    figures/*.png            lesion colour scatter, lesion fraction, confusion by arm
    images/mask_overlay.png  lesion-mask overlays                         (gitignored)

USAGE
-----
    python scripts/18_confusable_pair_analysis.py
    python scripts/18_confusable_pair_analysis.py --with-models      # CPU; minutes
    python scripts/18_confusable_pair_analysis.py --pair "Gray Blight" "Red spider"
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from campaign import location, matrix, require_campaign

REPO = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Identical to the evaluation transform both trainers apply, so the colour
# statistics describe exactly the pixels the networks saw.
CROP = transforms.Compose([
    transforms.Resize(int(224 * 1.14), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
FEATURE_SETS = ("global_mean", "global_hist", "lesion_colour", "lesion_morph", "lesion_all")
PAPER_S_MAX, PAPER_V_MIN = 0.12, 0.55     # paper background: low saturation, bright
MARGIN_PX = 3                             # band inside the leaf edge and around holes
N_BOOT = 2000


# ----------------------------------------------------------------------------- pixels

def load_hsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        rgb = np.asarray(CROP(im.convert("RGB")), dtype=np.float32) / 255.0
    return rgb, mcolors.rgb_to_hsv(rgb)


def lesion_mask(hsv: np.ndarray, a) -> np.ndarray:
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    warm = (h <= a.hue_max) | (h >= a.hue_wrap_min)
    return warm & (s >= a.s_min) & (v >= a.v_min) & (v <= a.v_max)


def leaf_mask(hsv: np.ndarray) -> np.ndarray:
    """The leaf: the largest non-paper object, with holes and lesions filled in.

    Paper is low-saturation and bright. Used only as a denominator and to locate the
    leaf margin -- it is a heuristic like the lesion mask, not a segmentation.
    """
    s, v = hsv[..., 1], hsv[..., 2]
    paper = (s < PAPER_S_MAX) & (v > PAPER_V_MIN)
    leaf = ndimage.binary_fill_holes(ndimage.binary_opening(~paper, iterations=2))
    lab, n = ndimage.label(leaf)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else leaf


def circular(h_deg: np.ndarray) -> tuple[float, float]:
    r = np.deg2rad(h_deg)
    return float(np.sin(r).mean()), float(np.cos(r).mean())


def signed_hue(sin_m: float, cos_m: float) -> float:
    deg = math.degrees(math.atan2(sin_m, cos_m)) % 360.0
    return deg - 360.0 if deg > 180.0 else deg


def describe(hsv: np.ndarray, mask: np.ndarray, a) -> tuple[dict, dict]:
    """Return (feature vectors by set, human-readable per-image columns)."""
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    gs, gc = circular(h.ravel())
    n = h.size
    f = {
        "global_mean": [gs, gc, float(s.mean()), float(v.mean()), float(s.std()), float(v.std())],
        "global_hist": list(np.concatenate([
            np.histogram(h, bins=18, range=(0, 360))[0],
            np.histogram(s, bins=8, range=(0, 1))[0],
            np.histogram(v, bins=8, range=(0, 1))[0]]) / n),
    }
    frac = float(mask.mean())
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes >= a.min_component_px]
    n_comp = int(sizes.size)
    med = float(np.median(sizes)) if n_comp else 0.0
    largest = float(sizes.max() / sizes.sum()) if n_comp else 0.0
    f["lesion_morph"] = [frac, math.log1p(n_comp), math.log1p(med), largest]

    if mask.any():
        lh, ls, lv = h[mask], s[mask], v[mask]
        cont = np.where(lh >= a.hue_wrap_min, lh - 360.0, lh)   # warm hues on one axis
        ms, mc = circular(lh)
        m = lh.size
        colour = [ms, mc, float(ls.mean()), float(lv.mean()), float(ls.std()), float(lv.std())]
        colour += list(np.histogram(cont, bins=8, range=(a.hue_wrap_min - 360.0, a.hue_max))[0] / m)
        colour += list(np.histogram(ls, bins=6, range=(a.s_min, 1.0))[0] / m)
        colour += list(np.histogram(lv, bins=6, range=(a.v_min, a.v_max))[0] / m)
        readable = {"lesion_hue_deg": round(signed_hue(ms, mc), 2),
                    "lesion_sat": round(float(ls.mean()), 4), "lesion_val": round(float(lv.mean()), 4)}
    else:
        colour = [0.0] * 26
        readable = {"lesion_hue_deg": "", "lesion_sat": "", "lesion_val": ""}
    f["lesion_colour"] = colour
    f["lesion_all"] = colour + f["lesion_morph"]
    # Reported quantities only -- the feature sets above are fixed and unchanged.
    # Two denominators: the whole crop (what a globally pooled branch sees) and the
    # leaf (lesion burden). The leaf fills only a fraction of the crop.
    leaf = leaf_mask(hsv)
    s_, v_ = hsv[..., 1], hsv[..., 2]
    tissue = leaf & ~((s_ < PAPER_S_MAX) & (v_ > PAPER_V_MIN))      # leaf minus holes
    margin = tissue & ~ndimage.binary_erosion(tissue, iterations=MARGIN_PX)
    n_mask = max(int(mask.sum()), 1)
    readable.update({
        "lesion_frac_of_image": round(frac, 5),
        "leaf_frac_of_image": round(float(leaf.mean()), 5),
        "lesion_frac_of_leaf": round(float((mask & leaf).sum() / max(int(leaf.sum()), 1)), 5),
        "mask_frac_on_margin": round(float((mask & margin).sum() / n_mask), 4),
        "mask_frac_on_paper": round(float((mask & ~leaf).sum() / n_mask), 4),
        "n_components": n_comp, "median_component_px": med})
    return f, readable


# ----------------------------------------------------------------------------- models

def probe():
    """One fixed classifier for every feature set; C chosen by CV on train only."""
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=10, cv=5, scoring="roc_auc", max_iter=5000, class_weight="balanced"))


def bootstrap_auc(y: np.ndarray, p: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(Xtr, ytr, Xte, yte) -> tuple[dict, np.ndarray]:
    m = probe().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    lo, hi = bootstrap_auc(yte, p)
    cv = m[-1].scores_[1].mean(axis=0).max()
    return {"test_auc": round(float(roc_auc_score(yte, p)), 4),
            "test_auc_ci95": [round(lo, 4), round(hi, 4)],
            "test_balanced_accuracy": round(float(balanced_accuracy_score(yte, p >= 0.5)), 4),
            "train_cv_auc": round(float(cv), 4),
            "n_features": int(np.asarray(Xtr).shape[1])}, p


# ----------------------------------------------------------------------------- runs

def load_arms(runs_dir: Path, seed: int) -> list[dict]:
    arms = []
    for run in sorted(runs_dir.glob(f"*_s{seed}")):
        if not (run / "metrics" / "test_results.json").exists():
            continue
        require_campaign(run)
        raw = run / "raw_outputs"
        arms.append({"name": run.name.rsplit("_s", 1)[0], "dir": run,
                     "ids": json.loads((raw / "test_ids.json").read_text()),
                     "y": np.load(raw / "test_targets.npy"),
                     "pred": np.load(raw / "test_predictions.npy"),
                     "prob": np.load(raw / "test_probabilities.npy"),
                     "classes": json.loads((run / "config" / "classes.json").read_text())})
    if not arms:
        raise SystemExit(f"No completed runs for seed {seed}")
    for a in arms[1:]:
        if a["ids"] != arms[0]["ids"] or a["classes"] != arms[0]["classes"]:
            raise SystemExit(f"{a['name']} does not share {arms[0]['name']}'s test set / classes")
    return arms


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_pass(run_dir: Path, paths: list[Path], device: str) -> dict:
    """Backbone feature, pooled HSV embedding and gate for each image, from one pass."""
    import torch
    sys.path.insert(0, str(REPO / "src"))
    robustness = load_module(REPO / "scripts" / "08_eval_robustness.py", "eval_robustness")
    model = robustness.load_model(run_dir, device)     # the same strict reconstruction 08 uses
    captured: dict[str, list] = {"feat": [], "hsv": []}
    hooks = [model.backbone.register_forward_hook(
        lambda _m, _i, o: captured["feat"].append(o.detach().float().cpu()))]
    has_hsv = getattr(model, "use_hsv_branch", False)
    if has_hsv:
        hooks.append(model.hsv_branch.register_forward_hook(
            lambda _m, _i, o: captured["hsv"].append(o.detach().float().cpu())))
    tensor = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    gates = []
    with torch.no_grad():
        for k in range(0, len(paths), 16):
            batch = []
            for p in paths[k:k + 16]:
                with Image.open(p) as im:
                    batch.append(tensor(CROP(im.convert("RGB"))))
            _, gate = model(torch.stack(batch).to(device), return_gate=True, gate_alpha=1.0)
            gates.append(gate.float().cpu().reshape(-1))
    for h in hooks:
        h.remove()
    return {"feat": torch.cat(captured["feat"]).numpy(),
            "hsv": torch.cat(captured["hsv"]).numpy() if has_hsv else None,
            "gate": torch.cat(gates).numpy() if has_hsv else None}


# ----------------------------------------------------------------------------- figures

def overlay_sheet(rows: list[dict], root: Path, a, out: Path, limit: int = 16) -> None:
    rows = rows[:limit]
    if not rows:
        return
    t, cap = 224, 30
    sheet = Image.new("RGB", (2 * t * 2, ((len(rows) + 1) // 2) * (t + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        rgb, hsv = load_hsv(root / r["image_id"])
        mask = lesion_mask(hsv, a)
        tinted = rgb.copy()
        tinted[mask] = 0.35 * tinted[mask] + 0.65 * np.array([1.0, 0.0, 1.0])
        x, y = (k % 2) * 2 * t, (k // 2) * (t + cap)
        sheet.paste(Image.fromarray((rgb * 255).astype(np.uint8)), (x, y))
        sheet.paste(Image.fromarray((tinted * 255).astype(np.uint8)), (x + t, y))
        draw.text((x + 3, y + t + 2), f"{Path(r['image_id']).name}"[:36], fill="black")
        draw.text((x + 3, y + t + 15),
                  f"confused {r['n_pair_confused']}/{r['n_arms']} | lesion {r['lesion_frac_of_image']*100:.1f}% "
                  f"of image, {r['lesion_frac_of_leaf']*100:.1f}% of leaf"[:72], fill="darkred")
    sheet.save(out)


def figures(train_rows, test_rows, arms_q1, A, B, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cls, colour in ((A, "#8c510a"), (B, "#d95f02")):
        pts = [r for r in train_rows if r["class"] == cls and r["lesion_hue_deg"] != ""]
        ax.scatter([r["lesion_hue_deg"] for r in pts], [r["lesion_sat"] for r in pts],
                   s=9, alpha=0.45, label=f"{cls} (train originals, n={len(pts)})", color=colour)
    ax.set_xlabel("mean lesion hue (degrees; negative = red side of 0)")
    ax.set_ylabel("mean lesion saturation")
    ax.set_title("Lesion colour per image")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "lesion_colour_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, label in ((axes[0], "lesion_frac_of_image", "% of the 224x224 crop"),
                           (axes[1], "lesion_frac_of_leaf", "% of the leaf")):
        ax.boxplot([[r[key] * 100 for r in train_rows if r["class"] == c] for c in (A, B)],
                   showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([A, B])      # boxplot(labels=) was removed in matplotlib 3.11
        ax.set_ylabel(f"heuristic lesion area ({label})")
    axes[0].set_title("What a globally pooled branch sees")
    axes[1].set_title("Lesion burden on the leaf")
    fig.savefig(out / "lesion_fraction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [q["arm"] for q in arms_q1]
    idx = np.arange(len(names))
    ax.bar(idx - 0.2, [q[f"{A} -> {B}"] for q in arms_q1], 0.4, label=f"{A} -> {B}")
    ax.bar(idx + 0.2, [q[f"{B} -> {A}"] for q in arms_q1], 0.4, label=f"{B} -> {A}")
    ax.set_xticks(idx)
    ax.set_xticklabels(names)
    ax.set_ylabel("test images")
    ax.set_title("Pair confusions by arm")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "confusion_by_arm.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, default=["Brown Blight", "Tea algal leaf spot"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-models", action="store_true",
                    help="Also extract embeddings and gates from the HSV-family checkpoints")
    ap.add_argument("--device", default="cpu",
                    help="Device for --with-models (default cpu, so a training run keeps the GPU)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--hue-max", type=float, default=50.0, dest="hue_max")
    ap.add_argument("--hue-wrap-min", type=float, default=330.0, dest="hue_wrap_min")
    ap.add_argument("--s-min", type=float, default=0.18, dest="s_min")
    ap.add_argument("--v-min", type=float, default=0.12, dest="v_min")
    ap.add_argument("--v-max", type=float, default=0.97, dest="v_max")
    ap.add_argument("--min-component-px", type=int, default=4, dest="min_component_px")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = matrix()
    defaults = cfg["defaults"]
    A, B = args.pair
    data_root = REPO / defaults["data_root"]

    arms = load_arms(location("results_dir"), args.seed)
    classes = arms[0]["classes"]
    for c in (A, B):
        if c not in classes:
            raise SystemExit(f"Unknown class {c!r}; choose from {classes}")
    ia, ib = classes.index(A), classes.index(B)
    ids = arms[0]["ids"]
    y_all = arms[0]["y"]

    # ---- images: training ORIGINALS only (derivatives repeat their sources), all test
    train, test = [], []
    with (REPO / defaults["manifest_csv"]).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["class_name"] not in (A, B):
                continue
            derived = str(r["is_derived"]).strip().lower() in {"1", "true", "t", "yes", "y"}
            item = {"split": r["split"], "class": r["class_name"],
                    "image_id": f"{r['class_name']}/{r['filename']}",
                    "label": int(r["class_name"] == B)}
            if r["split"] == "train" and not derived:
                train.append(item)
            elif r["split"] == "test":
                test.append(item)

    feats = {s: {"train": [], "test": []} for s in FEATURE_SETS}
    for split, items in (("train", train), ("test", test)):
        for it in items:
            _, hsv = load_hsv(data_root / split / it["image_id"])
            f, readable = describe(hsv, lesion_mask(hsv, args), args)
            it.update(readable)
            for s in FEATURE_SETS:
                feats[s][split].append(f[s])
    ytr = np.array([t["label"] for t in train])
    yte = np.array([t["label"] for t in test])

    # ---- Q2: colour-only separability
    q2, colour_prob = {}, {}
    for s in FEATURE_SETS:
        q2[s], colour_prob[s] = evaluate(np.array(feats[s]["train"]), ytr,
                                         np.array(feats[s]["test"]), yte)

    # ---- Q3: how much of each image is lesion
    def dist(vals):
        v = np.asarray(vals, dtype=float)
        return {"n": int(v.size), "median": round(float(np.median(v)), 5),
                "q25": round(float(np.percentile(v, 25)), 5),
                "q75": round(float(np.percentile(v, 75)), 5)}
    q3 = {}
    for split, items in (("train_originals", train), ("test", test)):
        per = {}
        for c in (A, B):
            rows = [r for r in items if r["class"] == c]
            per[c] = {
                "lesion_frac_of_image": dist([r["lesion_frac_of_image"] for r in rows]),
                "lesion_frac_of_leaf": dist([r["lesion_frac_of_leaf"] for r in rows]),
                "leaf_frac_of_image": dist([r["leaf_frac_of_image"] for r in rows]),
                "mask_frac_on_margin": dist([r["mask_frac_on_margin"] for r in rows]),
                "mask_frac_on_paper": dist([r["mask_frac_on_paper"] for r in rows]),
                "images_with_lt_0.1pct_lesion": int(sum(r["lesion_frac_of_image"] < 0.001 for r in rows)),
                "lesion_hue_deg": dist([r["lesion_hue_deg"] for r in rows if r["lesion_hue_deg"] != ""]),
                "lesion_saturation": dist([r["lesion_sat"] for r in rows if r["lesion_sat"] != ""]),
                "components": dist([r["n_components"] for r in rows]),
            }
        q3[split] = per
    for key, col in (("lesion_frac_of_image", "lesion_frac_of_image"),
                     ("lesion_frac_of_leaf", "lesion_frac_of_leaf"),
                     ("leaf_frac_of_image", "leaf_frac_of_image"),          # framing check
                     ("lesion_hue_deg", "lesion_hue_deg"),
                     ("lesion_saturation", "lesion_sat"), ("components", "n_components")):
        xa = [r[col] for r in train if r["class"] == A and r[col] != ""]
        xb = [r[col] for r in train if r["class"] == B and r[col] != ""]
        q3["train_originals"][f"mannwhitney_p_{key}"] = float(mannwhitneyu(xa, xb).pvalue)

    # ---- Q1: pair confusion per arm, and change against the family baseline
    exps = cfg["experiments"]
    baseline = {v["family"]: k for k, v in exps.items() if v.get("role") == "baseline"}
    q1 = []
    for a in arms:
        ma, mb = a["y"] == ia, a["y"] == ib
        q1.append({"arm": a["name"], "family": exps.get(a["name"], {}).get("family", ""),
                   f"{A} -> {B}": int((a["pred"][ma] == ib).sum()),
                   f"{B} -> {A}": int((a["pred"][mb] == ia).sum()),
                   f"recall {A}": round(float((a["pred"][ma] == ia).mean()), 4),
                   f"recall {B}": round(float((a["pred"][mb] == ib).mean()), 4),
                   f"mean p({B} | true {A})": round(float(a["prob"][ma, ib].mean()), 4),
                   f"mean p({A} | true {B})": round(float(a["prob"][mb, ia].mean()), 4)})
    by_arm = {q["arm"]: q for q in q1}
    for q in q1:
        base = baseline.get(q["family"])
        if base and base in by_arm and base != q["arm"]:
            q["pair_confusions_minus_baseline"] = (
                q[f"{A} -> {B}"] + q[f"{B} -> {A}"]
                - by_arm[base][f"{A} -> {B}"] - by_arm[base][f"{B} -> {A}"])

    # ---- per-image network behaviour on the test pair
    pos = {sid: i for i, sid in enumerate(ids)}
    for j, t in enumerate(test):
        i = pos[t["image_id"]]
        if y_all[i] != (ib if t["label"] else ia):
            raise SystemExit(f"Label disagreement for {t['image_id']}")
        other = ia if t["label"] else ib
        t["n_arms"] = len(arms)
        t["n_arms_wrong"] = int(sum(a["pred"][i] != y_all[i] for a in arms))
        t["n_pair_confused"] = int(sum(a["pred"][i] == other for a in arms))
        for a in arms:
            t[f"{a['name']}_pred"] = classes[a["pred"][i]]
        for s in ("global_mean", "lesion_colour", "lesion_all"):
            t[f"p_colour_{s}"] = round(float(colour_prob[s][j]), 4)

    # ---- Q4: do the networks fail where colour fails? (lesion_colour, fixed in advance)
    colour_correct = (colour_prob["lesion_colour"] >= 0.5) == (yte == 1)
    net_confused = np.array([t["n_pair_confused"] > len(arms) / 2 for t in test])
    table = [[int((colour_correct & ~net_confused).sum()), int((colour_correct & net_confused).sum())],
             [int((~colour_correct & ~net_confused).sum()), int((~colour_correct & net_confused).sum())]]
    q4 = {
        "colour_model": "lesion_colour",
        "table_rows_colour_correct_then_wrong_cols_networks_ok_then_majority_confused": table,
        "fisher_exact_p": float(fisher_exact(table)[1]),
        "mean_arms_confused_when_colour_correct": round(float(
            np.mean([t["n_pair_confused"] for t, c in zip(test, colour_correct) if c])), 3),
        "mean_arms_confused_when_colour_wrong": round(float(
            np.mean([t["n_pair_confused"] for t, c in zip(test, colour_correct) if not c]) if
            (~colour_correct).any() else float("nan")), 3),
    }

    # ---- Q5: embeddings and gates
    q5 = None
    if args.with_models:
        import torch
        torch.set_num_threads(args.threads)
        paths = ([data_root / "train" / t["image_id"] for t in train]
                 + [data_root / "test" / t["image_id"] for t in test])
        ntr = len(train)
        q5 = {"device": args.device, "probes": {}, "gates": {}}
        for a in arms:
            if exps.get(a["name"], {}).get("family") != "hsv":
                continue
            print(f"  model pass: {a['name']} ({len(paths)} images on {args.device})", flush=True)
            out = model_pass(a["dir"], paths, args.device)
            q5["probes"][f"{a['name']}_rgb_backbone_feature"], _ = evaluate(
                out["feat"][:ntr], ytr, out["feat"][ntr:], yte)
            if out["hsv"] is not None:
                q5["probes"][f"{a['name']}_hsv_branch_embedding"], _ = evaluate(
                    out["hsv"][:ntr], ytr, out["hsv"][ntr:], yte)
                g = out["gate"][ntr:]
                for t, gv in zip(test, g):
                    t[f"{a['name']}_gate"] = round(float(gv), 5)
                confused = np.array([t[f"{a['name']}_pred"] == (A if t["label"] else B) for t in test])
                q5["gates"][a["name"]] = {
                    f"mean_gate_true_{A}": round(float(g[yte == 0].mean()), 5),
                    f"mean_gate_true_{B}": round(float(g[yte == 1].mean()), 5),
                    "mean_gate_when_this_arm_confused_the_pair": round(float(g[confused].mean()), 5)
                    if confused.any() else None,
                    "mean_gate_otherwise": round(float(g[~confused].mean()), 5),
                    "spearman_gate_vs_lesion_frac_of_image": round(float(
                        spearmanr(g, [t["lesion_frac_of_image"] for t in test]).statistic), 4),
                    "spearman_gate_vs_lesion_frac_of_leaf": round(float(
                        spearmanr(g, [t["lesion_frac_of_leaf"] for t in test]).statistic), 4),
                }

    # ---- write
    slug = lambda c: c.lower().replace(" ", "_")
    out = Path(args.out_dir) if args.out_dir else (
        location("results_dir").parent / "analysis" / f"pair_{slug(A)}_vs_{slug(B)}_s{args.seed}")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = train + test
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    figures(train, test, q1, A, B, out / "figures")
    confused_first = sorted(test, key=lambda t: (-t["n_pair_confused"], -t["n_arms_wrong"]))
    overlay_sheet([t for t in confused_first if t["n_pair_confused"] > 0],
                  data_root / "test", args, out / "images" / "mask_overlay.png")

    summary = {
        "campaign_id": defaults["campaign_id"], "seed": args.seed, "pair": [A, B],
        "arms": [a["name"] for a in arms],
        "n_train_originals": {A: int((ytr == 0).sum()), B: int((ytr == 1).sum())},
        "n_test": {A: int((yte == 0).sum()), B: int((yte == 1).sum())},
        "lesion_mask": {k: getattr(args, k) for k in
                        ("hue_max", "hue_wrap_min", "s_min", "v_min", "v_max", "min_component_px")},
        "Q1_pair_confusion_by_arm": q1,
        "Q2_colour_only_separability": q2,
        "Q3_lesion_extent_and_colour": q3,
        "Q4_networks_vs_colour": q4,
        "Q5_embeddings_and_gates": q5,
        "note": "Post-hoc exploratory analysis; not a predeclared comparison. "
                "The lesion mask is a colour heuristic, not a segmentation.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"pair: {A} (0) vs {B} (1) | train originals {len(train)} | test {len(test)} | "
          f"arms {', '.join(summary['arms'])}")
    print("\nQ1 pair confusions (test):")
    for q in q1:
        d = q.get("pair_confusions_minus_baseline")
        print(f"  {q['arm']:4s} {A}->{B}: {q[f'{A} -> {B}']:2d}   {B}->{A}: {q[f'{B} -> {A}']:2d}"
              + (f"   vs baseline {d:+d}" if d is not None else ""))
    print("\nQ2 colour-only separability (fit on train originals, scored on test):")
    for s in FEATURE_SETS:
        r = q2[s]
        print(f"  {s:14s} test AUC {r['test_auc']:.3f} "
              f"[{r['test_auc_ci95'][0]:.3f}, {r['test_auc_ci95'][1]:.3f}]  "
              f"bal-acc {r['test_balanced_accuracy']:.3f}  train-CV AUC {r['train_cv_auc']:.3f}")
    t3 = q3["train_originals"]
    print("\nQ3 heuristic lesion area and colour (train originals; medians):")
    for c in (A, B):
        d = t3[c]
        print(f"  {c:22s} lesion {d['lesion_frac_of_image']['median']*100:5.2f}% of image, "
              f"{d['lesion_frac_of_leaf']['median']*100:5.2f}% of leaf | "
              f"leaf {d['leaf_frac_of_image']['median']*100:5.2f}% of image | "
              f"hue {d['lesion_hue_deg']['median']:5.1f} deg  sat {d['lesion_saturation']['median']:.3f}  "
              f"components {d['components']['median']:.0f}")
        print(f"  {'':22s} mask on leaf margin {d['mask_frac_on_margin']['median']*100:4.1f}%, "
              f"on paper {d['mask_frac_on_paper']['median']*100:4.1f}%  "
              f"<0.1% lesion: {d['images_with_lt_0.1pct_lesion']}")
    print(f"  Mann-Whitney p: lesion/image {t3['mannwhitney_p_lesion_frac_of_image']:.2g}, "
          f"lesion/leaf {t3['mannwhitney_p_lesion_frac_of_leaf']:.2g}, "
          f"leaf framing {t3['mannwhitney_p_leaf_frac_of_image']:.2g}")
    print(f"\nQ4 {q4['colour_model']}: arms confusing the pair, mean "
          f"{q4['mean_arms_confused_when_colour_correct']} where colour is right vs "
          f"{q4['mean_arms_confused_when_colour_wrong']} where colour is wrong "
          f"(Fisher p={q4['fisher_exact_p']:.3g})")
    if q5:
        print("\nQ5 linear probes (fit on train originals, scored on test):")
        for k, r in q5["probes"].items():
            print(f"  {k:36s} test AUC {r['test_auc']:.3f} "
                  f"[{r['test_auc_ci95'][0]:.3f}, {r['test_auc_ci95'][1]:.3f}]")
        for k, g in q5["gates"].items():
            print(f"  {k} gate: {g}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
