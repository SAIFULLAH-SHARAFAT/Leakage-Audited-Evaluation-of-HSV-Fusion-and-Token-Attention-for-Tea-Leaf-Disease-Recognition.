#!/usr/bin/env python3
"""
recover_aug_params.py
=====================

Recover the offline augmentation parameters from the dataset itself, by
comparing each derived training image against its source image.

For every `*_aug<op>*` file the corresponding original is located via the
canonical filename stem, and the transform parameter is estimated:

  rotation    angle, from an ORB feature match + partial affine fit
  zoom        scale factor, from the same affine fit
  brightness  multiplicative/additive factor, from a least-squares fit of
              derivative pixel values against source pixel values
  contrast    slope of that same fit
  flip        horizontal-flip check (parameter-free; verified, not estimated)
  blur        Gaussian sigma, by minimising the residual between the source
              blurred at candidate sigmas and the derivative

Reports per-operation min / median / max and a 5th-95th percentile range,
which is what belongs in the paper.

USAGE
-----
    pip install opencv-python-headless
    python recover_aug_params.py \
        --data-root Tea_leaf_dataset \
        --out dedup/aug_params.csv \
        --max-per-op 400

NOTES
-----
* Estimates are approximate. Brightness/contrast recover the *realised*
  photometric change, which is what a reader cares about, but may differ
  slightly from the nominal factor passed to the generator (JPEG requantisation,
  clipping at 0/255).
* Geometric estimates come from feature matching and can fail on low-texture
  images; failures are dropped and counted, not silently included.
* Report the recovered ranges as measured from the released derivatives.
  That is honest and verifiable, and stronger than a remembered parameter.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    raise SystemExit("opencv required:  pip install opencv-python-headless")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Matches the operation token in e.g.  brown_blight_00123_aug_rotation.jpg
OP_RE = re.compile(r"_aug[_-]?([a-z]+)", re.IGNORECASE)
AUG_RE = re.compile(
    r"(_aug(_|-)?(rotation|contrast|zoom|flip|brightness|blur|noise|color|crop|.*))$",
    re.IGNORECASE,
)

MAXDIM = 512          # downscale cap for speed
EPS = 1e-8


# =============================================================================
# HELPERS
# =============================================================================

def canonical_stem(fn: str) -> str:
    return AUG_RE.sub("", Path(fn).stem)


def op_of(fn: str) -> Optional[str]:
    m = OP_RE.search(fn)
    return m.group(1).lower() if m else None


def load_gray(p: Path) -> Optional[np.ndarray]:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    s = MAXDIM / max(h, w)
    if s < 1.0:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return im


# =============================================================================
# ESTIMATORS
# =============================================================================

def estimate_affine(src: np.ndarray, dst: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (rotation_degrees, scale) from a partial-affine fit, or None."""
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(src, None)
    k2, d2 = orb.detectAndCompute(dst, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None

    p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < 10:
        return None

    scale = float(np.hypot(M[0, 0], M[1, 0]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return angle, scale


def estimate_photometric(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares fit  dst ~= a * src + b  on matched-size images.
    `a` is the contrast gain, `b` the brightness offset (0-255 scale).
    """
    if src.shape != dst.shape:
        dst = cv2.resize(dst, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    x = src.astype(np.float64).ravel()
    y = dst.astype(np.float64).ravel()
    # exclude clipped pixels, which bias the fit
    keep = (x > 3) & (x < 252) & (y > 3) & (y < 252)
    if keep.sum() < 500:
        keep = np.ones_like(x, dtype=bool)
    x, y = x[keep], y[keep]
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def estimate_blur_sigma(src: np.ndarray, dst: np.ndarray,
                        sigmas: np.ndarray) -> Optional[float]:
    """Find the Gaussian sigma whose blurred source best matches the derivative."""
    if src.shape != dst.shape:
        dst = cv2.resize(dst, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    d = dst.astype(np.float32)
    best, best_err = None, np.inf
    for s in sigmas:
        k = int(2 * round(3 * s) + 1)
        blurred = cv2.GaussianBlur(src.astype(np.float32), (k, k), sigmaX=float(s))
        err = float(np.mean((blurred - d) ** 2))
        if err < best_err:
            best_err, best = err, float(s)
    # reject if even the best fit is poor
    base = float(np.mean((src.astype(np.float32) - d) ** 2))
    if best_err > 0.9 * base:
        return None
    return best


def check_flip(src: np.ndarray, dst: np.ndarray) -> Optional[bool]:
    """True if dst is closer to a horizontally flipped src than to src itself."""
    if src.shape != dst.shape:
        dst = cv2.resize(dst, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    a = float(np.mean(np.abs(src.astype(np.float32) - dst.astype(np.float32))))
    b = float(np.mean(np.abs(np.fliplr(src).astype(np.float32) - dst.astype(np.float32))))
    return b < a


# =============================================================================
# MAIN
# =============================================================================

def index_train(data_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    tr = data_root / "train"
    for cd in sorted(p for p in tr.iterdir() if p.is_dir()):
        for fp in sorted(cd.iterdir()):
            if fp.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append({
                "class_name": cd.name,
                "filename": fp.name,
                "filepath": str(fp),
                "canonical_id": canonical_stem(fp.name),
                "op": op_of(fp.name),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="dedup/aug_params.csv")
    ap.add_argument("--max-per-op", type=int, default=400,
                    help="Sample cap per operation (0 = all).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    df = index_train(root)
    parents = df[df["op"].isna()].set_index(["class_name", "canonical_id"])["filepath"].to_dict()
    derived = df[df["op"].notna()].copy()

    print(f"train originals : {int(df['op'].isna().sum())}")
    print(f"train derivatives: {len(derived)}")
    print(derived["op"].value_counts().to_string())

    if args.max_per_op > 0:
        derived = (derived.groupby("op", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), args.max_per_op),
                                             random_state=args.seed)))
    print(f"\nestimating on {len(derived)} derivatives\n")

    sigmas = np.arange(0.4, 6.05, 0.1)
    recs: List[Dict] = []
    missing_parent = 0

    for _, r in tqdm(derived.iterrows(), total=len(derived), desc="Estimate"):
        pp = parents.get((r["class_name"], r["canonical_id"]))
        if pp is None:
            missing_parent += 1
            continue
        src, dst = load_gray(Path(pp)), load_gray(Path(r["filepath"]))
        if src is None or dst is None:
            continue

        rec = {"class_name": r["class_name"], "op": r["op"],
               "derivative": r["filename"], "source": Path(pp).name}
        op = r["op"]

        if op in ("rotation", "zoom", "crop"):
            aff = estimate_affine(src, dst)
            if aff is None:
                rec["status"] = "affine_failed"
                recs.append(rec); continue
            rec["angle_deg"], rec["scale"] = aff
        elif op in ("brightness", "contrast", "color"):
            a, b = estimate_photometric(src, dst)
            rec["gain"], rec["offset"] = a, b
        elif op == "blur":
            s = estimate_blur_sigma(src, dst, sigmas)
            if s is None:
                rec["status"] = "blur_fit_failed"
                recs.append(rec); continue
            rec["sigma"] = s
        elif op == "flip":
            rec["is_hflip"] = check_flip(src, dst)

        rec["status"] = "ok"
        recs.append(rec)

    out = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 74)
    print("RECOVERED PARAMETERS")
    print("=" * 74)
    if missing_parent:
        print(f"derivatives with no locatable source: {missing_parent}")
    bad = out[out["status"] != "ok"]
    if len(bad):
        print("estimation failures:")
        print(bad.groupby(["op", "status"]).size().to_string(), "\n")

    ok = out[out["status"] == "ok"]

    def summarise(op: str, col: str, label: str, absolute: bool = False) -> None:
        v = ok.loc[ok["op"] == op, col].dropna()
        if v.empty:
            return
        w = v.abs() if absolute else v
        print(f"\n{op}  ({label}, n={len(v)})")
        print(f"   min {w.min():+.3f}   p5 {w.quantile(.05):+.3f}   "
              f"median {w.median():+.3f}   p95 {w.quantile(.95):+.3f}   "
              f"max {w.max():+.3f}")
        if not absolute and (v < 0).any() and (v > 0).any():
            print(f"   signed range: [{v.min():+.2f}, {v.max():+.2f}]")

    summarise("rotation", "angle_deg", "degrees")
    summarise("zoom", "scale", "scale factor")
    summarise("crop", "scale", "scale factor")
    summarise("brightness", "offset", "intensity offset, 0-255")
    summarise("brightness", "gain", "gain")
    summarise("contrast", "gain", "gain")
    summarise("color", "gain", "gain")
    summarise("blur", "sigma", "Gaussian sigma, px")

    f = ok.loc[ok["op"] == "flip", "is_hflip"].dropna()
    if len(f):
        print(f"\nflip  (n={len(f)}): horizontal in {100 * f.mean():.1f}% of samples")

    print(f"\nper-image estimates written to {args.out}")
    print("\nReport these as ranges measured from the released derivatives.")


if __name__ == "__main__":
    main()
