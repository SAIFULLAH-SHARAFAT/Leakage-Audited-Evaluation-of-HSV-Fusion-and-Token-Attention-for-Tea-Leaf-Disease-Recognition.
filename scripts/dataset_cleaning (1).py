#!/usr/bin/env python3
"""
build_tea_leaf_dataset.py

STAGES
------
  1  index and validate the source layout
  2  verify every image is readable
  3  copy to a writable working directory
  4  remove exact-duplicate and source-family leakage from train
  5  compute perceptual signals (pHash + ResNet-50 embeddings)
  6  link near-duplicate pairs, both signals required to agree
  7  prune the evaluation-side image of every cross-partition pair
  8  materialise the final dataset
  9  re-audit from scratch: exact, family, derivative, perceptual
 10  recover offline augmentation parameters by measurement
 11  write manifests and a SHA-256 fingerprint

USAGE
-----
    pip install imagehash opencv-python-headless
    python build_tea_leaf_dataset.py

    # verify a rebuild reproduces the same partition
    python build_tea_leaf_dataset.py --expect-manifest-sha256 <sha>

    # write verification montages for the threshold choice
    python build_tea_leaf_dataset.py --montages

OUTPUTS (in --manifests)
------------------------
    final_split_manifest.csv     every instance: split, class, file, MD5, ids
    final_fingerprint.json       SHA-256, counts, every parameter used
    pruned_images.csv            removed images, with the reason for each
    final_linked_pairs.csv       all detected near-duplicate pairs
    aug_params.csv               per-image recovered augmentation parameters
    final_decontam_index.json    loader order and labels for val and test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# =============================================================================
# CONSTANTS
# =============================================================================

CLASSES: Tuple[str, ...] = (
    "Brown Blight", "Gray Blight", "Green mirid bug", "Healthy leaf",
    "Helopeltis", "Red spider", "Tea algal leaf spot",
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

AUG_PATTERN = re.compile(
    r"(_aug(_|-)?(rotation|contrast|zoom|flip|brightness|blur|noise|color|crop|.*))$",
    re.IGNORECASE,
)
OP_RE = re.compile(r"_aug[_-]?([a-z]+)", re.IGNORECASE)

# Priority for prune: lower is kept. Train is never removed.
PRIORITY = {"train": 0, "test": 1, "val": 2}

SEED = 42


def canonical_stem(fn: str) -> str:
    return AUG_PATTERN.sub("", Path(fn).stem)


def is_derivative(fn: str) -> bool:
    return bool(re.search(r"_aug", fn, re.IGNORECASE))


def op_of(fn: str) -> Optional[str]:
    m = OP_RE.search(fn)
    return m.group(1).lower() if m else None


def md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fail(msg: str) -> None:
    print(f"\n*** FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# INDEXING
# =============================================================================

def index_dataset(root: Path, require_all: bool = True) -> pd.DataFrame:
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        sd = root / split
        if not sd.exists():
            if require_all:
                fail(f"missing split directory: {sd}")
            continue
        for ci, class_name in enumerate(CLASSES):
            cd = sd / class_name
            if not cd.exists():
                if require_all:
                    fail(f"missing class directory: {cd}")
                continue
            files: List[Path] = []
            for ext in IMAGE_EXTS:
                files.extend(cd.glob(f"*{ext}"))
                files.extend(cd.glob(f"*{ext.upper()}"))
            for fp in sorted(set(files)):
                rows.append({
                    "split": split,
                    "class_name": class_name,
                    "class_idx": ci,
                    "filename": fp.name,
                    "filepath": str(fp),
                    "canonical_id": canonical_stem(fp.name),
                    "is_derived": is_derivative(fp.name),
                    "op": op_of(fp.name),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        fail(f"no images found under {root}")
    return df


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.groupby(["class_name", "split"]).size()
         .unstack(fill_value=0).reindex(list(CLASSES)))
    for c in ("train", "val", "test"):
        if c not in t.columns:
            t[c] = 0
    t = t[["train", "val", "test"]]
    t["total"] = t.sum(axis=1)
    return t


# =============================================================================
# STAGE 1-2  VALIDATE
# =============================================================================

def stage_validate(src: Path) -> None:
    banner("STAGE 1  Validate source layout")
    if not src.exists():
        print(f"--src not found: {src}\nSearching /kaggle/input ...")
        for c in list(Path("/kaggle/input").rglob("mandely_source"))[:20]:
            print("  candidate:", c)
        fail("source directory not found")
    for split in ("train", "val", "test"):
        sp = src / split
        if not sp.exists():
            fail(f"missing split directory: {sp}")
        found = sorted(p.name for p in sp.iterdir() if p.is_dir())
        missing = [c for c in CLASSES if c not in found]
        if missing:
            fail(f"missing classes in {split}: {missing}")
        extra = [c for c in found if c not in CLASSES]
        if extra:
            print(f"  WARNING: extra folders in {split}: {extra}")
        print(f"  {split}: OK ({len(found)} class folders)")


def stage_readable(df: pd.DataFrame, man: Path) -> None:
    banner("STAGE 2  Verify readability")
    bad: List[Dict] = []
    for fp in tqdm(df["filepath"].tolist(), desc="Verify"):
        try:
            with Image.open(fp) as im:
                ImageOps.exif_transpose(im)
                im.verify()
        except Exception as exc:  # noqa: BLE001
            bad.append({"filepath": fp, "error": str(exc)})
    print(f"Unreadable images: {len(bad)}")
    if bad:
        pd.DataFrame(bad).to_csv(man / "unreadable_images.csv", index=False)
        for b in bad[:10]:
            print("  ", b["filepath"], "->", b["error"])
        fail("unreadable images present")


# =============================================================================
# STAGE 3-4  COPY AND REMOVE EXACT / FAMILY LEAKAGE
# =============================================================================

def stage_copy(src: Path, work: Path) -> None:
    banner("STAGE 3  Copy to writable directory")
    if work.exists():
        print(f"{work} exists -- reusing. Delete it for a guaranteed-fresh build.")
        return
    print(f"Copying {src} -> {work} ...")
    shutil.copytree(src, work)
    print("Copy complete.")


def stage_hash_prune(work: Path, man: Path) -> int:
    banner("STAGE 4  Remove exact-duplicate and source-family leakage")
    df = index_dataset(work)
    df["md5"] = [md5(p) for p in tqdm(df["filepath"].tolist(), desc="MD5")]

    vt = df[df["split"].isin(["val", "test"])]
    tr = df[df["split"] == "train"]

    exact = tr[tr["md5"].isin(set(vt["md5"]))]
    vt_keys = set(zip(vt["class_name"], vt["canonical_id"]))
    mask = pd.Series(
        [k in vt_keys for k in zip(tr["class_name"], tr["canonical_id"])],
        index=tr.index)
    family = tr[mask]

    to_delete = sorted(set(exact["filepath"]) | set(family["filepath"]))
    print(f"Exact MD5 leaks in train        : {len(exact)}")
    print(f"Canonical-family leaks in train : {len(family)}")
    print(f"Unique files to delete          : {len(to_delete)}")

    pd.DataFrame({"filepath": to_delete, "stage": "hash_family"}).to_csv(
        man / "stage4_deleted.csv", index=False)

    n = 0
    for fp in to_delete:
        if Path(fp).exists():
            Path(fp).unlink()
            n += 1
    print(f"Deleted {n} files.")
    return n


# =============================================================================
# STAGE 5  PERCEPTUAL SIGNALS
# =============================================================================

def compute_phash(paths: List[str]) -> np.ndarray:
    try:
        import imagehash
    except ImportError:
        fail("imagehash not installed:  pip install imagehash")
    out = []
    for p in tqdm(paths, desc="pHash"):
        with Image.open(p) as im:
            out.append(imagehash.phash(im).hash.flatten())
    return np.asarray(out, dtype=bool)


def compute_embeddings(paths: List[str], batch_size: int, model_name: str) -> np.ndarray:
    try:
        import torch
        import timm
    except ImportError:
        fail("torch and timm required for the embedding signal")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  embedding model: {model_name} on {device}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Embed"):
            ims = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    ims.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(ims).to(device)).float().cpu().numpy())
    F = np.concatenate(feats, 0)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return F


# =============================================================================
# STAGE 6  LINK PAIRS
# =============================================================================

def link_pairs(df: pd.DataFrame, H: np.ndarray, E: np.ndarray,
               phash_max: int, cosine_min: float, link_mode: str) -> pd.DataFrame:
    """Within-class candidate near-duplicate pairs. df must be 0..n-1 indexed."""
    out: List[Dict] = []
    for class_name in df["class_name"].unique():
        idx = df.index[df["class_name"] == class_name].to_numpy()
        if len(idx) < 2:
            continue
        h, e = H[idx], E[idx]
        D = (h[:, None, :] != h[None, :, :]).sum(-1)
        C = e @ e.T
        ph_ok, em_ok = D <= phash_max, C >= cosine_min
        link = {"both": ph_ok & em_ok, "any": ph_ok | em_ok,
                "phash": ph_ok, "embed": em_ok}.get(link_mode)
        if link is None:
            fail(f"unknown link mode: {link_mode}")
        iu = np.triu_indices(len(idx), k=1)
        sel = link[iu]
        for a, b, d, c in zip(iu[0][sel], iu[1][sel], D[iu][sel], C[iu][sel]):
            out.append({"class_name": class_name,
                        "i": int(idx[a]), "j": int(idx[b]),
                        "file_i": df["filename"].iloc[idx[a]],
                        "file_j": df["filename"].iloc[idx[b]],
                        "split_i": df["split"].iloc[idx[a]],
                        "split_j": df["split"].iloc[idx[b]],
                        "hamming": int(d), "cosine": float(c)})
    return pd.DataFrame(out, columns=["class_name", "i", "j", "file_i", "file_j",
                                      "split_i", "split_j", "hamming", "cosine"])


def cross_class_screen(df: pd.DataFrame, E: np.ndarray, cosine_min: float,
                       limit: int = 500) -> pd.DataFrame:
    C = E @ E.T
    np.fill_diagonal(C, -1.0)
    cls = df["class_name"].to_numpy()
    ii, jj = np.where(np.triu(C >= cosine_min, k=1))
    out = []
    for a, b in zip(ii, jj):
        if cls[a] != cls[b]:
            out.append({"class_a": cls[a], "file_a": df["filename"].iloc[a],
                        "class_b": cls[b], "file_b": df["filename"].iloc[b],
                        "cosine": float(C[a, b])})
            if len(out) >= limit:
                break
    return pd.DataFrame(out)


# =============================================================================
# STAGE 7  PRUNE
# =============================================================================

def select_prunes(orig: pd.DataFrame, pairs: pd.DataFrame,
                  keep_val_test: bool) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["index", "split", "class_name",
                                     "filename", "reason"])
    split = orig["split"].to_dict()

    def severity(r) -> int:
        s = {split[r["i"]], split[r["j"]]}
        if s == {"train", "test"}:
            return 0
        if s == {"train", "val"}:
            return 1
        if s == {"val", "test"}:
            return 2
        return 3

    pr = pairs.copy()
    pr["severity"] = pr.apply(severity, axis=1)
    pr = pr[pr["severity"] < 3].sort_values(["severity", "hamming"])

    removed, reasons = set(), {}
    for _, r in pr.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if i in removed or j in removed:
            continue
        si, sj = split[i], split[j]
        if si == sj:
            continue
        if {si, sj} == {"val", "test"} and keep_val_test:
            continue
        drop = i if PRIORITY[si] > PRIORITY[sj] else j
        other = j if drop == i else i
        removed.add(drop)
        reasons[drop] = (f"near-duplicate of {split[other]} image "
                         f"{orig['filename'].loc[other]} "
                         f"(Hamming {int(r['hamming'])}, cosine {r['cosine']:.3f})")

    return pd.DataFrame([{
        "index": k,
        "split": orig["split"].loc[k],
        "class_name": orig["class_name"].loc[k],
        "filename": orig["filename"].loc[k],
        "reason": reasons[k],
    } for k in sorted(removed)])


# =============================================================================
# STAGE 10  RECOVER AUGMENTATION PARAMETERS
# =============================================================================

def recover_aug_params(out_root: Path, max_per_op: int, seed: int) -> pd.DataFrame:
    try:
        import cv2
    except ImportError:
        print("  opencv not installed; skipping parameter recovery "
              "(pip install opencv-python-headless)")
        return pd.DataFrame()

    MAXDIM = 512

    def load_gray(p) -> Optional[np.ndarray]:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        h, w = im.shape
        s = MAXDIM / max(h, w)
        if s < 1.0:
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return im

    def affine(src, dst):
        orb = cv2.ORB_create(nfeatures=2000)
        k1, d1 = orb.detectAndCompute(src, None)
        k2, d2 = orb.detectAndCompute(dst, None)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in raw if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return None
        p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < 10:
            return None
        return (float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                float(np.hypot(M[0, 0], M[1, 0])))

    def photometric(src, dst):
        if src.shape != dst.shape:
            dst = cv2.resize(dst, (src.shape[1], src.shape[0]),
                             interpolation=cv2.INTER_AREA)
        x = src.astype(np.float64).ravel()
        y = dst.astype(np.float64).ravel()
        keep = (x > 3) & (x < 252) & (y > 3) & (y < 252)
        if keep.sum() < 500:
            keep = np.ones_like(x, dtype=bool)
        a, b = np.polyfit(x[keep], y[keep], 1)
        return float(a), float(b)

    def blur_sigma(src, dst, sigmas):
        if src.shape != dst.shape:
            dst = cv2.resize(dst, (src.shape[1], src.shape[0]),
                             interpolation=cv2.INTER_AREA)
        d = dst.astype(np.float32)
        base = float(np.mean((src.astype(np.float32) - d) ** 2))
        best, best_err = None, np.inf
        for s in sigmas:
            k = int(2 * round(3 * s) + 1)
            bl = cv2.GaussianBlur(src.astype(np.float32), (k, k), sigmaX=float(s))
            err = float(np.mean((bl - d) ** 2))
            if err < best_err:
                best_err, best = err, float(s)
        return None if best_err > 0.9 * base else best

    def flip_axis(src, dst):
        if src.shape != dst.shape:
            dst = cv2.resize(dst, (src.shape[1], src.shape[0]),
                             interpolation=cv2.INTER_AREA)
        d = dst.astype(np.float32)
        cand = {"identity": src, "hflip": np.fliplr(src), "vflip": np.flipud(src),
                "rot180": np.flipud(np.fliplr(src))}
        errs = {k: float(np.mean(np.abs(v.astype(np.float32) - d)))
                for k, v in cand.items()}
        return min(errs, key=errs.get)

    df = index_dataset(out_root)
    tr = df[df["split"] == "train"]
    parents = (tr[~tr["is_derived"]]
               .set_index(["class_name", "canonical_id"])["filepath"].to_dict())
    deriv = tr[tr["is_derived"]].copy()
    if deriv.empty:
        return pd.DataFrame()

    if max_per_op > 0:
        rng = np.random.default_rng(seed)
        picks = []
        for _, g in deriv.groupby("op"):
            if len(g) <= max_per_op:
                picks.append(g)
            else:
                picks.append(g.iloc[rng.choice(len(g), max_per_op, replace=False)])
        deriv = pd.concat(picks, ignore_index=True)

    sigmas = np.arange(0.1, 3.05, 0.05)
    recs: List[Dict] = []
    for _, r in tqdm(deriv.iterrows(), total=len(deriv), desc="Params"):
        pp = parents.get((r["class_name"], r["canonical_id"]))
        if pp is None:
            continue
        src, dst = load_gray(pp), load_gray(r["filepath"])
        if src is None or dst is None:
            continue
        rec = {"class_name": r["class_name"], "op": r["op"],
               "derivative": r["filename"], "source": Path(pp).name,
               "status": "ok"}
        op = r["op"]
        if op in ("rotation", "zoom", "crop"):
            a = affine(src, dst)
            if a is None:
                rec["status"] = "affine_failed"
            else:
                rec["angle_deg"], rec["scale"] = a
        elif op in ("brightness", "contrast", "color"):
            g, b = photometric(src, dst)
            rec["gain"], rec["offset"] = g, b
        elif op == "blur":
            s = blur_sigma(src, dst, sigmas)
            if s is None:
                rec["status"] = "blur_fit_failed"
            else:
                rec["sigma"] = s
        elif op == "flip":
            rec["flip_axis"] = flip_axis(src, dst)
        recs.append(rec)

    out = pd.DataFrame(recs)
    ok = out[out["status"] == "ok"]

    print("\n  recovered parameter ranges (5th-95th percentile):")

    def rng(op, col, label, absolute=False):
        v = ok.loc[ok["op"] == op, col].dropna() if col in ok.columns else pd.Series(dtype=float)
        if v.empty:
            return
        w = v.abs() if absolute else v
        print(f"    {op:<11} {label:<22} "
              f"[{w.quantile(.05):+.3f}, {w.quantile(.95):+.3f}]  n={len(v)}")

    rng("rotation", "angle_deg", "angle, degrees", absolute=True)
    rng("zoom", "scale", "scale factor")
    rng("brightness", "gain", "gain")
    rng("contrast", "gain", "gain")
    rng("blur", "sigma", "Gaussian sigma, px")
    if "flip_axis" in ok.columns:
        fa = ok.loc[ok["op"] == "flip", "flip_axis"].dropna()
        if len(fa):
            share = (fa.value_counts(normalize=True) * 100).round(1).to_dict()
            print(f"    flip        axis distribution      {share}  n={len(fa)}")
    bad = out[out["status"] != "ok"]
    if len(bad):
        print("\n  estimation failures:")
        print("   ", bad.groupby(["op", "status"]).size().to_dict())
    return out


# =============================================================================
# MONTAGES (optional verification)
# =============================================================================

def write_montages(orig: pd.DataFrame, H: np.ndarray, E: np.ndarray,
                   man: Path, cosine_min: float, link_mode: str, n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for band in [(0, 2), (3, 5), (6, 8)]:
        pairs = link_pairs(orig, H, E, band[1], cosine_min, link_mode)
        sub = pairs[pairs["hamming"].between(*band)] if not pairs.empty else pairs
        if sub.empty:
            continue
        s = sub.sample(min(n, len(sub)), random_state=0)
        fig, ax = plt.subplots(len(s), 2, figsize=(6, 3 * len(s)))
        if len(s) == 1:
            ax = np.array([ax])
        for k, (_, r) in enumerate(s.iterrows()):
            for col, key in enumerate(("i", "j")):
                row = orig.loc[r[key]]
                with Image.open(row["filepath"]) as im:
                    ax[k, col].imshow(im.convert("RGB"))
                ax[k, col].set_title(f"{row['split']}/{row['filename'][:26]}", fontsize=6)
                ax[k, col].axis("off")
        fig.suptitle(f"Hamming {band[0]}-{band[1]} ({link_mode}) "
                     f"-- same specimen?", fontsize=9)
        plt.tight_layout()
        p = man / f"montage_d{band[0]}{band[1]}.png"
        plt.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", p)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=(""))
    ap.add_argument("--work", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--phash-max", type=int, default=5)
    ap.add_argument("--cosine-min", type=float, default=0.92)
    ap.add_argument("--link-mode", default="both",
                    choices=["both", "any", "phash", "embed"])
    ap.add_argument("--cross-class-cosine", type=float, default=0.97)
    ap.add_argument("--keep-val-test-dups", action="store_true")
    ap.add_argument("--embed-model", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-op", type=int, default=0,
                    help="Cap on derivatives sampled per operation (0 = all)")
    ap.add_argument("--montages", action="store_true")
    ap.add_argument("--montage-n", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--expect-manifest-sha256", default=None)
    args = ap.parse_args()

    t0 = time.time()
    src, work = Path(args.src), Path(args.work)
    out, man = Path(args.out), Path(args.manifests)
    man.mkdir(parents=True, exist_ok=True)

    banner("BUILD TeaLeafBD AUDITED DATASET")
    print(f"src       : {src}")
    print(f"working   : {work}")
    print(f"out       : {out}")
    print(f"manifests : {man}")
    print(f"operating point: {args.link_mode}, "
          f"pHash<={args.phash_max}, cosine>={args.cosine_min}")

    # ---------------------------------------------------------------- 1, 2, 3
    stage_validate(src)
    df_src = index_dataset(src)
    print(f"\nSource images: {len(df_src)}")
    print(counts_table(df_src).to_string())
    stage_readable(df_src, man)
    stage_copy(src, work)

    # ------------------------------------------------------------------- 4
    stage_hash_prune(work, man)

    # ------------------------------------------------------------------- 5
    banner("STAGE 5  Perceptual signals")
    df = index_dataset(work)
    orig = df[~df["is_derived"]].reset_index(drop=True)
    print(f"Originals: {len(orig)}   Derivatives: {int(df['is_derived'].sum())}")
    H = compute_phash(orig["filepath"].tolist())
    E = compute_embeddings(orig["filepath"].tolist(), args.batch_size, args.embed_model)
    np.save(man / "phash_bits.npy", H)
    np.save(man / "embeddings.npy", E)

    # ------------------------------------------------------------------- 6
    banner("STAGE 6  Link near-duplicate pairs")
    pairs = link_pairs(orig, H, E, args.phash_max, args.cosine_min, args.link_mode)
    cross = pairs[pairs["split_i"] != pairs["split_j"]] if not pairs.empty else pairs
    print(f"Linked pairs      : {len(pairs)}")
    print(f"  cross-partition : {len(cross)}")
    print(f"  within-partition: {len(pairs) - len(cross)}  (kept; not leakage)")
    pairs.to_csv(man / "final_linked_pairs.csv", index=False)

    xc = cross_class_screen(orig, E, args.cross_class_cosine)
    print(f"Cross-class pairs at cosine >= {args.cross_class_cosine}: {len(xc)}")
    xc.to_csv(man / "cross_class_pairs.csv", index=False)

    if args.montages:
        banner("Verification montages")
        write_montages(orig, H, E, man, args.cosine_min, args.link_mode,
                       args.montage_n)

    # ------------------------------------------------------------------- 7
    banner("STAGE 7  Prune cross-partition near-duplicates")
    prunes = select_prunes(orig, pairs, args.keep_val_test_dups)
    print(f"Images to remove: {len(prunes)}")
    if len(prunes):
        print(prunes.groupby(["split", "class_name"]).size()
              .unstack(fill_value=0).to_string())
    prunes.to_csv(man / "pruned_images.csv", index=False)

    drop = set(prunes["index"].tolist())
    keep_orig = orig[~orig.index.isin(drop)]
    final = pd.concat([keep_orig, df[df["is_derived"]]], ignore_index=True)

    # ------------------------------------------------------------------- 8
    banner("STAGE 8  Materialise final dataset")
    if out.exists():
        if not args.overwrite:
            fail(f"{out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(final.iterrows(), total=len(final), desc="Copy"):
        shutil.copy2(r["filepath"], out / r["split"] / r["class_name"] / r["filename"])
    print(f"Wrote {len(final)} files")

    # ------------------------------------------------------------------- 9
    banner("STAGE 9  Re-audit the final dataset from scratch")
    df2 = index_dataset(out)
    if len(df2) != len(final):
        fail(f"materialised {len(df2)} files, expected {len(final)}")

    n_deriv_eval = int(df2[(df2["split"] != "train") & df2["is_derived"]].shape[0])
    print(f"(3) derivatives outside train      : {n_deriv_eval}")
    if n_deriv_eval:
        fail("derivatives leaked into val/test")

    df2["md5"] = [md5(p) for p in tqdm(df2["filepath"].tolist(), desc="MD5")]
    vt = set(df2[df2["split"] != "train"]["md5"])
    n_exact = int(df2[(df2["split"] == "train") & df2["md5"].isin(vt)].shape[0])
    print(f"(1) exact cross-split duplicates   : {n_exact}")
    if n_exact:
        fail("exact duplicates remain")

    fam = (df2.groupby(["class_name", "canonical_id"])
           .agg(splits=("split", lambda x: sorted(set(x)))).reset_index())
    n_fam = int(fam["splits"].apply(lambda x: len(x) > 1).sum())
    print(f"(2) cross-split source families    : {n_fam}")
    if n_fam:
        fail("family leakage remains")

    orig2 = df2[~df2["is_derived"]].reset_index(drop=True)
    H2 = compute_phash(orig2["filepath"].tolist())
    E2 = compute_embeddings(orig2["filepath"].tolist(), args.batch_size,
                            args.embed_model)
    pairs2 = link_pairs(orig2, H2, E2, args.phash_max, args.cosine_min,
                        args.link_mode)
    if pairs2.empty:
        n_cross2 = 0
    else:
        c2 = pairs2[pairs2["split_i"] != pairs2["split_j"]]
        if args.keep_val_test_dups and len(c2):
            c2 = c2[[{a, b} != {"val", "test"}
                     for a, b in zip(c2["split_i"], c2["split_j"])]]
        n_cross2 = len(c2)
        if n_cross2:
            c2.to_csv(man / "final_cross_split_pairs.csv", index=False)
    print(f"(4) cross-split near-duplicate pairs: {n_cross2}")
    if n_cross2:
        fail("near-duplicate leakage remains -- see final_cross_split_pairs.csv")
    print("\nAll four audits passed.")

    # ------------------------------------------------------------------ 10
    banner("STAGE 10  Recover offline augmentation parameters")
    aug = recover_aug_params(out, args.max_per_op, SEED)
    if not aug.empty:
        aug.to_csv(man / "aug_params.csv", index=False)
        print(f"\n  wrote {man/'aug_params.csv'}")

    # ------------------------------------------------------------------ 11
    banner("STAGE 11  Manifests and fingerprint")
    counts = counts_table(df2)
    print(counts.to_string())
    n = {s: int((df2["split"] == s).sum()) for s in ("train", "val", "test")}
    n_orig_train = int(df2[(df2["split"] == "train") & ~df2["is_derived"]].shape[0])
    n_deriv = int(df2["is_derived"].sum())
    print(f"\ntrain={n['train']} (orig {n_orig_train} + deriv {n_deriv})  "
          f"val={n['val']}  test={n['test']}  total={len(df2)}")
    counts.to_csv(man / "final_class_counts.csv")

    manifest = (df2[["split", "class_name", "filename", "md5",
                     "canonical_id", "is_derived", "op"]]
                .sort_values(["split", "class_name", "filename"])
                .reset_index(drop=True))
    manifest.to_csv(man / "final_split_manifest.csv", index=False)
    sha = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()

    decontam: Dict = {"classes": list(CLASSES)}
    for split in ("val", "test"):
        order, labels = [], []
        for ci, c in enumerate(sorted(p.name for p in (out / split).iterdir()
                                      if p.is_dir())):
            for fn in sorted(p.name for p in (out / split / c).glob("*.*")):
                order.append(fn)
                labels.append(ci)
        decontam[split] = {
            "order": order, "labels": labels,
            "sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        }
    json.dump(decontam, open(man / "final_decontam_index.json", "w"), indent=1)

    fingerprint = {
        "manifest_sha256": sha,
        "n_total": int(len(df2)),
        "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
        "n_train_original": n_orig_train, "n_derivatives": n_deriv,
        "n_removed_by_perceptual_audit": int(len(prunes)),
        "n_linked_pairs": int(len(pairs)),
        "n_cross_partition_pairs": int(len(cross)),
        "n_cross_class_pairs": int(len(xc)),
        "phash_max": args.phash_max,
        "cosine_min": args.cosine_min,
        "link_mode": args.link_mode,
        "cross_class_cosine": args.cross_class_cosine,
        "embed_model": args.embed_model,
        "keep_val_test_dups": bool(args.keep_val_test_dups),
        "classes": list(CLASSES),
        "source": str(src),
        "audits_passed": {"exact": True, "source_family": True,
                          "derivative_confinement": True,
                          "perceptual_near_duplicate": True},
    }
    json.dump(fingerprint, open(man / "final_fingerprint.json", "w"), indent=2)

    print(f"\nManifest : {man/'final_split_manifest.csv'}  ({len(manifest)} rows)")
    print(f"SHA256   : {sha}")
    if args.expect_manifest_sha256 and sha != args.expect_manifest_sha256:
        fail(f"manifest SHA mismatch\n  expected {args.expect_manifest_sha256}\n"
             f"  got      {sha}")

    banner("DONE")
    print(f"Final dataset : {out}")
    print(f"Manifests     : {man}")
    for f in sorted(man.iterdir()):
        print(f"   {f.name}")
    print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
    print("\n Use for every experiment:")
    print(f"    --data_root {out}")
    print("\nDo NOT pass --strict_clean_counts: split counts differ from the "
          "earlier 6090/851/852 protocol.")


if __name__ == "__main__":
    main()
