#!/usr/bin/env python3
"""
build_clean_dataset.py
======================

Builds the leakage-audited TeaLeafBD working dataset used by every experiment.

  1. Index the 70/15/15 source split by class.
  2. Verify every image is readable.
  3. Compute MD5 hashes for exact-duplicate detection.
  4. Copy the dataset to a writable directory.
  5. Delete any TRAIN image that (a) duplicates a val/test image by MD5, or
     (b) shares a canonical source-image family with a val/test image.
  6. Re-index, assert zero leakage, assert the 6090/851/852 protocol counts.
  7. Emit a split manifest and a SHA-256 fingerprint of that manifest.

AUDIT SCOPE (state this in the paper; do not overclaim):
  The family rule is FILENAME-BASED: an `_aug...` suffix is stripped to recover
  the canonical source stem. Exact MD5 hashing catches byte-identical files;
  family IDs catch derivatives that follow the `_aug` naming convention.
  Neither catches a near-duplicate saved under a non-conforming filename.

USAGE
-----
    python build_clean_dataset.py \
        --src  source_dir \
        --out  output_dir \
        --artifacts audit_dir \

    # Verify a later rebuild reproduces the identical data build:
    python build_clean_dataset.py --src ... --out ... \
        --expect-manifest-sha256 a3f1...

EXIT CODES
----------
    0  success
    1  hard failure (leakage remains, counts wrong, manifest mismatch)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kw):
        return x


# =============================================================================
# CONSTANTS -- must not drift between runs or the dataset identity changes
# =============================================================================

CLASSES: Tuple[str, ...] = (
    "Brown Blight",
    "Gray Blight",
    "Green mirid bug",
    "Healthy leaf",
    "Helopeltis",
    "Red spider",
    "Tea algal leaf spot",
)

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Published protocol counts (effective train / val / test)
EXPECTED_COUNTS: Tuple[int, int, int] = (6090, 851, 852)

# Identical to the regex used in the original project notebooks.
AUG_PATTERN = re.compile(
    r"(_aug(_|-)?(rotation|contrast|zoom|flip|brightness|blur|noise|color|crop|.*))$",
    re.IGNORECASE,
)

SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_md5(path: str | Path, chunk: int = 4096) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_stem(filename: str) -> str:
    """Strip an `_aug...` suffix to recover the canonical source-image stem."""
    return AUG_PATTERN.sub("", Path(filename).stem)


def list_images(split_dir: str | Path, split_name: str) -> pd.DataFrame:
    """Index one split, preserving the canonical class order."""
    rows: List[Dict] = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = Path(split_dir) / class_name
        files: List[Path] = []
        for ext in IMAGE_EXTS:
            files.extend(class_dir.glob(f"*{ext}"))
            files.extend(class_dir.glob(f"*{ext.upper()}"))
        for fp in sorted(set(files)):
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "filepath": str(fp),
                    "filename": fp.name,
                    "suffix": fp.suffix.lower(),
                }
            )
    return pd.DataFrame(rows)


def index_all(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    return pd.concat(
        [list_images(root / s, s) for s in ("train", "val", "test")],
        ignore_index=True,
    )


def class_count_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="count")
        .pivot(index="class_name", columns="split", values="count")
        .fillna(0)
        .astype(int)
    )
    counts = counts.loc[list(CLASSES)]
    for col in ("train", "val", "test"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    return counts


def fail(msg: str) -> None:
    print(f"\n*** FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_validate_source(src: Path) -> None:
    banner("STEP 1  Validate source layout")
    if not src.exists():
        print(f"Configured --src not found: {src}")
        print("Searching /kaggle/input for candidates ...")
        for c in list(Path("/kaggle/input").rglob("tea_leaf_processed_dataset"))[:20]:
            print("  candidate:", c)
        fail("source directory not found")

    for split in ("train", "val", "test"):
        sp = src / split
        if not sp.exists():
            fail(f"missing split directory: {sp}")
        found = sorted(p.name for p in sp.iterdir() if p.is_dir())
        missing = [c for c in CLASSES if c not in found]
        extra = [c for c in found if c not in CLASSES]
        if missing:
            fail(f"missing classes in {split}: {missing}")
        if extra:
            print(f"  WARNING: extra folders in {split}: {extra}")
        print(f"  {split}: OK ({len(found)} class folders)")
    print("\nSource layout valid.")


def step_index_source(src: Path, artifacts: Path) -> pd.DataFrame:
    banner("STEP 2  Index source dataset")
    df = index_all(src)
    print(f"Total source images: {len(df)}")
    counts = class_count_table(df)
    print("\nSource class counts:")
    print(counts.to_string())
    df.to_csv(artifacts / "image_index_source.csv", index=False)
    counts.to_csv(artifacts / "class_counts_source.csv")
    with open(artifacts / "class_to_idx.json", "w") as f:
        json.dump({c: i for i, c in enumerate(CLASSES)}, f, indent=2)
    return df


def step_verify_readable(df: pd.DataFrame, artifacts: Path) -> None:
    banner("STEP 3  Verify image readability")
    bad: List[Dict] = []
    for fp in tqdm(df["filepath"].tolist(), desc="Verify"):
        try:
            with Image.open(fp) as img:
                ImageOps.exif_transpose(img)
                img.verify()
        except Exception as exc:  # noqa: BLE001
            bad.append({"filepath": fp, "error": str(exc)})
    print(f"Unreadable images: {len(bad)}")
    pd.DataFrame(bad).to_csv(artifacts / "unreadable_images.csv", index=False)
    if bad:
        for b in bad[:10]:
            print("  ", b["filepath"], "->", b["error"])
        fail("unreadable images present; resolve before building")


def step_source_duplicate_audit(df: pd.DataFrame, artifacts: Path) -> pd.DataFrame:
    banner("STEP 4  Source duplicate audit (pre-clean, informational)")
    tqdm_iter = tqdm(df["filepath"].tolist(), desc="MD5")
    df = df.copy()
    df["md5"] = [compute_md5(p) for p in tqdm_iter]

    dup = (
        df.groupby("md5")
        .agg(n=("filepath", "count"), splits=("split", lambda x: sorted(set(x))))
        .reset_index()
    )
    dup = dup[dup["n"] > 1]
    cross = dup[dup["splits"].apply(lambda x: len(x) > 1)]
    print(f"Duplicate hash groups      : {len(dup)}")
    print(f"Cross-split duplicate groups: {len(cross)}")
    df.to_csv(artifacts / "image_index_source_md5.csv", index=False)
    cross.to_csv(artifacts / "cross_split_duplicates_source.csv", index=False)
    return df


def step_copy(src: Path, out: Path) -> None:
    banner("STEP 5  Copy to writable directory")
    if out.exists():
        print(f"{out} already exists -- reusing it.")
        print("Delete it first if you want a guaranteed-fresh build.")
        return
    print(f"Copying {src} -> {out} ...")
    shutil.copytree(src, out)
    print("Copy complete.")


def step_remove_leaks(out: Path, artifacts: Path) -> int:
    banner("STEP 6  Remove cross-split leakage from TRAIN")
    records: List[Dict] = []
    for split in ("train", "val", "test"):
        split_dir = out / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for img in class_dir.glob("*.*"):
                if img.suffix.lower() in IMAGE_EXTS:
                    records.append(
                        {
                            "split": split,
                            "class_name": class_dir.name,
                            "filename": img.name,
                            "filepath": str(img),
                        }
                    )

    df = pd.DataFrame(records)
    df["md5"] = [compute_md5(p) for p in tqdm(df["filepath"].tolist(), desc="MD5")]
    df["canonical_id"] = df["filename"].apply(canonical_stem)

    val_test = df[df["split"].isin(["val", "test"])]
    train_df = df[df["split"] == "train"]

    exact_leaks = train_df[train_df["md5"].isin(set(val_test["md5"]))]

    vt_keys = set(zip(val_test["class_name"], val_test["canonical_id"]))
    mask = pd.Series(
        [k in vt_keys for k in zip(train_df["class_name"], train_df["canonical_id"])],
        index=train_df.index,
    )
    family_leaks = train_df[mask]

    to_delete = sorted(set(exact_leaks["filepath"]) | set(family_leaks["filepath"]))
    print(f"Exact MD5 leaks in train        : {len(exact_leaks)}")
    print(f"Canonical-family leaks in train : {len(family_leaks)}")
    print(f"Unique files to delete          : {len(to_delete)}")

    pd.DataFrame({"filepath": to_delete}).to_csv(
        artifacts / "deleted_leaks.csv", index=False
    )

    deleted = 0
    for fp in to_delete:
        if os.path.exists(fp):
            os.remove(fp)
            deleted += 1
    print(f"Deleted {deleted} files.")
    return deleted


def step_verify_clean(out: Path, artifacts: Path, strict: bool) -> pd.DataFrame:
    banner("STEP 7  Re-index and verify the clean build")
    df = index_all(out)
    counts = class_count_table(df)
    print("Clean class counts:")
    print(counts.to_string())

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())
    print(f"\ntrain={n_train}  val={n_val}  test={n_test}")

    if (n_train, n_val, n_test) != EXPECTED_COUNTS:
        msg = (
            f"split counts {(n_train, n_val, n_test)} != published protocol "
            f"{EXPECTED_COUNTS}. Checkpoints from earlier runs are NOT "
            f"comparable to runs on this build."
        )
        if strict:
            fail(msg)
        print(f"\n*** WARNING: {msg}")
    else:
        print("Split counts match the published protocol.")

    print("\nRe-hashing for the final leakage assertion ...")
    df["md5"] = [compute_md5(p) for p in tqdm(df["filepath"].tolist(), desc="MD5")]
    df["canonical_id"] = df["filename"].apply(canonical_stem)

    vt_md5 = set(df[df["split"].isin(["val", "test"])]["md5"])
    exact_remaining = df[(df["split"] == "train") & (df["md5"].isin(vt_md5))]
    print(f"Remaining exact cross-split duplicates: {len(exact_remaining)}")
    if len(exact_remaining):
        fail("exact leakage remains -- do not train on this build")

    fam = (
        df.groupby(["class_name", "canonical_id"])
        .agg(splits=("split", lambda x: sorted(set(x))))
        .reset_index()
    )
    fam_cross = fam[fam["splits"].apply(lambda x: len(x) > 1)]
    print(f"Remaining canonical-family cross-split cases: {len(fam_cross)}")
    if len(fam_cross):
        fam_cross.to_csv(artifacts / "family_leaks_remaining.csv", index=False)
        fail("family leakage remains -- do not train on this build")

    df.to_csv(artifacts / "image_index_clean.csv", index=False)
    counts.to_csv(artifacts / "class_counts_clean.csv")
    print("\nLeakage audit passed.")
    return df


def step_manifest(df: pd.DataFrame, artifacts: Path, expect_sha: str | None) -> str:
    banner("STEP 8  Split manifest and fingerprint")
    manifest = (
        df[["split", "class_name", "filename", "md5", "canonical_id"]]
        .sort_values(["split", "class_name", "filename"])
        .reset_index(drop=True)
    )
    manifest_path = artifacts / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    payload = manifest.to_csv(index=False).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()

    fingerprint = {
        "manifest_sha256": sha,
        "n_rows": int(len(manifest)),
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "classes": list(CLASSES),
        "aug_pattern": AUG_PATTERN.pattern,
        "expected_counts": list(EXPECTED_COUNTS),
    }
    with open(artifacts / "manifest_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(f"Manifest : {manifest_path}")
    print(f"Rows     : {len(manifest)}")
    print(f"SHA256   : {sha}")

    if expect_sha:
        if sha != expect_sha:
            fail(
                f"manifest SHA-256 mismatch.\n"
                f"       expected {expect_sha}\n"
                f"       got      {sha}\n"
                f"       This build differs from the reference. Runs are NOT comparable."
            )
        print("Manifest matches the reference build.")

    return sha


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the leakage-audited TeaLeafBD working dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src",
        type=str,
        default=(),
        help="Source dataset root containing train/ val/ test/.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Writable output root for the clean dataset.",
    )
    p.add_argument(
        "--artifacts",
        type=str,
        default="",
        help="Directory for manifests, indices and audit CSVs.",
    )
    p.add_argument(
        "--expect-manifest-sha256",
        type=str,
        default=None,
        help="If given, fail unless the rebuilt manifest matches this hash.",
    )
    p.add_argument(
        "--skip-source-audit",
        action="store_true",
        help="Skip the informational pre-clean duplicate audit (saves one MD5 pass).",
    )
    p.add_argument(
        "--strict-counts",
        action="store_true",
        help="Fail (not warn) if split counts differ from the published protocol.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    src = Path(args.src)
    out = Path(args.out)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    banner("TeaLeafBD clean-dataset builder")
    print(f"src       : {src}")
    print(f"out       : {out}")
    print(f"artifacts : {artifacts}")

    step_validate_source(src)
    df_src = step_index_source(src, artifacts)
    step_verify_readable(df_src, artifacts)
    if not args.skip_source_audit:
        step_source_duplicate_audit(df_src, artifacts)
    step_copy(src, out)
    step_remove_leaks(out, artifacts)
    df_clean = step_verify_clean(out, artifacts, strict=args.strict_counts)
    sha = step_manifest(df_clean, artifacts, args.expect_manifest_sha256)

    banner("DONE")
    print(f"Clean dataset : {out}")
    print(f"Artifacts     : {artifacts}")
    print(f"Manifest SHA  : {sha}")
    print("\nPass this to training as --data_root:")
    print(f"    --data_root {out}")
    print("\nRecord the manifest SHA. Re-run later sessions with")
    print(f"    --expect-manifest-sha256 {sha}")
    print("to prove the data build is identical.")


if __name__ == "__main__":
    main()
