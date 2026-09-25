#!/usr/bin/env python3
"""
build_dedup_split.py
====================

Near-duplicate audit and cluster-level re-split for TeaLeafBD.

MOTIVATION
----------
The published 70/15/15 release splits at the level of individual FILES. Tea
leaf datasets contain burst/session captures: several frames of the same
physical specimen, seconds apart, under identical lighting. A file-level
random split distributes those frames across train/val/test, so a model can
be evaluated on near-identical views of leaves it trained on.

Neither MD5 hashing (different bytes) nor filename-family matching (different
source stems) detects this. This script detects it perceptually, then rebuilds
the split so that every perceptual duplicate cluster lies entirely within one
partition.

TWO SIMILARITY SIGNALS
----------------------
  pHash      64-bit perceptual hash, Hamming distance. Cheap, but weak on
             visually homogeneous subjects -- every image here is a green leaf
             on a similar background, so pHash alone produces false positives
             at loose thresholds.
  Embedding  ImageNet-pretrained CNN penultimate features, cosine similarity.
             Far more reliable for this data. Used as the primary signal.

Two images are linked only if BOTH signals agree (--require-both, default) or
if EITHER fires (--link-mode any). Requiring both is conservative in the sense
that it produces fewer, higher-confidence clusters; requiring either is
conservative in the sense that it removes more potential leakage. Choose
deliberately and report the choice.

MODES
-----
  audit   Compute signals, sweep thresholds, write a montage of candidate
          pairs at several distances, and report how many val/test images
          would be affected. USE THIS FIRST and inspect the montage.

  build   Cluster at the chosen thresholds, re-split 70/15/15 over CLUSTERS
          (stratified by class), materialise the new dataset, assert zero
          cross-partition links, and emit a manifest + SHA-256.

USAGE
-----
    # 1. Audit. Inspect audit_montage_*.png before choosing thresholds.
    python build_dedup_split.py audit \
        --src source_dir \
        --artifacts dedup \

    # 2. Build, using thresholds justified by the audit.
    python build_dedup_split.py build \
        --src source_dir \
        --out dedup \
        --artifacts dedup \
        --phash-max 6 --cosine-min 0.92 --seed 42

    # 3. Verify a later rebuild is identical.
    python build_dedup_split.py build ... --expect-manifest-sha256 <sha>

NOTES
-----
* Derivatives (`_aug...` filenames) are attached to their parent's cluster and
  are retained ONLY in the training partition, matching the original protocol.
* Clustering is within-class. Cross-class near-duplicates are reported
  separately: they would indicate a labelling problem, not a split problem.
* Every threshold, seed and count is written to disk so the split is
  reproducible and auditable by a reviewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

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

SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, over CLUSTERS


def canonical_stem(filename: str) -> str:
    return AUG_PATTERN.sub("", Path(filename).stem)


def is_derivative(filename: str) -> bool:
    return bool(re.search(r"_aug", filename, re.IGNORECASE))


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

def index_dataset(src: Path) -> pd.DataFrame:
    """Index every image under src/{train,val,test}/<class>/."""
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        sd = src / split
        if not sd.exists():
            continue
        for class_name in CLASSES:
            cd = sd / class_name
            if not cd.exists():
                continue
            files: List[Path] = []
            for ext in IMAGE_EXTS:
                files.extend(cd.glob(f"*{ext}"))
                files.extend(cd.glob(f"*{ext.upper()}"))
            for fp in sorted(set(files)):
                rows.append({
                    "orig_split": split,
                    "class_name": class_name,
                    "filename": fp.name,
                    "filepath": str(fp),
                    "canonical_id": canonical_stem(fp.name),
                    "is_derived": is_derivative(fp.name),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        fail(f"no images found under {src}")
    return df


# =============================================================================
# SIGNALS
# =============================================================================

def compute_phash(paths: List[str]) -> np.ndarray:
    """Return (N, 64) bool array of perceptual hash bits."""
    try:
        import imagehash
    except ImportError:
        fail("imagehash not installed. Run: pip install imagehash")
    bits = []
    for p in tqdm(paths, desc="pHash"):
        with Image.open(p) as im:
            bits.append(imagehash.phash(im).hash.flatten())
    return np.asarray(bits, dtype=bool)


def compute_embeddings(paths: List[str], batch_size: int = 64,
                       model_name: str = "resnet50") -> np.ndarray:
    """Return (N, D) L2-normalised penultimate features."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)

    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc=f"Embed({model_name})"):
            batch = paths[i:i + batch_size]
            ims = []
            for p in batch:
                with Image.open(p) as im:
                    ims.append(tf(im.convert("RGB")))
            x = torch.stack(ims).to(device)
            f = model(x).float().cpu().numpy()
            feats.append(f)
    F = np.concatenate(feats, 0)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return F


# =============================================================================
# PAIR FINDING
# =============================================================================

def within_class_pairs(df: pd.DataFrame, H: np.ndarray, E: np.ndarray,
                       phash_max: int, cosine_min: float,
                       link_mode: str) -> pd.DataFrame:
    """
    Find candidate near-duplicate pairs within each class.

    link_mode:
        "both"  -> pHash distance <= phash_max AND cosine >= cosine_min
        "any"   -> pHash distance <= phash_max OR  cosine >= cosine_min
        "phash" -> pHash only
        "embed" -> embedding only
    """
    out: List[Dict] = []
    for class_name in df["class_name"].unique():
        idx = df.index[df["class_name"] == class_name].to_numpy()
        if len(idx) < 2:
            continue
        h, e = H[idx], E[idx]

        D = (h[:, None, :] != h[None, :, :]).sum(-1)      # Hamming
        C = e @ e.T                                       # cosine

        ph_ok = D <= phash_max
        em_ok = C >= cosine_min
        if link_mode == "both":
            link = ph_ok & em_ok
        elif link_mode == "any":
            link = ph_ok | em_ok
        elif link_mode == "phash":
            link = ph_ok
        elif link_mode == "embed":
            link = em_ok
        else:
            fail(f"unknown link_mode: {link_mode}")

        iu = np.triu_indices(len(idx), k=1)
        sel = link[iu]
        for a, b, d, c in zip(iu[0][sel], iu[1][sel], D[iu][sel], C[iu][sel]):
            out.append({
                "class_name": class_name,
                "i": int(idx[a]), "j": int(idx[b]),
                "hamming": int(d), "cosine": float(c),
            })
    return pd.DataFrame(out)


def cross_class_pairs(df: pd.DataFrame, E: np.ndarray,
                      cosine_min: float, limit: int = 200) -> pd.DataFrame:
    """Near-duplicates spanning different classes -> possible label problem."""
    out: List[Dict] = []
    C = E @ E.T
    np.fill_diagonal(C, -1.0)
    cls = df["class_name"].to_numpy()
    ii, jj = np.where(np.triu(C >= cosine_min, k=1))
    for a, b in zip(ii, jj):
        if cls[a] != cls[b]:
            out.append({
                "class_a": cls[a], "file_a": df["filename"].iloc[a],
                "class_b": cls[b], "file_b": df["filename"].iloc[b],
                "cosine": float(C[a, b]),
            })
            if len(out) >= limit:
                break
    return pd.DataFrame(out)


# =============================================================================
# CLUSTERING
# =============================================================================

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_clusters(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """
    Connected components over (a) perceptual pair edges and
    (b) canonical-family edges (a derivative shares its parent's cluster).
    """
    dsu = DSU(len(df))
    pos = {ix: k for k, ix in enumerate(df.index)}

    for _, r in pairs.iterrows():
        dsu.union(pos[r["i"]], pos[r["j"]])

    fam: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for k, (cn, cid) in enumerate(zip(df["class_name"], df["canonical_id"])):
        fam[(cn, cid)].append(k)
    for members in fam.values():
        for m in members[1:]:
            dsu.union(members[0], m)

    roots = [dsu.find(k) for k in range(len(df))]
    remap = {r: i for i, r in enumerate(sorted(set(roots)))}
    return pd.Series([remap[r] for r in roots], index=df.index, name="cluster_id")


# =============================================================================
# SPLITTING
# =============================================================================

def split_clusters(df: pd.DataFrame, seed: int) -> Dict[int, str]:
    """
    Assign whole clusters to train/val/test, stratified by class.

    Clusters are shuffled per class and assigned greedily to whichever
    partition is furthest below its target ORIGINAL-image quota, so class
    balance is preserved even though cluster sizes vary.
    """
    rng = np.random.default_rng(seed)
    assign: Dict[int, str] = {}

    originals = df[~df["is_derived"]]
    for class_name in CLASSES:
        sub = originals[originals["class_name"] == class_name]
        if sub.empty:
            continue
        sizes = sub.groupby("cluster_id").size()
        cl = sizes.index.to_numpy()
        sz = sizes.to_numpy()

        order = rng.permutation(len(cl))
        cl, sz = cl[order], sz[order]
        # Largest clusters first stabilises the greedy balance
        order = np.argsort(-sz, kind="stable")
        cl, sz = cl[order], sz[order]

        total = sz.sum()
        target = {s: f * total for s, f in zip(("train", "val", "test"), SPLIT_FRACS)}
        cur = {"train": 0, "val": 0, "test": 0}

        for c, n in zip(cl, sz):
            deficit = {s: target[s] - cur[s] for s in cur}
            pick = max(deficit, key=deficit.get)
            assign[int(c)] = pick
            cur[pick] += int(n)

    return assign


# =============================================================================
# AUDIT MODE
# =============================================================================

def montage(df: pd.DataFrame, pairs: pd.DataFrame, out_png: Path,
            n: int = 8, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pairs.empty:
        return
    sample = pairs.sample(min(n, len(pairs)), random_state=0)
    fig, ax = plt.subplots(len(sample), 2, figsize=(6, 3 * len(sample)))
    if len(sample) == 1:
        ax = np.array([ax])
    for k, (_, r) in enumerate(sample.iterrows()):
        for col, key in enumerate(("i", "j")):
            row = df.loc[r[key]]
            with Image.open(row["filepath"]) as im:
                ax[k, col].imshow(im.convert("RGB"))
            ax[k, col].set_title(
                f"{row['orig_split']}/{row['filename'][:28]}", fontsize=6)
            ax[k, col].axis("off")
        ax[k, 0].set_ylabel(f"d={r['hamming']} cos={r['cosine']:.3f}", fontsize=6)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()
    print("  wrote", out_png)


def run_audit(args: argparse.Namespace) -> None:
    src, art = Path(args.src), Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    banner("AUDIT  index")
    df = index_dataset(src).reset_index(drop=True)
    print(df.groupby(["orig_split", "is_derived"]).size().to_string())

    orig = df[~df["is_derived"]].reset_index(drop=True)
    print(f"\nOriginal images: {len(orig)}  (derivatives excluded from signals)")

    banner("AUDIT  signals")
    H = compute_phash(orig["filepath"].tolist())
    E = compute_embeddings(orig["filepath"].tolist(),
                           batch_size=args.batch_size, model_name=args.embed_model)
    np.save(art / "phash_bits.npy", H)
    np.save(art / "embeddings.npy", E)
    orig.to_csv(art / "originals_index.csv", index=False)

    banner("AUDIT  threshold sweep")
    print(f"{'phash<=':>8} {'cos>=':>7} {'mode':>6} {'pairs':>8} "
          f"{'x-split pairs':>14} {'x-split v/t imgs':>17}")
    sweep = []
    for mode in ("phash", "embed", "both"):
        for ph in args.sweep_phash:
            for cs in args.sweep_cosine:
                if mode == "phash" and cs != args.sweep_cosine[0]:
                    continue
                if mode == "embed" and ph != args.sweep_phash[0]:
                    continue
                pairs = within_class_pairs(orig, H, E, ph, cs, mode)
                if pairs.empty:
                    sweep.append(dict(mode=mode, phash_max=ph, cosine_min=cs,
                                      n_pairs=0, n_cross=0, n_vt=0))
                    continue
                si = orig["orig_split"].to_numpy()
                xs = pairs[si[pairs["i"]] != si[pairs["j"]]]
                vt = set()
                for _, r in xs.iterrows():
                    for k in ("i", "j"):
                        if si[r[k]] != "train":
                            vt.add(r[k])
                sweep.append(dict(mode=mode, phash_max=ph, cosine_min=cs,
                                  n_pairs=len(pairs), n_cross=len(xs), n_vt=len(vt)))
                print(f"{ph:>8} {cs:>7.2f} {mode:>6} {len(pairs):>8} "
                      f"{len(xs):>14} {len(vt):>17}")
    pd.DataFrame(sweep).to_csv(art / "threshold_sweep.csv", index=False)

    banner("AUDIT  montages -- INSPECT THESE BEFORE CHOOSING THRESHOLDS")
    for ph in args.sweep_phash:
        pairs = within_class_pairs(orig, H, E, ph, args.cosine_min, "phash")
        band = pairs[pairs["hamming"] == ph] if not pairs.empty else pairs
        montage(orig, band, art / f"audit_montage_phash_{ph}.png", n=args.montage_n,
                title=f"pHash Hamming == {ph}  (are these the same specimen?)")

    banner("AUDIT  cross-class near-duplicates (possible label issues)")
    xc = cross_class_pairs(orig, E, cosine_min=args.cross_class_cosine)
    print(f"cross-class pairs at cosine >= {args.cross_class_cosine}: {len(xc)}")
    if len(xc):
        print(xc.head(15).to_string())
    xc.to_csv(art / "cross_class_pairs.csv", index=False)

    print("\nNEXT: open the montages. Pick the LARGEST pHash distance at which")
    print("      pairs are still clearly the same specimen, and a cosine")
    print("      threshold consistent with it. Then run `build`.")


# =============================================================================
# BUILD MODE
# =============================================================================

def run_build(args: argparse.Namespace) -> None:
    src, out, art = Path(args.src), Path(args.out), Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    banner("BUILD  index")
    df = index_dataset(src).reset_index(drop=True)
    orig = df[~df["is_derived"]].reset_index(drop=True)

    banner("BUILD  signals")
    hp, ep = art / "phash_bits.npy", art / "embeddings.npy"
    ip = art / "originals_index.csv"
    reuse = hp.exists() and ep.exists() and ip.exists()
    if reuse:
        cached = pd.read_csv(ip)
        reuse = (len(cached) == len(orig)
                 and (cached["filename"].tolist() == orig["filename"].tolist()))
    if reuse:
        print("Reusing cached signals from audit.")
        H, E = np.load(hp), np.load(ep)
    else:
        H = compute_phash(orig["filepath"].tolist())
        E = compute_embeddings(orig["filepath"].tolist(),
                               batch_size=args.batch_size, model_name=args.embed_model)
        np.save(hp, H); np.save(ep, E)
        orig.to_csv(ip, index=False)

    banner("BUILD  cluster")
    pairs = within_class_pairs(orig, H, E, args.phash_max, args.cosine_min,
                               args.link_mode)
    print(f"Linked pairs ({args.link_mode}, phash<={args.phash_max}, "
          f"cos>={args.cosine_min}): {len(pairs)}")
    pairs.to_csv(art / "linked_pairs.csv", index=False)

    orig["cluster_id"] = build_clusters(orig, pairs)
    sizes = orig.groupby("cluster_id").size()
    print(f"Clusters: {orig['cluster_id'].nunique()} over {len(orig)} originals")
    print("Cluster size distribution:")
    print(sizes.value_counts().sort_index().to_string())
    print(f"Largest cluster: {sizes.max()} images")

    banner("BUILD  split clusters 70/15/15 (stratified by class)")
    assign = split_clusters(orig, seed=args.seed)
    orig["new_split"] = orig["cluster_id"].map(assign)

    tab = (orig.groupby(["class_name", "new_split"]).size()
           .unstack(fill_value=0).reindex(columns=["train", "val", "test"]))
    tab["total"] = tab.sum(axis=1)
    for s in ("train", "val", "test"):
        tab[f"{s}_%"] = (100 * tab[s] / tab["total"]).round(1)
    print(tab.to_string())

    # Derivatives follow their parent's cluster; kept only if that cluster
    # landed in train, matching the original protocol.
    parent = {}
    for cn, cid, ns in zip(orig["class_name"], orig["canonical_id"], orig["new_split"]):
        parent[(cn, cid)] = ns

    deriv = df[df["is_derived"]].copy()
    deriv["new_split"] = [parent.get((cn, cid)) for cn, cid
                          in zip(deriv["class_name"], deriv["canonical_id"])]
    orphan = int(deriv["new_split"].isna().sum())
    if orphan:
        print(f"\nWARNING: {orphan} derivatives have no parent original; dropped.")
    deriv = deriv[deriv["new_split"] == "train"]
    print(f"Derivatives retained (train only): {len(deriv)} of "
          f"{int(df['is_derived'].sum())}")

    final = pd.concat([orig.drop(columns=["cluster_id"]).assign(
                           cluster_id=orig["cluster_id"]),
                       deriv.assign(cluster_id=-1)], ignore_index=True)

    banner("BUILD  materialise dataset")
    if out.exists():
        if not args.overwrite:
            fail(f"{out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(final.iterrows(), total=len(final), desc="Copy"):
        shutil.copy2(r["filepath"], out / r["new_split"] / r["class_name"] / r["filename"])

    banner("BUILD  verify")
    counts = (final.groupby(["class_name", "new_split"]).size()
              .unstack(fill_value=0).reindex(columns=["train", "val", "test"]))
    counts["total"] = counts.sum(axis=1)
    print(counts.to_string())
    n = {s: int((final["new_split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"\ntrain={n['train']}  val={n['val']}  test={n['test']}  "
          f"total={sum(n.values())}")

    assert not final[(final["new_split"] != "train") & final["is_derived"]].shape[0], \
        "derivatives leaked into val/test"

    # No cluster may span partitions
    span = (orig.groupby("cluster_id")["new_split"].nunique() > 1).sum()
    print(f"Clusters spanning partitions: {span}")
    if span:
        fail("cluster split is inconsistent")

    # Re-verify no linked pair spans partitions
    ns = orig["new_split"].to_numpy()
    bad = sum(1 for _, r in pairs.iterrows() if ns[r["i"]] != ns[r["j"]])
    print(f"Linked pairs spanning partitions: {bad}")
    if bad:
        fail("near-duplicate leakage remains")

    # Exact-duplicate sweep on the materialised copy
    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(4096), b""):
                h.update(b)
        return h.hexdigest()
    final["md5"] = [md5(out / r["new_split"] / r["class_name"] / r["filename"])
                    for _, r in tqdm(final.iterrows(), total=len(final), desc="MD5")]
    vt = set(final[final["new_split"] != "train"]["md5"])
    exact = int(final[(final["new_split"] == "train") & final["md5"].isin(vt)].shape[0])
    print(f"Exact cross-split duplicates: {exact}")
    if exact:
        fail("exact duplicates remain")

    banner("BUILD  manifest")
    manifest = (final[["new_split", "class_name", "filename", "md5",
                       "canonical_id", "cluster_id", "is_derived", "orig_split"]]
                .rename(columns={"new_split": "split"})
                .sort_values(["split", "class_name", "filename"])
                .reset_index(drop=True))
    mpath = art / "split_manifest_dedup.csv"
    manifest.to_csv(mpath, index=False)
    sha = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()

    fingerprint = {
        "manifest_sha256": sha,
        "n_rows": int(len(manifest)),
        "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
        "n_clusters": int(orig["cluster_id"].nunique()),
        "link_mode": args.link_mode,
        "phash_max": args.phash_max,
        "cosine_min": args.cosine_min,
        "embed_model": args.embed_model,
        "split_fracs": list(SPLIT_FRACS),
        "seed": args.seed,
        "classes": list(CLASSES),
        "source": str(src),
    }
    with open(art / "manifest_fingerprint_dedup.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(f"Manifest : {mpath}")
    print(f"SHA256   : {sha}")
    if args.expect_manifest_sha256 and sha != args.expect_manifest_sha256:
        fail(f"manifest SHA mismatch\n  expected {args.expect_manifest_sha256}\n"
             f"  got      {sha}")

    banner("DONE")
    print(f"Clean dataset : {out}")
    print(f"Use as        : --data_root {out}")
    print(f"Manifest SHA  : {sha}")
    print("\nNOTE: split counts differ from the original 6090/851/852 protocol.")
    print("      Pass the new counts to training; do NOT use --strict_clean_counts.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(q):
        q.add_argument("--src", type=str, default="")
        q.add_argument("--artifacts", type=str, default="")
        q.add_argument("--batch-size", type=int, default=64)
        q.add_argument("--embed-model", type=str, default="resnet50")

    a = sub.add_parser("audit", help="Sweep thresholds and write montages.")
    common(a)
    a.add_argument("--sweep-phash", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    a.add_argument("--sweep-cosine", type=float, nargs="+",
                   default=[0.85, 0.90, 0.92, 0.95, 0.98])
    a.add_argument("--cosine-min", type=float, default=0.92)
    a.add_argument("--cross-class-cosine", type=float, default=0.97)
    a.add_argument("--montage-n", type=int, default=8)

    b = sub.add_parser("build", help="Cluster, re-split, materialise, verify.")
    common(b)
    b.add_argument("--out", type=str, default="")
    b.add_argument("--phash-max", type=int, required=True)
    b.add_argument("--cosine-min", type=float, required=True)
    b.add_argument("--link-mode", type=str, default="both",
                   choices=["both", "any", "phash", "embed"])
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--expect-manifest-sha256", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        run_audit(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
