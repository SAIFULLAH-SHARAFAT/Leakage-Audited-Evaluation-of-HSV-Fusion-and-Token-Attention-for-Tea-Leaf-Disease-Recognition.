#!/usr/bin/env python3
"""
finalize_dataset.py
===================

Produce the FINAL, provably clean TeaLeafBD working dataset.

Takes the leakage-audited `tea_leaf_clean` build and removes the one class of
contamination that byte-hashing and filename-family matching cannot detect:
perceptual near-duplicates (burst/session captures of the same physical
specimen stored as distinct files) that span partitions.

TWO MODES
---------
  prune    (default, recommended)
           Delete the val/test member of every cross-split near-duplicate
           pair. The TRAINING partition is left completely untouched, so
           existing checkpoints remain valid and can simply be rescored on
           the pruned evaluation sets. Split proportions and the published
           training protocol are preserved.

  resplit  Rebuild 70/15/15 over perceptual duplicate CLUSTERS. Maximally
           rigorous, but every partition changes, all existing checkpoints
           become void, and every experiment must be retrained. Use only if
           the measured contamination effect is large.

WHAT COUNTS AS CONTAMINATION
----------------------------
  train <-> val    leakage into checkpoint selection
  train <-> test   leakage into the reported metric
  val   <-> test   not training leakage, but duplicates the evaluation
                   sample; pruned by default (keep test, drop val) so the
                   two evaluation sets are independent. Disable with
                   --keep-val-test-dups.

  Near-duplicates WITHIN a partition are legitimate (redundant training data,
  or repeated evaluation samples inside one set). They are reported but never
  removed.

SIMILARITY SIGNALS
------------------
  pHash      64-bit perceptual hash, Hamming distance.
  Embedding  ImageNet-pretrained CNN features, cosine similarity.

  Two images are linked only when BOTH agree (default). pHash alone produces
  false positives on visually homogeneous subjects; embeddings alone are too
  permissive. Thresholds should be calibrated by visual inspection -- see
  build_dedup_split.py audit.

USAGE
-----
    python finalize_dataset.py \
        --src "source dir" \
        --out "Tea_leaf_dataset" \
        --artifacts "dedup" \
        --mode prune --phash-max 5 --cosine-min 0.92 --link-mode both

    # verify a later rebuild is byte-identical in composition
    python finalize_dataset.py ... --expect-manifest-sha256 <sha>

OUTPUTS
-------
  <out>/{train,val,test}/<class>/*          the final dataset
  <artifacts>/final_split_manifest.csv      per-file split, hash, cluster
  <artifacts>/final_fingerprint.json        SHA-256 + every parameter used
  <artifacts>/pruned_images.csv             what was removed and why
  <artifacts>/final_linked_pairs.csv        all detected near-duplicate pairs
  <artifacts>/final_decontam_index.json     loader order + labels for rescoring
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
from typing import Dict, List, Set, Tuple

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

SPLIT_FRACS = (0.70, 0.15, 0.15)


def canonical_stem(fn: str) -> str:
    return AUG_PATTERN.sub("", Path(fn).stem)


def is_derivative(fn: str) -> bool:
    return bool(re.search(r"_aug", fn, re.IGNORECASE))


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fail(msg: str) -> None:
    print(f"\n*** FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def md5(p: Path | str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            h.update(b)
    return h.hexdigest()


# =============================================================================
# INDEXING
# =============================================================================

def index_dataset(src: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        sd = src / split
        if not sd.exists():
            fail(f"missing split directory: {sd}")
        for class_name in CLASSES:
            cd = sd / class_name
            if not cd.exists():
                fail(f"missing class directory: {cd}")
            files: List[Path] = []
            for ext in IMAGE_EXTS:
                files.extend(cd.glob(f"*{ext}"))
                files.extend(cd.glob(f"*{ext.upper()}"))
            for fp in sorted(set(files)):
                rows.append({
                    "split": split,
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
    try:
        import imagehash
    except ImportError:
        fail("imagehash not installed. Run: pip install imagehash")
    out = []
    for p in tqdm(paths, desc="pHash"):
        with Image.open(p) as im:
            out.append(imagehash.phash(im).hash.flatten())
    return np.asarray(out, dtype=bool)


def compute_embeddings(paths: List[str], batch_size: int, model_name: str) -> np.ndarray:
    import torch
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc=f"Embed({model_name})"):
            ims = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    ims.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(ims).to(device)).float().cpu().numpy())
    F = np.concatenate(feats, 0)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return F


def load_or_compute_signals(orig: pd.DataFrame, art: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    """Reuse cached signals from a prior audit when the index matches exactly."""
    hp, ep, ip = art / "phash_bits.npy", art / "embeddings.npy", art / "originals_index.csv"
    if hp.exists() and ep.exists() and ip.exists():
        cached = pd.read_csv(ip)
        same = (len(cached) == len(orig)
                and cached["filename"].tolist() == orig["filename"].tolist()
                and cached["class_name"].tolist() == orig["class_name"].tolist())
        if same:
            print("Reusing cached signals from the audit stage.")
            return np.load(hp), np.load(ep)
        print("Cached signals do not match this index; recomputing.")
    H = compute_phash(orig["filepath"].tolist())
    E = compute_embeddings(orig["filepath"].tolist(), args.batch_size, args.embed_model)
    np.save(hp, H); np.save(ep, E)
    orig.to_csv(ip, index=False)
    return H, E


# =============================================================================
# PAIRS
# =============================================================================

def within_class_pairs(df: pd.DataFrame, H: np.ndarray, E: np.ndarray,
                       phash_max: int, cosine_min: float, link_mode: str) -> pd.DataFrame:
    out: List[Dict] = []
    pos = {ix: k for k, ix in enumerate(df.index)}
    for class_name in df["class_name"].unique():
        idx = df.index[df["class_name"] == class_name].to_numpy()
        if len(idx) < 2:
            continue
        rows = np.array([pos[i] for i in idx])
        h, e = H[rows], E[rows]
        D = (h[:, None, :] != h[None, :, :]).sum(-1)
        C = e @ e.T
        ph_ok, em_ok = D <= phash_max, C >= cosine_min
        link = {"both": ph_ok & em_ok, "any": ph_ok | em_ok,
                "phash": ph_ok, "embed": em_ok}.get(link_mode)
        if link is None:
            fail(f"unknown link_mode: {link_mode}")
        iu = np.triu_indices(len(idx), k=1)
        sel = link[iu]
        for a, b, d, c in zip(iu[0][sel], iu[1][sel], D[iu][sel], C[iu][sel]):
            out.append({"class_name": class_name,
                        "i": int(idx[a]), "j": int(idx[b]),
                        "hamming": int(d), "cosine": float(c)})
    return pd.DataFrame(out,
                        columns=["class_name", "i", "j", "hamming", "cosine"])


# =============================================================================
# MODE: PRUNE
# =============================================================================

PRIORITY = {"train": 0, "test": 1, "val": 2}   # lower is kept


def select_prunes(orig: pd.DataFrame, pairs: pd.DataFrame,
                  keep_val_test_dups: bool) -> pd.DataFrame:
    """
    Greedily choose which images to delete so that no near-duplicate pair
    spans partitions.

    Rules:
      * train never deleted -- the training protocol stays intact
      * for train<->val and train<->test, the evaluation member is deleted
      * for val<->test, the val member is deleted (unless disabled)
    Iterated so that an image linked to several partners is deleted once and
    resolves all of its pairs.
    """
    split = orig["split"].to_dict()
    removed: Set[int] = set()
    reasons: Dict[int, str] = {}

    # Order pairs so the most serious contamination is resolved first
    def severity(r):
        s = {split[r["i"]], split[r["j"]]}
        if s == {"train", "test"}:
            return 0
        if s == {"train", "val"}:
            return 1
        if s == {"val", "test"}:
            return 2
        return 3

    if pairs.empty:
        return pd.DataFrame(columns=["index", "split", "class_name",
                                     "filename", "reason"])

    pr = pairs.copy()
    pr["severity"] = pr.apply(severity, axis=1)
    pr = pr[pr["severity"] < 3].sort_values(["severity", "hamming"])

    for _, r in pr.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if i in removed or j in removed:
            continue
        si, sj = split[i], split[j]
        if si == sj:
            continue
        if {si, sj} == {"val", "test"} and keep_val_test_dups:
            continue
        # keep the higher-priority partition, delete the other
        drop = i if PRIORITY[si] > PRIORITY[sj] else j
        removed.add(drop)
        reasons[drop] = f"near-duplicate of a {split[i if drop == j else j]} image " \
                        f"(Hamming {int(r['hamming'])}, cosine {r['cosine']:.3f})"

    return pd.DataFrame([{
        "index": k, "split": orig["split"].loc[k],
        "class_name": orig["class_name"].loc[k],
        "filename": orig["filename"].loc[k],
        "reason": reasons[k],
    } for k in sorted(removed)])


def run_prune(df: pd.DataFrame, orig: pd.DataFrame, pairs: pd.DataFrame,
              art: Path, args) -> pd.DataFrame:
    banner("MODE prune  select images to remove")
    prunes = select_prunes(orig, pairs, args.keep_val_test_dups)
    print(f"Images to delete: {len(prunes)}")
    if len(prunes):
        print(prunes.groupby(["split", "class_name"]).size()
              .unstack(fill_value=0).to_string())
    prunes.to_csv(art / "pruned_images.csv", index=False)

    drop_idx = set(prunes["index"].tolist())
    final = df[~df.index.isin(drop_idx)].copy()
    final["new_split"] = final["split"]
    return final


# =============================================================================
# MODE: RESPLIT
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


def build_clusters(orig: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    dsu = DSU(len(orig))
    pos = {ix: k for k, ix in enumerate(orig.index)}
    for _, r in pairs.iterrows():
        dsu.union(pos[int(r["i"])], pos[int(r["j"])])
    fam: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for k, (cn, cid) in enumerate(zip(orig["class_name"], orig["canonical_id"])):
        fam[(cn, cid)].append(k)
    for members in fam.values():
        for m in members[1:]:
            dsu.union(members[0], m)
    roots = [dsu.find(k) for k in range(len(orig))]
    remap = {r: i for i, r in enumerate(sorted(set(roots)))}
    return pd.Series([remap[r] for r in roots], index=orig.index, name="cluster_id")


def run_resplit(df: pd.DataFrame, orig: pd.DataFrame, pairs: pd.DataFrame,
                art: Path, args) -> pd.DataFrame:
    banner("MODE resplit  cluster and re-partition")
    orig = orig.copy()
    orig["cluster_id"] = build_clusters(orig, pairs)
    sizes = orig.groupby("cluster_id").size()
    print(f"Clusters: {orig['cluster_id'].nunique()} over {len(orig)} originals")
    print("Cluster size distribution:")
    print(sizes.value_counts().sort_index().to_string())

    rng = np.random.default_rng(args.seed)
    assign: Dict[int, str] = {}
    for class_name in CLASSES:
        sub = orig[orig["class_name"] == class_name]
        if sub.empty:
            continue
        s = sub.groupby("cluster_id").size()
        cl, sz = s.index.to_numpy(), s.to_numpy()
        perm = rng.permutation(len(cl))
        cl, sz = cl[perm], sz[perm]
        order = np.argsort(-sz, kind="stable")
        cl, sz = cl[order], sz[order]
        total = sz.sum()
        target = {k: f * total for k, f in zip(("train", "val", "test"), SPLIT_FRACS)}
        cur = {"train": 0, "val": 0, "test": 0}
        for c, n in zip(cl, sz):
            pick = max(cur, key=lambda s_: target[s_] - cur[s_])
            assign[int(c)] = pick
            cur[pick] += int(n)

    orig["new_split"] = orig["cluster_id"].map(assign)
    parent = {(cn, cid): ns for cn, cid, ns
              in zip(orig["class_name"], orig["canonical_id"], orig["new_split"])}

    deriv = df[df["is_derived"]].copy()
    deriv["new_split"] = [parent.get((cn, cid)) for cn, cid
                          in zip(deriv["class_name"], deriv["canonical_id"])]
    orphan = int(deriv["new_split"].isna().sum())
    if orphan:
        print(f"WARNING: {orphan} derivatives without a parent original; dropped.")
    deriv = deriv[deriv["new_split"] == "train"]
    print(f"Derivatives retained (train only): {len(deriv)} of {int(df['is_derived'].sum())}")

    keep = orig.drop(columns=["cluster_id"])
    return pd.concat([keep, deriv], ignore_index=False)


# =============================================================================
# MATERIALISE + VERIFY
# =============================================================================

def materialise(final: pd.DataFrame, out: Path, overwrite: bool) -> None:
    banner("MATERIALISE")
    if out.exists():
        if not overwrite:
            fail(f"{out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(final.iterrows(), total=len(final), desc="Copy"):
        shutil.copy2(r["filepath"], out / r["new_split"] / r["class_name"] / r["filename"])
    print(f"Wrote {len(final)} files to {out}")


def verify(final: pd.DataFrame, out: Path, art: Path, args) -> pd.DataFrame:
    banner("VERIFY  re-audit the materialised dataset from scratch")
    df2 = index_dataset(out)
    if len(df2) != len(final):
        fail(f"materialised {len(df2)} files, expected {len(final)}")

    # (1) derivatives confined to train
    bad = df2[(df2["split"] != "train") & df2["is_derived"]]
    print(f"Derivatives outside train      : {len(bad)}")
    if len(bad):
        fail("derivatives leaked into val/test")

    # (2) exact duplicates
    df2["md5"] = [md5(p) for p in tqdm(df2["filepath"], desc="MD5")]
    vt = set(df2[df2["split"] != "train"]["md5"])
    exact = int(df2[(df2["split"] == "train") & df2["md5"].isin(vt)].shape[0])
    print(f"Exact cross-split duplicates   : {exact}")
    if exact:
        fail("exact duplicates remain")

    # (3) canonical families
    fam = (df2.groupby(["class_name", "canonical_id"])
           .agg(splits=("split", lambda x: sorted(set(x)))).reset_index())
    fam_x = fam[fam["splits"].apply(lambda x: len(x) > 1)]
    print(f"Cross-split canonical families : {len(fam_x)}")
    if len(fam_x):
        fam_x.to_csv(art / "final_family_leaks.csv", index=False)
        fail("canonical-family leakage remains")

    # (4) perceptual near-duplicates -- recomputed on the OUTPUT, not inherited
    orig2 = df2[~df2["is_derived"]].reset_index(drop=True)
    H2 = compute_phash(orig2["filepath"].tolist())
    E2 = compute_embeddings(orig2["filepath"].tolist(), args.batch_size, args.embed_model)
    pairs2 = within_class_pairs(orig2, H2, E2, args.phash_max,
                                args.cosine_min, args.link_mode)
    sp = orig2["split"].to_numpy()
    if pairs2.empty:
        n_cross = 0
        cross2 = pairs2
    else:
        cross2 = pairs2[sp[pairs2["i"]] != sp[pairs2["j"]]]
        if args.keep_val_test_dups and len(cross2):
            keep = [not ({sp[int(r['i'])], sp[int(r['j'])]} == {"val", "test"})
                    for _, r in cross2.iterrows()]
            cross2 = cross2[keep]
        n_cross = len(cross2)
    print(f"Cross-split near-duplicate pairs: {n_cross}")
    pairs2.to_csv(art / "final_linked_pairs.csv", index=False)
    if n_cross:
        cross2.to_csv(art / "final_cross_split_pairs.csv", index=False)
        fail("near-duplicate leakage remains -- inspect final_cross_split_pairs.csv")

    print("\nAll four audits passed: exact, family, derivative, perceptual.")
    return df2


def write_manifest(df2: pd.DataFrame, out: Path, art: Path, args) -> str:
    banner("MANIFEST")
    manifest = (df2[["split", "class_name", "filename", "md5",
                     "canonical_id", "is_derived"]]
                .sort_values(["split", "class_name", "filename"])
                .reset_index(drop=True))
    mpath = art / "final_split_manifest.csv"
    manifest.to_csv(mpath, index=False)
    sha = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()

    counts = (df2.groupby(["class_name", "split"]).size()
              .unstack(fill_value=0).reindex(columns=["train", "val", "test"])
              .reindex(list(CLASSES)))
    counts["total"] = counts.sum(axis=1)
    print(counts.to_string())
    counts.to_csv(art / "final_class_counts.csv")

    n = {s: int((df2["split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"\ntrain={n['train']}  val={n['val']}  test={n['test']}  total={len(df2)}")

    fingerprint = {
        "manifest_sha256": sha,
        "mode": args.mode,
        "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
        "n_total": int(len(df2)),
        "phash_max": args.phash_max,
        "cosine_min": args.cosine_min,
        "link_mode": args.link_mode,
        "embed_model": args.embed_model,
        "keep_val_test_dups": bool(args.keep_val_test_dups),
        "seed": args.seed,
        "source": str(args.src),
        "classes": list(CLASSES),
    }
    with open(art / "final_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    # Loader-order index so existing checkpoints can be rescored consistently
    decontam = {}
    for split in ("val", "test"):
        order, labels = [], []
        for ci, c in enumerate(sorted(p.name for p in (out / split).iterdir() if p.is_dir())):
            for fn in sorted(p.name for p in (out / split / c).glob("*.*")):
                order.append(fn); labels.append(ci)
        decontam[split] = {
            "order": order, "labels": labels,
            "sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        }
    decontam["classes"] = sorted(p.name for p in (out / "test").iterdir() if p.is_dir())
    with open(art / "final_decontam_index.json", "w") as f:
        json.dump(decontam, f, indent=1)

    print(f"\nManifest : {mpath}")
    print(f"SHA256   : {sha}")
    if args.expect_manifest_sha256 and sha != args.expect_manifest_sha256:
        fail(f"manifest SHA mismatch\n  expected {args.expect_manifest_sha256}\n"
             f"  got      {sha}")
    return sha


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=str, default="")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--artifacts", type=str, default="")
    p.add_argument("--mode", type=str, default="prune", choices=["prune", "resplit"])
    p.add_argument("--phash-max", type=int, default=5)
    p.add_argument("--cosine-min", type=float, default=0.92)
    p.add_argument("--link-mode", type=str, default="both",
                   choices=["both", "any", "phash", "embed"])
    p.add_argument("--keep-val-test-dups", action="store_true",
                   help="Do not prune val<->test near-duplicates "
                        "(they are not training leakage).")
    p.add_argument("--seed", type=int, default=42, help="resplit only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-model", type=str, default="resnet50")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--expect-manifest-sha256", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src, out, art = Path(args.src), Path(args.out), Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)

    banner("FINALIZE TeaLeafBD DATASET")
    print(f"src   : {src}")
    print(f"out   : {out}")
    print(f"mode  : {args.mode}")
    print(f"link  : {args.link_mode}  phash<={args.phash_max}  cos>={args.cosine_min}")

    banner("INDEX")
    df = index_dataset(src)
    print(df.groupby(["split", "is_derived"]).size().to_string())
    orig = df[~df["is_derived"]]
    print(f"\nOriginals: {len(orig)}   Derivatives: {int(df['is_derived'].sum())}")

    banner("SIGNALS")
    orig_r = orig.reset_index(drop=True)
    H, E = load_or_compute_signals(orig_r, art, args)
    # map back onto the original (non-reset) index
    orig_idx = orig.index.to_numpy()
    remap = {k: orig_idx[k] for k in range(len(orig_r))}
    pairs = within_class_pairs(orig_r, H, E, args.phash_max,
                               args.cosine_min, args.link_mode)
    if not pairs.empty:
        pairs["i"] = pairs["i"].map(remap)
        pairs["j"] = pairs["j"].map(remap)

    sp = orig["split"].to_dict()
    if pairs.empty:
        print("No near-duplicate pairs detected at this operating point.")
        cross = pairs
    else:
        cross = pairs[[sp[int(a)] != sp[int(b)]
                       for a, b in zip(pairs["i"], pairs["j"])]]
    print(f"Linked pairs      : {len(pairs)}")
    print(f"  cross-partition : {len(cross)}")
    print(f"  within-partition: {len(pairs) - len(cross)}  (legitimate; not removed)")

    if args.mode == "prune":
        final = run_prune(df, orig, pairs, art, args)
    else:
        final = run_resplit(df, orig, pairs, art, args)

    materialise(final, out, args.overwrite)
    df2 = verify(final, out, art, args)
    sha = write_manifest(df2, out, art, args)

    banner("DONE")
    print(f"Final dataset : {out}")
    print(f"Manifest SHA  : {sha}")
    print("\nUse for every experiment:")
    print(f"    --data_root {out}")
    print("\nIMPORTANT: split counts have changed. Do NOT pass")
    print("--strict_clean_counts (it asserts the old 6090/851/852 protocol).")


if __name__ == "__main__":
    main()
