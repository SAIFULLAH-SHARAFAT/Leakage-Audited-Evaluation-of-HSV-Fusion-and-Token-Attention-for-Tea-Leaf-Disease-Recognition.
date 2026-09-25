#!/usr/bin/env python3
"""Non-destructive derivative-vs-evaluation perceptual leakage audit.

The original perceptual audit screened originals only.  This script closes that
coverage gap without changing the frozen dataset: it compares every retained
training derivative against validation/test images of the same class using the
same two signals and operating point recorded in final_fingerprint.json:
ImageHash pHash + L2-normalized timm embedding cosine similarity.

A detected linked pair is treated as a pre-training STOP condition.  The script
always writes the pair table and summary before exiting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from campaign import matrix


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compute_phash(paths: list[Path]) -> np.ndarray:
    try:
        import imagehash
    except ImportError as e:
        raise SystemExit("ImageHash is required: python -m pip install ImageHash") from e
    out = []
    for p in tqdm(paths, desc="pHash"):
        with Image.open(p) as im:
            out.append(imagehash.phash(im).hash.flatten())
    return np.asarray(out, dtype=bool)


def compute_embeddings(paths: list[Path], batch_size: int, model_name: str) -> np.ndarray:
    try:
        import torch
        import timm
    except ImportError as e:
        raise SystemExit("torch and timm are required for the embedding signal") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embedding model: {model_name} on {device}")
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
            batch = torch.stack(ims).to(device)
            feats.append(model(batch).float().cpu().numpy())
    f = np.concatenate(feats, axis=0)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    return f


def main() -> None:
    cfg = matrix()
    defaults = cfg["defaults"]
    repo = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=str((repo / defaults["results_dir"]).parent / "audits"))
    args = ap.parse_args()

    data_root = repo / defaults["data_root"]
    manifest_path = repo / defaults["manifest_csv"]
    fingerprint_path = repo / defaults["manifest_fingerprint"]
    expected_sha = defaults["expected_manifest_sha256"]

    if sha256(manifest_path) != expected_sha:
        raise SystemExit("Frozen manifest SHA does not match campaign configuration")
    fp = json.loads(fingerprint_path.read_text())
    if fp.get("manifest_sha256") != expected_sha:
        raise SystemExit("Fingerprint JSON does not match campaign manifest SHA")

    phash_max = int(fp["phash_max"])
    cosine_min = float(fp["cosine_min"])
    link_mode = str(fp["link_mode"])
    model_name = str(fp["embed_model"])
    if link_mode not in {"both", "any", "phash", "embed"}:
        raise ValueError(f"Unsupported link_mode={link_mode}")

    df = pd.read_csv(manifest_path)
    required = {"split", "class_name", "filename", "is_derived"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    if df["is_derived"].dtype != bool:
        df["is_derived"] = df["is_derived"].map(
            lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"}
        )

    deriv = df[(df["split"] == "train") & df["is_derived"]].copy()
    eval_df = df[df["split"].isin(["val", "test"])].copy()
    if len(deriv) != int(fp.get("n_derivatives", len(deriv))):
        raise SystemExit(f"Derivative count mismatch: manifest has {len(deriv)}")
    if eval_df["is_derived"].any():
        raise SystemExit("Evaluation split unexpectedly contains derived images")

    work = pd.concat([deriv.assign(_kind="derivative"), eval_df.assign(_kind="evaluation")],
                     ignore_index=True)
    work["filepath"] = [
        data_root / r.split / r.class_name / r.filename for r in work.itertuples()
    ]
    absent = [str(p) for p in work["filepath"] if not Path(p).exists()]
    if absent:
        raise SystemExit(f"Missing dataset files; first: {absent[:3]}")

    paths = [Path(p) for p in work["filepath"]]
    H = compute_phash(paths)
    E = compute_embeddings(paths, args.batch_size, model_name)

    pairs = []
    for class_name in sorted(work["class_name"].unique()):
        di = work.index[(work["class_name"] == class_name) & (work["_kind"] == "derivative")].to_numpy()
        ei = work.index[(work["class_name"] == class_name) & (work["_kind"] == "evaluation")].to_numpy()
        if not len(di) or not len(ei):
            continue
        D = (H[di][:, None, :] != H[ei][None, :, :]).sum(axis=-1)
        C = E[di] @ E[ei].T
        ph_ok = D <= phash_max
        em_ok = C >= cosine_min
        link = {"both": ph_ok & em_ok, "any": ph_ok | em_ok,
                "phash": ph_ok, "embed": em_ok}[link_mode]
        aa, bb = np.where(link)
        for a, b in zip(aa.tolist(), bb.tolist()):
            rd = work.loc[int(di[a])]
            re = work.loc[int(ei[b])]
            pairs.append({
                "class_name": class_name,
                "train_derivative": rd["filename"],
                "eval_split": re["split"],
                "eval_filename": re["filename"],
                "hamming": int(D[a, b]),
                "cosine": float(C[a, b]),
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "derivative_eval_perceptual_pairs.csv"
    pd.DataFrame(pairs, columns=[
        "class_name", "train_derivative", "eval_split", "eval_filename", "hamming", "cosine"
    ]).to_csv(pair_path, index=False)

    summary = {
        "campaign_id": defaults["campaign_id"],
        "manifest_sha256": expected_sha,
        "n_derivatives_screened": int(len(deriv)),
        "n_eval_images_screened": int(len(eval_df)),
        "phash_max": phash_max,
        "cosine_min": cosine_min,
        "link_mode": link_mode,
        "embed_model": model_name,
        "n_linked_derivative_eval_pairs": int(len(pairs)),
        "status": "PASS" if not pairs else "FAIL",
        "non_destructive": True,
    }
    summary_path = out_dir / "derivative_eval_perceptual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {pair_path}")
    print(f"wrote {summary_path}")

    if pairs:
        raise SystemExit(
            "FAIL: derivative-vs-evaluation perceptual links were detected. "
            "Do not start the final training campaign until they are reviewed."
        )
    print("PASS: no training derivative is linked to validation/test at the frozen operating point")


if __name__ == "__main__":
    main()
