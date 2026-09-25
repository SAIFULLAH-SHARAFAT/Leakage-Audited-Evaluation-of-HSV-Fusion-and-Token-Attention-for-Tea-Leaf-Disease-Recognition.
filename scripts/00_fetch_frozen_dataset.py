#!/usr/bin/env python3
"""Download the frozen experiment dataset from Hugging Face.

The revision experiments use the public frozen release:
  saifullah03/tea-leaf-disease-dataset

The script records the resolved Hugging Face commit SHA. For the final paper,
report that SHA (or tag) together with the dataset URL so the exact public
snapshot can be recovered even if `main` later changes.
"""
from __future__ import annotations
import argparse, json, shutil, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "saifullah03/tea-leaf-disease-dataset"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--revision", default="main", help="HF branch/tag/commit")
    ap.add_argument("--snapshot-dir", default=str(REPO/"data"/"hf_snapshot"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy", action="store_true", help="Copy data instead of creating local symlinks")
    args=ap.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    dst=Path(args.snapshot_dir)
    if dst.resolve() == REPO or not dst.resolve().is_relative_to(REPO/"data"):
        raise SystemExit("Snapshot directory must be a dedicated directory beneath repo/data")
    if dst.resolve() in [(REPO/"data").resolve(), (REPO/"data/manifests").resolve(),
                         (REPO/"data/Tea_leaf_dataset").resolve()]:
        raise SystemExit("Snapshot directory overlaps protected data")
    dst.mkdir(parents=True,exist_ok=True)

    api=HfApi()
    info=api.dataset_info(REPO_ID, revision=args.revision)
    resolved_sha=info.sha
    print("Resolved Hugging Face commit:", resolved_sha)

    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=resolved_sha,
        local_dir=str(dst),
    )
    src_data=dst/"data"/"Tea_leaf_dataset"
    src_man=dst/"data"/"manifests"
    if not src_data.exists() or not src_man.exists():
        raise SystemExit(f"Unexpected HF layout under {dst}")

    expected = json.loads((REPO/"data/manifests/final_fingerprint.json").read_text())["manifest_sha256"]
    if hashlib.sha256((src_man/"final_split_manifest.csv").read_bytes()).hexdigest() != expected:
        raise SystemExit("Downloaded snapshot does not match the frozen manifest")
    # Preserve all tracked audit files, including montages. Compare before exposing data.
    for local in (REPO/"data/manifests").rglob("*"):
        if local.is_file():
            remote = src_man/local.relative_to(REPO/"data/manifests")
            if not remote.is_file() or remote.read_bytes() != local.read_bytes():
                raise SystemExit(f"Downloaded audit differs or is missing: {local.name}")

    # Expose stable paths expected by the experiment matrix. Symlinks avoid
    # duplicating several GB on Kaggle; --copy is available when symlinks are
    # undesirable.
    data_out=REPO/"data"/"Tea_leaf_dataset"
    man_out=REPO/"data"/"manifests"
    for src,out in [(src_data,data_out)]:
        if out.exists() or out.is_symlink():
            if not args.overwrite:
                raise SystemExit(f"{out} exists; pass --overwrite to replace it")
            if out.is_symlink() or out.is_file(): out.unlink()
            else:
                raise SystemExit(f"Refusing to delete existing image directory {out}. Verify it or choose a new checkout.")
        if args.copy:
            shutil.copytree(src,out)
        else:
            try:
                out.symlink_to(src.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(src,out)

    meta={
        "repo_id":REPO_ID,
        "requested_revision":args.revision,
        "resolved_commit_sha":resolved_sha,
        "download_url":f"https://huggingface.co/datasets/{REPO_ID}",
        "manifest_sha256": expected,
    }
    (REPO/"data"/"hf_dataset_lock.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    print("Dataset copied to",data_out)
    print("Manifests copied to",man_out)

if __name__=="__main__": main()
