#!/usr/bin/env python3
"""Verify the frozen 7,714-instance dataset before start the experiments"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
EXPECTED_SHA="d407fdb133ebc95c3b0bc6f815280786bcdef37cd6c3c009192198fd21943d3d"
EXPECTED={
 "train":{"Brown Blight":858,"Gray Blight":896,"Green mirid bug":897,
          "Healthy leaf":893,"Helopeltis":863,"Red spider":835,"Tea algal leaf spot":848},
 "val":{"Brown Blight":83,"Gray Blight":163,"Green mirid bug":185,
        "Healthy leaf":147,"Helopeltis":86,"Red spider":76,"Tea algal leaf spot":76},
 "test":{"Brown Blight":86,"Gray Blight":160,"Green mirid bug":179,
         "Healthy leaf":141,"Helopeltis":89,"Red spider":75,"Tea algal leaf spot":78},
}
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default=str(REPO/"data"/"Tea_leaf_dataset"))
    ap.add_argument("--manifest",default=str(REPO/"data"/"manifests"/"final_split_manifest.csv"))
    ap.add_argument("--fingerprint",default=str(REPO/"data"/"manifests"/"final_fingerprint.json"))
    ap.add_argument("--verify-md5",action="store_true",help="Re-hash all 7,714 files")
    args=ap.parse_args()
    root,mp,fp=Path(args.data_root),Path(args.manifest),Path(args.fingerprint)
    if not root.exists() or not mp.exists() or not fp.exists():
        raise SystemExit("Dataset/manifests missing. Run 00_fetch_frozen_dataset.py first.")

    # Cryptographic lock: hash the immutable manifest artifact itself.
    # Do NOT round-trip through pandas before hashing: pandas versions may change
    # dtype inference / CSV serialization while the underlying file is unchanged.
    sha = hashlib.sha256(mp.read_bytes()).hexdigest()

    df = pd.read_csv(mp)
    required = {"split", "class_name", "filename", "md5",
                "canonical_id", "is_derived"}
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f"Manifest missing columns: {sorted(miss)}")

    cols = ["split", "class_name", "filename", "md5",
            "canonical_id", "is_derived", "op"]
    if "op" not in df.columns:
        df["op"] = ""

    # Canonical table is retained for all semantic audits, but is no longer used
    # as the cryptographic fingerprint.
    canon = (
        df[cols]
        .sort_values(["split", "class_name", "filename"])
        .reset_index(drop=True)
    )
    
    # Separate parsed boolean view for audits without altering the bytes used
    # to reproduce the manifest fingerprint.
    is_derived = canon["is_derived"]
    if is_derived.dtype != bool:
        is_derived = is_derived.map(
            lambda x: str(x).strip().lower() in {"1","true","t","yes","y"}
        )
    meta=json.load(open(fp))
    print("manifest SHA256:",sha)
    if sha != EXPECTED_SHA or meta.get("manifest_sha256") != EXPECTED_SHA:
        raise SystemExit(f"Manifest lock mismatch. Expected {EXPECTED_SHA}")

    counts={}
    actual_rows=[]
    for split,classes in EXPECTED.items():
        counts[split]=0
        for cls,nexp in classes.items():
            d=root/split/cls
            if not d.exists(): raise SystemExit(f"Missing {d}")
            files=sorted(q for q in d.iterdir() if q.suffix.lower() in EXTS)
            if len(files)!=nexp:
                raise SystemExit(f"Count mismatch {split}/{cls}: got {len(files)}, expected {nexp}")
            counts[split]+=len(files)
            actual_rows.extend((split,cls,q.name,q) for q in files)
    if counts != {"train":6090,"val":816,"test":808}:
        raise SystemExit(f"Split totals wrong: {counts}")
    if sum(counts.values())!=7714: raise SystemExit("Total is not 7,714")

    # File-set equivalence to manifest.
    actual={(a,b,c) for a,b,c,_ in actual_rows}
    declared={(str(r.split),str(r.class_name),str(r.filename)) for r in canon.itertuples()}
    if actual != declared:
        print("files only on disk:",list(sorted(actual-declared))[:10])
        print("files only in manifest:",list(sorted(declared-actual))[:10])
        raise SystemExit("Filesystem and final manifest differ")

    # Cross-split family confinement and derivative confinement.
    spans=canon.groupby(["class_name","canonical_id"])["split"].nunique()
    n_family=int((spans>1).sum())
    n_deriv_eval=int(canon[(canon["split"]!="train") & is_derived].shape[0])
    md5sp=canon.groupby("md5")["split"].nunique()
    n_exact=int((md5sp>1).sum())
    if meta.get("n_train_original") not in (None,3971):
        raise SystemExit(f"Unexpected n_train_original in fingerprint: {meta.get('n_train_original')}")
    if meta.get("n_derivatives") not in (None,2119):
        raise SystemExit(f"Unexpected n_derivatives in fingerprint: {meta.get('n_derivatives')}")
    print("split totals:",counts)
    print("cross-split exact-MD5 groups:",n_exact)
    print("cross-split canonical families:",n_family)
    print("derived images outside train:",n_deriv_eval)
    if any([n_family,n_deriv_eval,n_exact]):
        raise SystemExit("Leakage audit failed")

    if args.verify_md5:
        import hashlib as _h
        lookup={(r.split,r.class_name,r.filename):r.md5 for r in canon.itertuples()}
        bad=[]
        for split,cls,name,path in actual_rows:
            h=_h.md5()
            with open(path,"rb") as f:
                for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
            if h.hexdigest()!=lookup[(split,cls,name)]: bad.append(str(path))
        if bad: raise SystemExit(f"MD5 mismatch for {len(bad)} files; first: {bad[:3]}")
        print("per-file MD5 verification: PASS")
    print("PASS: frozen Hugging Face experiment dataset verified.")

if __name__=="__main__": main()