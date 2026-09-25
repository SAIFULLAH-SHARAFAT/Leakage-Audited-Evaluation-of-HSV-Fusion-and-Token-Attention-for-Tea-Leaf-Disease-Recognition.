#!/usr/bin/env python3
"""Sanity/demo for the two confidence-mixture mappings on seven classes."""
import torch, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from importlib import import_module
# file begins with a digit, load through importlib.util
import importlib.util
spec=importlib.util.spec_from_file_location("components",Path(__file__).parent/"01_shared_color_components.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

for top in [1/7,0.30,0.50,0.70,0.90]:
    rest=(1-top)/6; p=torch.tensor([[top]+[rest]*6],dtype=torch.float32)
    logits=torch.log(p)
    row=[f"maxp={top:.3f}"]
    for mode in ["decode_branch","normalized"]:
        mix=m.ConfidenceMixture(.5,mode); a=mix.alpha(p).item(); row.append(f"{mode} alpha={a:.3f}")
    print(" | ".join(row))
