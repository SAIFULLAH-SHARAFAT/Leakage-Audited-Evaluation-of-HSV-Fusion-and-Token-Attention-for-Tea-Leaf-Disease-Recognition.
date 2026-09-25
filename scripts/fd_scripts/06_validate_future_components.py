#!/usr/bin/env python3
"""Cheap unit tests for future design components (does not require timm)."""
from pathlib import Path
import importlib.util, torch
P=Path(__file__).parent/"01_shared_color_components.py"
spec=importlib.util.spec_from_file_location("components",P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# patch stem
stem=m.HSVPatchStem(4,96,4); assert sum(p.numel() for p in stem.parameters())==6240
# couplings
abc=m.LayerCoupling(96,4,"abc"); assert sum(p.numel() for p in abc.parameters())==576
ac=m.LayerCoupling(96,4,"ac"); assert sum(p.numel() for p in ac.parameters())==192
# mixture is normalized
p=torch.randn(5,7); q=torch.randn(5,7)
for mode in ["decode_branch","normalized"]:
    mix=m.ConfidenceMixture(.5,mode); r,a,_,_=mix(p,q)
    assert torch.allclose(r.sum(-1),torch.ones(5),atol=1e-6); assert (a>=.5).all() and (a<=1).all()
# F4 attention-ablation copy: identity at initialization, and one D-vector per block
copy=m.GatedAttentionCopy(32,1.0); x=torch.randn(2,8,32); assert torch.allclose(copy(x),x)
assert sum(p.numel() for p in m.GatedAttentionCopy(96).parameters())==sum(p.numel() for p in [abc.a])
# F5 information-null input: four channels, so its stem is parameter-identical to the HSV stem
rgb=torch.rand(2,3,16,16); rep=m.rgb_luma_rep(rgb)
assert rep.shape==(2,4,16,16) and torch.allclose(rep[:,:3],rgb)
assert torch.allclose(rep[:,3:],0.299*rgb[:,0:1]+0.587*rgb[:,1:2]+0.114*rgb[:,2:3])
f5_stem=m.HSVPatchStem(rep.shape[1],96,4)
assert sum(p.numel() for p in f5_stem.parameters())==sum(p.numel() for p in stem.parameters())==6240
# F3 ceiling diagnostic
y=torch.tensor([0,1,2,3,0,1]); same=torch.tensor([0,1,0,3,1,1])
d=m.prediction_diversity(y,same,same)
assert abs(d["error_phi"]-1.0)<1e-9 and d["oracle_accuracy"]==d["accuracy_a"]   # identical heads: nothing to mix
disjoint=m.prediction_diversity(torch.tensor([0,1,2,3]),torch.tensor([9,1,2,3]),torch.tensor([0,9,2,3]))
assert disjoint["both_wrong"]==0 and disjoint["oracle_accuracy"]==1.0
print("PASS: future component contracts")
