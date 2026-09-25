#!/usr/bin/env python3
"""Reusable components for a future shared-weight chromatic Swin study.

These components are runnable and unit-testable independently of timm internals.
The block-level shared-KV Swin adapter is intentionally separate.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class HSVPatchStem(nn.Module):
    """Separate color patch embedding; all subsequent large weights can be shared."""
    def __init__(self,in_chans=4,embed_dim=96,patch_size=4,bias=True):
        super().__init__()
        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size,bias=bias)
    def forward(self,x): return self.proj(x)

class LayerCoupling(nn.Module):
    """Decode-Branch-inspired elementwise couplings for one Transformer block."""
    def __init__(self,dim:int,mlp_ratio:int=4,mode:str="abc"):
        super().__init__(); assert mode in {"abc","ac"}; self.mode=mode
        self.a=nn.Parameter(torch.zeros(dim))
        self.c=nn.Parameter(torch.zeros(dim))
        self.b=nn.Parameter(torch.zeros(dim*mlp_ratio)) if mode=="abc" else None
    def mix_query(self,q2,q1): return q2 + self.a * q1
    def mix_hidden(self,h2,h1): return h2 if self.b is None else h2 + self.b * h1
    def mix_mlp(self,m2,m1): return m2 + self.c * m1

class ConfidenceMixture(nn.Module):
    """Primary/color probability mixture with collision-probability confidence.

    mode='decode_branch': alpha=clip(sum p^2, min_alpha, 1), matching the
    source idea.

    mode='normalized': maps the K-class uniform collision 1/K to min_alpha and
    a degenerate distribution to 1 before clipping. This is a proposed
    multiclass adaptation, not a claim from Decode-Branch.
    """
    def __init__(self,min_alpha:float=.5,mode:str="decode_branch",detach_alpha:bool=False):
        super().__init__(); assert 0<=min_alpha<=1; assert mode in {"decode_branch","normalized"}
        self.min_alpha=min_alpha; self.mode=mode; self.detach_alpha=detach_alpha
    def alpha(self,p):
        coll=p.square().sum(dim=-1,keepdim=True)
        if self.mode=="normalized":
            k=p.shape[-1]; coll=(coll-1.0/k)/(1.0-1.0/k); coll=coll.clamp(0,1)
            a=self.min_alpha+(1-self.min_alpha)*coll
        else:
            a=coll.clamp(self.min_alpha,1.0)
        return a.detach() if self.detach_alpha else a
    def forward(self,primary_logits,color_logits):
        p=F.softmax(primary_logits,dim=-1); q=F.softmax(color_logits,dim=-1); a=self.alpha(p)
        mix=a*p+(1-a)*q
        return mix,a,p,q
    def nll(self,primary_logits,color_logits,target):
        mix,a,p,q=self(primary_logits,color_logits)
        loss=-torch.log(mix.gather(1,target[:,None]).clamp_min(1e-12)).mean()
        return loss,{"alpha_mean":float(a.detach().mean()),"alpha_min":float(a.detach().min()),"alpha_max":float(a.detach().max())}

class GatedAttentionCopy(nn.Module):
    """F4 attention-ablation primitive: the branch receives g * primary attention output.

    The branch skips its own attention pass, so F4 performs ONE attention pass per
    block where F1/F2/F5 perform two. It is parameter-matched to F1 (g replaces
    the query coupling a, both D-sized) but NOT compute-matched: it asks whether
    the branch's own attention pass matters. The compute-matched control is F5.
    """
    def __init__(self,dim:int,init:float=1.0): super().__init__(); self.g=nn.Parameter(torch.full((dim,),float(init)))
    def forward(self,primary_attention_output): return self.g * primary_attention_output

def rgb_luma_rep(x_rgb01:torch.Tensor)->torch.Tensor:
    """F5 information-null branch input: [R, G, B, Y] with Y = BT.601 luma.

    Four channels, like the sin/cos-HSV representation, so the patch stem is
    parameter-identical (4*96*16 + 96 = 6,240) and F5 matches F1 in parameters
    and compute exactly. It carries nothing the primary RGB trajectory does not
    already see, so F1 vs F5 isolates the chromatic re-parameterisation from the
    extra attention pass and the added parameters. Normalise it the same way as
    the HSV input before the stem.
    """
    r,g,b=x_rgb01[:,0:1],x_rgb01[:,1:2],x_rgb01[:,2:3]
    return torch.cat([r,g,b,0.299*r+0.587*g+0.114*b],dim=1)

def prediction_diversity(y,pred_a,pred_b)->dict:
    """Error overlap between two predictors scored on the same images.

    A confidence mixture of two heads (F3) can only recover images that exactly
    one head gets right, so `oracle_accuracy` bounds what F3 can reach and
    `error_phi` measures how correlated the heads' mistakes are (1 = identical).
    Measure this for the primary and colour heads before crediting F3 with a gain.
    """
    y,pa,pb=(torch.as_tensor(t) for t in (y,pred_a,pred_b))
    ea,eb=pa!=y,pb!=y
    n=int(y.numel())
    both=int((ea&eb).sum()); a_only=int((ea&~eb).sum()); b_only=int((~ea&eb).sum())
    none=n-both-a_only-b_only
    denom=math.sqrt((both+a_only)*(both+b_only)*(none+a_only)*(none+b_only))
    return {"both_wrong":both,"a_only_wrong":a_only,"b_only_wrong":b_only,"both_right":none,
            "accuracy_a":1-(both+a_only)/n,"accuracy_b":1-(both+b_only)/n,
            "oracle_accuracy":1-both/n,
            "error_phi":(both*none-a_only*b_only)/denom if denom else float("nan")}

@torch.no_grad()
def rms_match_linear_(linear:nn.Linear,input_tensor:torch.Tensor,target_rms:float):
    """Data-driven future initializer: rescale a Linear weight to a target output RMS.

    Use only on a declared training calibration batch before optimization. Do not
    tune this using validation/test performance.
    """
    y=linear(input_tensor.float()); cur=y.square().mean().sqrt().item()
    if cur<=0: raise ValueError("current RMS is zero")
    scale=float(target_rms)/cur; linear.weight.mul_(scale)
    if linear.bias is not None: linear.bias.mul_(scale)
    return {"before_rms":cur,"target_rms":float(target_rms),"weight_scale":scale}
