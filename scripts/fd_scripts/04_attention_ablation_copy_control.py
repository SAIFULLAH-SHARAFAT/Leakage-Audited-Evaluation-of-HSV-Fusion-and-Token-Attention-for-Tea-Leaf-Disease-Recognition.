#!/usr/bin/env python3
"""Standalone primitive for the F4 attention-ablation control.

The branch receives a learned vector-gated copy of the primary attention output
instead of running its own attention pass: one attention pass per block, where
F1/F2/F5 run two. F4 is parameter-matched to F1 (g replaces the query coupling a,
both D-sized) but deliberately NOT compute-matched -- it tests whether the
branch's own attention pass matters. The compute-matched control is F5, which
keeps F1's full graph and feeds the branch information-null [R, G, B, Y] input.

This file was previously named 04_compute_matched_copy_control.py; the old name
claimed a compute match the design does not have.
"""
import torch
from torch import nn


class GatedAttentionCopy(nn.Module):
    def __init__(self, dim: int, init: float = 1.0):
        super().__init__()
        self.g = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, primary_attention_output):
        return self.g * primary_attention_output


if __name__ == "__main__":
    m = GatedAttentionCopy(768)
    x = torch.randn(2, 49, 768)
    y = m(x)
    print("params", sum(p.numel() for p in m.parameters()), "shape", tuple(y.shape),
          "max init diff", (y - x).abs().max().item())
