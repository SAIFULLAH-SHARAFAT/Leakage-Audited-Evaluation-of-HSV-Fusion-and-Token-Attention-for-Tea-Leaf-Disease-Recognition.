#!/usr/bin/env python3
"""Reference contract for the future asymmetric shared-KV Swin color branch.

This file intentionally does NOT pretend that a generic timm Swin block can be
patched safely without knowing the installed timm block API. The equations and
state contract are executable at the tensor level; the future timm adapter must
supply the actual window-attention operations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import torch
from torch import nn

@dataclass
class PrimaryCache:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_out: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_out: torch.Tensor

class SharedSwinBlockAdapter(Protocol):
    def primary(self,block:nn.Module,s1:torch.Tensor)->tuple[torch.Tensor,PrimaryCache]: ...
    def color(self,block:nn.Module,s2:torch.Tensor,cache:PrimaryCache,coupling:nn.Module)->torch.Tensor: ...

# Pseudocode contract used for future implementation:
#   s1, cache = adapter.primary(shared_block, s1)
#   s2        = adapter.color(shared_block, s2, cache, coupling_l)
# The adapter must guarantee that adapter.primary never reads s2.
# Large block parameters are the *same module objects* for both trajectories.

raise_on_direct_execution = "This is a reference contract, not a timm-version-specific trainer."
if __name__ == "__main__":
    print(raise_on_direct_execution)
    print("Run 05_inspect_timm_swin_api.py in the future environment, then implement the adapter against the pinned timm version.")
