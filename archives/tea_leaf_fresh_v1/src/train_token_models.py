import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import csv
import json
import time
import math
import shutil
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, ContextManager, Union
from collections import Counter
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from PIL import Image

import timm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from train_swin_three_models import (
        set_seed, worker_init_fn, get_autocast_ctx, make_grad_scaler,
        rgb_to_hsv_torch, hsv_to_sincos_sv, normalize_hsv_rep,
        count_params_m, try_get_gflops, _json_safe_scalar,
        strip_module_prefix, add_module_prefix, load_state_dict_robust,
        EMA, CheckpointManager,
    )
    print(" Imported utilities from train_swin_three_models.py")
except ImportError:
    raise ImportError(
        "train_swin_three_models.py must be in the same directory.\n"
        "Copy it alongside this script before running."
    )


# =============================================================================
# TCCA MODULE
# =============================================================================

class ChromaticCrossAttention(nn.Module):
    """
    Token-Level Chromatic Cross-Attention.

    RGB token sequence is the Query source.
    Color feature map tokens are the Key/Value source.

    Shape contract:
      rgb_tokens  : (B, N, D)   — flattened Swin stage tokens
      color_tokens: (B, N, D)   — projected color feature map tokens
    Returns:
      attended    : (B, N, D)   — LayerNorm applied after attention

    Uses PyTorch's built-in MultiheadAttention for correctness and
    efficiency (fused attention kernel on CUDA).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

        self.attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # (B, N, D) convention throughout
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_tokens: torch.Tensor,
                color_tokens: torch.Tensor) -> torch.Tensor:
        # rgb_tokens  : (B, N, D)  → Query
        # color_tokens: (B, N, D)  → Key, Value
        attended, _ = self.attn(
            rgb_tokens,     # Passed positionally for ptflops compatibility
            color_tokens,   # Key
            color_tokens,   # Value
        )
        # Residual + LayerNorm (pre-norm style for stability)
        return self.norm(rgb_tokens + attended)


class ColorEncoder(nn.Module):
    """
    Lightweight CNN encoder producing spatial feature maps at two resolutions
    matching Swin-S stage 3 (14×14) and stage 4 (7×7).

    Input  : (B, in_channels, 224, 224)  HSV or LAB color image
    Outputs: feat3 (B, base_dim*2, 14, 14)
             feat4 (B, base_dim*4,  7,  7)

    base_dim=64 gives 128 and 256 channels respectively,
    matching the projection dimensions below.
    """

    def __init__(self, in_channels: int = 3, base_dim: int = 64):
        super().__init__()
        # Stride-2 convolutions to reach 56×56
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim), nn.GELU(),
        )
        # 56×56 → 28×28
        self.layer1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim), nn.GELU(),
        )
        # 28×28 → 14×14  (matches Swin-S stage 3 spatial resolution)
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2), nn.GELU(),
        )
        # 14×14 → 7×7  (matches Swin-S stage 4 spatial resolution)
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 4), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)    # (B,  64, 112, 112)
        x = self.layer1(x)  # (B,  64,  56,  56)
        f3 = self.layer2(x) # (B, 128,  14,  14)
        f4 = self.layer3(f3)# (B, 256,   7,   7)
        return f3, f4


class TCCAFusion(nn.Module):
    """
    Token-Level Chromatic Cross-Attention Fusion.

    Projects ColorEncoder outputs into Swin token space,
    then applies gated ChromaticCrossAttention at:
      - Stage 3 output: 14×14 tokens, D=384
      - Stage 4 output:  7× 7 tokens, D=768

    Gate scalar per stage, initialized conservatively.
    Warmup ramp applied externally (same gate_alpha as HSV branch).
    """

    def __init__(
        self,
        swin_dim_stage3: int  = 384,   # Swin-S stage 3 output dim
        swin_dim_stage4: int  = 768,   # Swin-S stage 4 output dim
        color_dim_stage3: int = 128,   # ColorEncoder feat3 channels
        color_dim_stage4: int = 256,   # ColorEncoder feat4 channels
        num_heads: int        = 8,
        attn_dropout: float   = 0.0,
    ):
        super().__init__()

        # Linear projections: color channels → Swin token dim
        self.proj3 = nn.Linear(color_dim_stage3, swin_dim_stage3)
        self.proj4 = nn.Linear(color_dim_stage4, swin_dim_stage4)

        self.tcca3 = ChromaticCrossAttention(swin_dim_stage3, num_heads, attn_dropout)
        self.tcca4 = ChromaticCrossAttention(swin_dim_stage4, num_heads, attn_dropout)

        # Per-stage scalar gates, initialized to -2.0 → sigmoid(-2) ≈ 0.12
        # This mirrors the conservative initialization in your existing code.
        self.gate3 = nn.Parameter(torch.full((1,), -2.0))
        self.gate4 = nn.Parameter(torch.full((1,), -2.0))

    def forward(
        self,
        rgb_tokens3:  torch.Tensor,   # (B, 196, 384)  — 14×14 = 196 tokens
        rgb_tokens4:  torch.Tensor,   # (B,  49, 768)  —  7× 7 =  49 tokens
        color_feat3:  torch.Tensor,   # (B, 128,  14, 14)
        color_feat4:  torch.Tensor,   # (B, 256,   7,  7)
        gate_alpha:   float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = rgb_tokens3.shape[0]

        # Flatten spatial dims: (B, C, H, W) → (B, H*W, C)
        c3 = color_feat3.flatten(2).transpose(1, 2)   # (B, 196, 128)
        c4 = color_feat4.flatten(2).transpose(1, 2)   # (B,  49, 256)

        # Project to Swin dims
        c3 = self.proj3(c3)    # (B, 196, 384)
        c4 = self.proj4(c4)    # (B,  49, 768)

        # Cast color tokens to same dtype as RGB tokens (handles AMP fp16)
        c3 = c3.to(dtype=rgb_tokens3.dtype)
        c4 = c4.to(dtype=rgb_tokens4.dtype)

        # Gated residual: f_out = f_rgb + alpha * sigmoid(gate) * TCCA(f_rgb, c)
        g3 = gate_alpha * torch.sigmoid(self.gate3)
        g4 = gate_alpha * torch.sigmoid(self.gate4)

        out3 = rgb_tokens3 + g3 * self.tcca3(rgb_tokens3, c3)
        out4 = rgb_tokens4 + g4 * self.tcca4(rgb_tokens4, c4)

        return out3, out4



# =============================================================================
# STAGE-4-ONLY TOKEN MODULES FOR THE FRESH REVISION EXPERIMENTS
# =============================================================================

class Stage4TCCAFusion(nn.Module):
    """
    Stage-4-only version of the token attention module.

    The classification head consumes only Stage-4 tokens.  In the earlier
    implementation, Stage-3 attention was applied after the backbone had
    already produced Stage 4 and therefore could not affect classification
    when segmentation was disabled.  Fresh classification experiments remove
    that inactive branch instead of counting it as model capacity.
    """
    def __init__(self, color_dim: int = 768, dim: int = 768,
                 num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(color_dim, dim)
        self.attn = ChromaticCrossAttention(dim, num_heads, attn_dropout)
        self.gate = nn.Parameter(torch.full((1,), -2.0))

    def forward(self, rgb_tokens: torch.Tensor, color_feat: torch.Tensor,
                gate_alpha: float = 1.0) -> torch.Tensor:
        c = color_feat.flatten(2).transpose(1, 2)
        c = self.proj(c).to(dtype=rgb_tokens.dtype)
        g = float(gate_alpha) * torch.sigmoid(self.gate)
        return rgb_tokens + g * self.attn(rgb_tokens, c)


class Stage4TokenMLPAdapter(nn.Module):
    """
    Non-attention capacity control at the same active insertion point.

    With expansion r=3 and D=768:
      Stage4 TLA active params = 6D^2 + 8D + 1
      Stage4 MLP params        = 2rD^2 + (r+5)D + 1
    so r=3 is an exact parameter match.
    """
    def __init__(self, dim: int = 768, expansion: int = 3):
        super().__init__()
        hidden = dim * expansion
        self.norm_in = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.full((1,), -2.0))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor, gate_alpha: float = 1.0) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.norm_in(tokens))))
        h = self.norm_out(tokens + h)
        g = float(gate_alpha) * torch.sigmoid(self.gate)
        return tokens + g * h


class Stage4TokenECA(nn.Module):
    """Lightweight ECA-style attention comparator on Stage-4 token channels."""
    def __init__(self, dim: int = 768, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        t = int(abs((math.log2(dim) + b) / gamma))
        k = t if (t % 2 == 1) else t + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=(k - 1) // 2, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.full((1,), -2.0))

    def forward(self, tokens: torch.Tensor, gate_alpha: float = 1.0) -> torch.Tensor:
        z = tokens.mean(dim=1)  # B,D
        w = torch.sigmoid(self.conv(z.unsqueeze(1))).squeeze(1).unsqueeze(1)
        h = self.norm(tokens + tokens * w)
        g = float(gate_alpha) * torch.sigmoid(self.gate)
        return tokens + g * h


# =============================================================================
# FPN SEGMENTATION DECODER
# =============================================================================

# =============================================================================
# PARAMETER-MATCHED NON-ATTENTION ADAPTER  (capacity control, arch="adapter")
# =============================================================================

class TokenMLPAdapter(nn.Module):
    """
    Non-attention capacity control for TCCAFusion.

    Same insertion points (Swin stage-3 and stage-4 token sequences), same
    gated-residual integration, same conservative gate init (-2.0), same
    warmup schedule.  The only difference is that token MIXING is replaced by
    per-token channel mixing: there is no attention, so tokens never exchange
    information.

    With expansion ratio 3 the parameter count matches TCCAFusion (in RGB
    self-attention mode) EXACTLY, per stage:

        TCCAFusion stage-l params (arch="tcca", use_hsv=False)
          = color_proj_l (D^2 + D)        # key/value projection
          + TCCAFusion.proj_l (D^2 + D)   # projection into Swin token dim
          + MultiheadAttention (4D^2 + 4D)
          + LayerNorm (2D)
          + gate (1)
          = 6D^2 + 8D + 1

        TokenMLPAdapter stage-l params (ratio r)
          = LayerNorm (2D) + W1 (rD^2 + rD) + W2 (rD^2 + D) + LayerNorm (2D)
          + gate (1)
          = 2rD^2 + (r + 5)D + 1

        Setting 2r = 6 gives r = 3, and (r+5)D = 8D.  Exact match.

    Verified at runtime by assert_param_budget().
    """

    def __init__(self, dim: int, expansion: int = 3):
        super().__init__()
        hidden = dim * expansion
        self.norm_in  = nn.LayerNorm(dim)
        self.fc1      = nn.Linear(dim, hidden)
        self.act      = nn.GELU()
        self.fc2      = nn.Linear(hidden, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.gate     = nn.Parameter(torch.full((1,), -2.0))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor, gate_alpha: float = 1.0) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.norm_in(tokens))))
        h = self.norm_out(tokens + h)
        g = gate_alpha * torch.sigmoid(self.gate)
        return tokens + g * h


class AdapterFusion(nn.Module):
    """Drop-in structural twin of TCCAFusion using TokenMLPAdapter."""

    def __init__(self, swin_dim_stage3: int = 384, swin_dim_stage4: int = 768,
                 expansion: int = 3):
        super().__init__()
        self.adapter3 = TokenMLPAdapter(swin_dim_stage3, expansion)
        self.adapter4 = TokenMLPAdapter(swin_dim_stage4, expansion)

    def forward(self, rgb_tokens3: torch.Tensor, rgb_tokens4: torch.Tensor,
                gate_alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.adapter3(rgb_tokens3, gate_alpha),
                self.adapter4(rgb_tokens4, gate_alpha))



class FPNSegDecoder(nn.Module):
    """
    Lightweight FPN-style segmentation decoder.

    Consumes Swin stage 1, 2, 3 feature maps (spatial conv features
    reconstructed from token sequences) and produces a binary lesion
    mask at 224×224 resolution.

    Only active during training when pseudo-masks are provided.
    Returns logits (before sigmoid); sigmoid applied in loss function.
    """

    def __init__(
        self,
        stage_dims: List[int] = [96, 192, 384],  # Swin-S stages 1–3
        decoder_dim: int      = 128,
        num_classes: int      = 1,                # binary lesion mask
    ):
        super().__init__()

        # Lateral projections: reduce each stage to decoder_dim
        self.lat1 = nn.Conv2d(stage_dims[0], decoder_dim, 1)
        self.lat2 = nn.Conv2d(stage_dims[1], decoder_dim, 1)
        self.lat3 = nn.Conv2d(stage_dims[2], decoder_dim, 1)

        # Top-down upsampling path
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2),
            nn.BatchNorm2d(decoder_dim), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2),
            nn.BatchNorm2d(decoder_dim), nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim // 2, 2, stride=2),
            nn.BatchNorm2d(decoder_dim // 2), nn.GELU(),
        )

        # Final upsample from 56×56 to 224×224 (4× bilinear)
        self.final_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.head = nn.Conv2d(decoder_dim // 2, num_classes, 1)

    def forward(
        self,
        f1: torch.Tensor,   # (B,  96, 56, 56)  Stage 1
        f2: torch.Tensor,   # (B, 192, 28, 28)  Stage 2
        f3: torch.Tensor,   # (B, 384, 14, 14)  Stage 3
    ) -> torch.Tensor:      # (B,   1, 224, 224) logits

        p3 = self.lat3(f3)                   # (B, 128, 14, 14)
        p2 = self.lat2(f2) + self.up3(p3)    # (B, 128, 28, 28)
        p1 = self.lat1(f1) + self.up2(p2)    # (B, 128, 56, 56)
        out = self.up1(p1)                   # (B,  64, 112, 112)
        out = self.final_up(out)             # (B,  64, 224, 224)
        return self.head(out)                # (B,   1, 224, 224)


# =============================================================================
# SWIN FEATURE EXTRACTOR WITH HOOKS
# =============================================================================

class SwinWithHooks(nn.Module):
    """
    Wraps a timm Swin backbone and extracts intermediate stage outputs
    via forward hooks — NO modification to timm internals.

    Captured outputs (after each stage's patch_merging / norm):
      stage1_out: (B,  96, 56, 56)   — reshaped from (B, 3136, 96)
      stage2_out: (B, 192, 28, 28)   — reshaped from (B,  784, 192)
      stage3_out: (B, 384, 14, 14)   — reshaped from (B,  196, 384)
      stage4_out: (B, 768,  7,  7)   — reshaped from (B,   49, 768)

    For TCCA we need stage3 and stage4 as TOKEN sequences (B, N, D).
    For the segmentation FPN we need stages 1–3 as SPATIAL maps (B, C, H, W).
    Both are derived from the same hooks.
    """

    # Swin-S spatial resolutions after each stage
    STAGE_SPATIAL = {0: (56, 56), 1: (28, 28), 2: (14, 14), 3: (7, 7)}
    STAGE_DIMS    = {0: 96, 1: 192, 2: 384, 3: 768}   # Swin-S/T channel dims

    def __init__(self, timm_id: str, pretrained: bool, drop_path_rate: float):
        super().__init__()
        self.swin = timm.create_model(
            timm_id,
            pretrained     = pretrained,
            num_classes    = 0,
            drop_path_rate = drop_path_rate,
            global_pool    = "",   # CRITICAL: disable GAP so we get token output
        )
        self._stage_outputs: Dict[int, torch.Tensor] = {}
        self._register_stage_hooks()

    def _register_stage_hooks(self):
        """
        Register forward hooks on each of the 4 Swin stages.
        timm Swin stores stages in self.swin.layers (a ModuleList of length 4).
        Each layer outputs (B, N, C) tokens.
        """
        for stage_idx in range(4):
            stage = self.swin.layers[stage_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # output: (B, N, C)  — token sequence
                    # Store as-is; reshape to spatial in forward() as needed
                    self._stage_outputs[idx] = output
                return hook

            stage.register_forward_hook(make_hook(stage_idx))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalized RGB

        Returns dict with:
            tokens3   : (B, 196, 384)    Stage 3 tokens    — for TCCA
            tokens4   : (B,  49, 768)    Stage 4 tokens    — for TCCA
            feat1     : (B,  96, 56, 56) Stage 1 spatial   — for FPN
            feat2     : (B, 192, 28, 28) Stage 2 spatial   — for FPN
            feat3     : (B, 384, 14, 14) Stage 3 spatial   — for FPN
            pooled    : (B, 768)         Global avg pool   — for classifier

        Handles all timm Swin output formats:
            (B, N, C)    — older timm  (already tokens)
            (B, H, W, C) — newer timm  (channels-last spatial, most common on Kaggle)
            (B, C, H, W) — channels-first spatial (rare)
        """
        self._stage_outputs.clear()

        # Run full Swin forward (hooks populate _stage_outputs automatically)
        _ = self.swin(x)

        B = x.shape[0]

        def normalize_to_tokens(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 3:
                return t                          # (B, N, C) already
            elif t.dim() == 4:
                if t.shape[-1] < t.shape[1]:
                    B_, C, H, W = t.shape         # (B, C, H, W)
                    return t.permute(0, 2, 3, 1).reshape(B_, H * W, C)
                else:
                    B_, H, W, C = t.shape         # (B, H, W, C)  ← your timm version
                    return t.reshape(B_, H * W, C)
            else:
                raise ValueError(f"Unexpected shape: {t.shape}")

        def tokens_to_spatial(tokens: torch.Tensor) -> torch.Tensor:
            B_, N, C = tokens.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Non-square token count N={N}"
            return tokens.transpose(1, 2).reshape(B_, C, H, W)

        s0 = normalize_to_tokens(self._stage_outputs[0])
        s1 = normalize_to_tokens(self._stage_outputs[1])
        s2 = normalize_to_tokens(self._stage_outputs[2])
        s3 = normalize_to_tokens(self._stage_outputs[3])

        feat1 = tokens_to_spatial(s0)
        feat2 = tokens_to_spatial(s1)
        feat3 = tokens_to_spatial(s2)

        tokens3 = s2
        tokens4 = s3
        pooled  = s3.mean(dim=1)

        return {
            "tokens3": tokens3,
            "tokens4": tokens4,
            "feat1":   feat1,
            "feat2":   feat2,
            "feat3":   feat3,
            "pooled":  pooled,
        }


# =============================================================================
# FULL MODEL: Swin + TCCA + optional HSV + optional SegHead
# =============================================================================

class SwinTCCA(nn.Module):
    """
    Fresh classification harness.

    arch:
      tcca    Stage-4 token attention (RGB self-attention or HSV cross-attention)
      rgb     recipe/head-matched control, no token module
      adapter Stage-4 non-attention parameter-matched MLP
      eca     Stage-4 lightweight ECA comparator

    Only Stage 4 is modified because it is the representation consumed by the
    classifier.  This removes the classification-inactive Stage-3 branch that
    existed in the exploratory implementation.
    """

    SWIN_ID = "swin_small_patch4_window7_224.ms_in1k"

    def __init__(
        self,
        arch: str = "tcca",
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        num_classes: int = 7,
        use_hsv: bool = False,
        hsv_use_sincos: bool = True,
        use_seg: bool = False,
        tcca_num_heads: int = 8,
        tcca_dropout: float = 0.0,
        adapter_expansion: int = 3,
        color_base_dim: int = 64,
        fuse_dropout: float = 0.2,
        img_mean: Tuple[float,...] = (0.485, 0.456, 0.406),
        img_std: Tuple[float,...] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        if arch not in ("tcca", "rgb", "adapter", "eca"):
            raise ValueError(f"unknown arch: {arch}")
        if arch != "tcca" and use_hsv:
            raise ValueError("--use_hsv is only meaningful with --arch tcca")
        if use_seg:
            raise ValueError(
                "Fresh paper experiments are classification-only; "
                "the segmentation branch is intentionally disabled."
            )
        self.arch = arch
        self.use_hsv = use_hsv
        self.hsv_use_sincos = hsv_use_sincos
        self.use_seg = False
        self.num_classes = num_classes

        self.register_buffer("img_mean", torch.tensor(img_mean).view(1,3,1,1))
        self.register_buffer("img_std", torch.tensor(img_std).view(1,3,1,1))

        self.swin = SwinWithHooks(
            timm_id=self.SWIN_ID,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        if arch == "adapter":
            self.adapter = Stage4TokenMLPAdapter(768, adapter_expansion)
        elif arch == "eca":
            self.eca = Stage4TokenECA(768)
        elif arch == "tcca":
            if use_hsv:
                in_ch = 4 if hsv_use_sincos else 3
                self.color_encoder = ColorEncoder(
                    in_channels=in_ch, base_dim=color_base_dim
                )
                color_dim4 = color_base_dim * 4
            else:
                # Separate source projection retained so the active RGB-TLA
                # capacity matches the original mechanism algebra.
                self.color_proj4 = nn.Linear(768, 768)
                color_dim4 = 768
            self.tcca = Stage4TCCAFusion(
                color_dim=color_dim4, dim=768,
                num_heads=tcca_num_heads, attn_dropout=tcca_dropout
            )

        self.classifier = nn.Sequential(
            nn.Dropout(fuse_dropout),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(fuse_dropout),
            nn.Linear(384, num_classes),
        )

    def _source_feat4(self, x_norm, swin_out):
        if self.use_hsv:
            autocast_ctx = (
                torch.cuda.amp.autocast(enabled=False)
                if torch.cuda.is_available() else nullcontext()
            )
            with autocast_ctx:
                x_raw = (
                    x_norm.float() * self.img_std.float()
                    + self.img_mean.float()
                ).clamp(0.0, 1.0)
                hsv = rgb_to_hsv_torch(x_raw)
                x_color = (
                    normalize_hsv_rep(hsv_to_sincos_sv(hsv))
                    if self.hsv_use_sincos else hsv
                )
            x_color = x_color.to(dtype=swin_out["tokens4"].dtype)
            _, feat4 = self.color_encoder(x_color)
            return feat4

        B = x_norm.shape[0]
        c4 = self.color_proj4(swin_out["tokens4"])
        return c4.transpose(1,2).reshape(B,768,7,7)

    def forward(self, x_norm: torch.Tensor, gate_alpha: float = 1.0,
                return_seg: bool = False):
        swin_out = self.swin(x_norm)
        tokens4 = swin_out["tokens4"]

        if self.arch == "tcca":
            tokens4 = self.tcca(
                tokens4, self._source_feat4(x_norm, swin_out), gate_alpha
            )
        elif self.arch == "adapter":
            tokens4 = self.adapter(tokens4, gate_alpha)
        elif self.arch == "eca":
            tokens4 = self.eca(tokens4, gate_alpha)
        # rgb: unchanged

        return self.classifier(tokens4.mean(dim=1))




# =============================================================================
# MULTI-TASK DATASET  (handles missing masks gracefully)
# =============================================================================

class TeaLeafDataset(Dataset):
    """
    Drop-in replacement for ImageFolder that additionally loads pseudo-masks.

    For val/test splits or images without masks, mask=None is returned.
    The training loop handles None masks by skipping the seg loss contribution
    for those samples.
    """

    CLASS_NAMES = [
        "Brown Blight", "Gray Blight", "Green mirid bug",
        "Healthy leaf",  "Helopeltis",  "Red spider",
        "Tea algal leaf spot",
    ]

    def __init__(
        self,
        root_dir:    str,
        mask_dir:    Optional[str] = None,
        split:       str           = "train",
        img_size:    int           = 224,
        augment:     bool          = True,
        hue_jitter: float = 0.0
    ):
        self.root_dir  = Path(root_dir)
        self.mask_dir  = Path(mask_dir) if mask_dir else None
        self.split     = split
        self.augment   = augment and (split == "train")

        # Build class → index map from sorted directory listing
        # (matches ImageFolder ordering)
        self.class_to_idx = {c: i for i, c in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Discover all images
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.CLASS_NAMES:
            cls_path = self.root_dir / cls_name
            if not cls_path.exists():
                continue
            for img_path in sorted(cls_path.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        # Also accept ImageFolder-style numeric subdirs (fallback)
        if len(self.samples) == 0:
            # Try using ImageFolder to discover samples
            tmp = ImageFolder(str(self.root_dir))
            for path, lbl in tmp.samples:
                self.samples.append((Path(path), lbl))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

        # Expose .targets for DataLoader sampler compatibility
        self.targets = [lbl for _, lbl in self.samples]
        self.classes = self.CLASS_NAMES

        # Transforms
        hue = float(hue_jitter)
        if self.augment:
            self.img_transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.85, 1.0),
                                    interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=hue),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize(int(img_size * 1.14),
                         interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.mask_resize = T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            T.CenterCrop(img_size),
        ])

        self.img_size = img_size

    def _load_mask(self, img_path: Path) -> Optional[torch.Tensor]:
        """Returns (1, H, W) float32 tensor in [0,1], or None."""
        if self.mask_dir is None:
            return None
        try:
            rel      = img_path.relative_to(self.root_dir)
            mask_p   = self.mask_dir / rel.parent / (rel.stem + "_mask.png")
            if not mask_p.exists():
                return None
            mask_pil = Image.open(mask_p).convert("L")
            mask_pil = self.mask_resize(mask_pil)
            mask_np  = np.array(mask_pil, dtype=np.float32) / 255.0
            return torch.from_numpy(mask_np).unsqueeze(0)   # (1, H, W)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)
        mask_tensor = self._load_mask(img_path)
        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "mask":  mask_tensor,   # (1,H,W) float32 or None
        }


def collate_with_masks(batch: List[Dict]) -> Dict:
    """
    Custom collate that handles variable mask presence.
    Stacks images and labels normally; keeps masks as a list (not a tensor)
    because some entries may be None.
    """
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    masks  = [b["mask"] for b in batch]   # list of (1,H,W) tensors or Nones
    return {"image": images, "label": labels, "mask": masks}


# =============================================================================
# MULTI-TASK LOSS
# =============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation. pred is raw logits."""
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


def multitask_loss(
    cls_logits:      torch.Tensor,          # (B, K)
    seg_logits:      Optional[torch.Tensor],# (B, 1, H, W) or None
    labels:          torch.Tensor,          # (B,)
    masks:           List,                  # list length B: (1,H,W) or None
    lambda_cls:      float = 1.0,
    lambda_seg:      float = 0.5,
    label_smoothing: float = 0.1,
    device:          str   = "cuda",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns (total_loss, log_dict).
    Segmentation loss is computed only for samples with valid pseudo-masks.
    Samples with mask=None contribute 0 to seg loss (no gradient there).
    """
    loss_cls = F.cross_entropy(cls_logits, labels,
                                label_smoothing=label_smoothing)
    log = {"loss_cls": loss_cls.item(), "loss_seg": 0.0, "n_seg": 0}

    loss_seg = torch.tensor(0.0, device=device)
    if seg_logits is not None:
        valid_idx = [i for i, m in enumerate(masks) if m is not None]
        if valid_idx:
            idx_t   = torch.tensor(valid_idx, device=device)
            mask_t  = torch.stack(
                [masks[i].to(device) for i in valid_idx]
            )                                        # (n, 1, H, W)
            seg_sub = seg_logits[idx_t]                # (n, 1, H, W)

            bce      = F.binary_cross_entropy_with_logits(
                seg_sub, mask_t, reduction="mean"
            )
            dice     = dice_loss(seg_sub, mask_t)
            loss_seg = bce + dice
            log["loss_seg"] = loss_seg.item()
            log["n_seg"]    = len(valid_idx)

    total = lambda_cls * loss_cls + lambda_seg * loss_seg
    log["loss_total"] = total.item()
    return total, log


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TCCAConfig:
    # Model
    arch:            str   = "tcca"   # "tcca" | "rgb" | "adapter" | "eca"
    adapter_expansion: int = 3
    num_classes:     int   = 7
    pretrained:      bool  = True
    drop_path_rate:  float = 0.2
    use_hsv:         bool  = False
    hsv_use_sincos:  bool  = True    # True = sin/cos hue; use --hsv_raw for raw HSV
    use_seg:         bool  = False
    tcca_num_heads:  int   = 8
    gate_warmup_epochs: int = 5
    hue_jitter:       float = 0.0

    # Multi-task loss
    lambda_cls:  float = 1.0
    lambda_seg:  float = 0.5

    # Data
    data_root:   str  = "/kaggle/working/tea_leaf_clean"
    mask_dir:    str  = ""    # empty = no segmentation
    input_size:  int  = 224
    batch_size:  int  = 48    # slightly smaller than 64 due to TCCA memory overhead
    num_workers: int  = 2

    # Training
    epochs:                    int   = 80
    warmup_epochs:             int   = 5
    lr:                        float = 5e-4
    warmup_lr_init:            float = 1e-6
    weight_decay:              float = 0.05
    min_lr:                    float = 1e-6
    label_smoothing:           float = 0.1
    grad_clip_norm:            float = 1.0
    gradient_accumulation_steps: int = 1
    use_amp:                   bool  = True
    use_ema:                   bool  = False
    ema_decay:                 float = 0.9998
    early_stopping_patience:   int   = 25
    compute_val_auc:           bool  = False

    # Logging
    run_dir:           str  = "/kaggle/working/experiments_tcca"
    experiment_name:   str  = "tcca_run"
    ckpt_temp_dir:     str  = "/kaggle/temp"
    log_interval:      int  = 50
    save_cm_png:       bool = True
    save_epoch_checkpoints: bool = False
    save_epoch_every:  int  = 5
    keep_last_n_checkpoints: int = 3

    # System
    device:           str  = "cuda" if torch.cuda.is_available() else "cpu"
    seed:             int  = 42
    deterministic:    bool = True
    use_data_parallel: bool = True

    def validate(self):
        if not Path(self.data_root).exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")
        if self.use_seg and not self.mask_dir:
            raise ValueError(
                "--use_seg requires --mask_dir pointing to pseudo-mask directory."
            )
        if self.use_seg and self.mask_dir and not Path(self.mask_dir).exists():
            raise FileNotFoundError(f"mask_dir not found: {self.mask_dir}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.arch not in {"tcca", "rgb", "adapter", "eca"}:
            raise ValueError(f"invalid arch: {self.arch}")
        if self.use_seg:
            raise ValueError("Fresh classification experiments do not use --use_seg")


# =============================================================================
# TRAINER
# =============================================================================

class TCCATrainer:

    def _assert_param_budget(self):
        """Fail loudly if a reviewer-control architecture is mis-built."""
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        def n(mod):
            return sum(p.numel() for p in mod.parameters())

        D = 768
        if m.arch == "rgb":
            for attr in ("tcca", "adapter", "eca", "color_encoder", "color_proj4"):
                assert not hasattr(m, attr), (
                    f"arch='rgb' unexpectedly contains {attr}"
                )
            token_params, expected = 0, 0

        elif m.arch == "adapter":
            token_params = n(m.adapter)
            r = self.cfg.adapter_expansion
            expected = 2*r*D*D + (r+5)*D + 1

        elif m.arch == "eca":
            token_params = n(m.eca)
            expected = None  # lightweight comparator, intentionally not matched

        else:  # tcca
            token_params = n(m.tcca)
            if not m.use_hsv:
                token_params += n(m.color_proj4)
                expected = 6*D*D + 8*D + 1
            else:
                token_params += n(m.color_encoder)
                expected = None

        print(f"   arch={m.arch} active token-module params: {token_params:,}")
        if expected is not None:
            assert token_params == expected, (
                f"PARAM BUDGET MISMATCH for arch={m.arch}: "
                f"got {token_params:,}, expected {expected:,}"
            )
            print(f"   token-module parameter budget verified ({expected:,})")
        self.token_module_params = int(token_params)


    def __init__(self, cfg: TCCAConfig):
        cfg.validate()
        self.cfg = cfg
        set_seed(cfg.seed, deterministic=cfg.deterministic)

        self.run_root   = Path(cfg.run_dir)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.exp_dir    = self.run_root / cfg.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir   = self.exp_dir / "logs"; self.logs_dir.mkdir(exist_ok=True)
        self.config_dir = self.exp_dir / "config"; self.config_dir.mkdir(exist_ok=True)

        self.ckpt_manager = CheckpointManager(
            temp_root       = Path(cfg.ckpt_temp_dir),
            final_root      = self.run_root,
            experiment_name = cfg.experiment_name,
        )

        # ── Build model ───────────────────────────────────────────────────
        print(f"\n🏗️  Building SwinTCCA")
        print(f"   use_hsv={cfg.use_hsv}  use_seg={cfg.use_seg}  "
              f"use_ema={cfg.use_ema}")
        self.model = SwinTCCA(
            pretrained     = cfg.pretrained,
            drop_path_rate = cfg.drop_path_rate,
            num_classes    = cfg.num_classes,
            use_hsv        = cfg.use_hsv,
            hsv_use_sincos = cfg.hsv_use_sincos,
            use_seg        = cfg.use_seg,
            tcca_num_heads = cfg.tcca_num_heads,
            arch           = cfg.arch,
            adapter_expansion = cfg.adapter_expansion,
        ).to(cfg.device)

        self.params_m   = float(count_params_m(self.model))
        self._pending_budget_check = True
        self.gflops     = try_get_gflops(self.model, cfg.input_size, cfg.device)
        self.gflops_out = float(self.gflops) if self.gflops is not None else -1.0
        print(f"   Params: {self.params_m:.2f}M")
        if self.gflops_out >= 0:
            print(f"   GFLOPs: {self.gflops_out:.2f}")
        else:
            print("   GFLOPs: N/A")

        self._assert_param_budget()

        # ── DataParallel ──────────────────────────────────────────────────
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and cfg.use_data_parallel:
            print(f"🚀 DataParallel: {self.n_gpus} GPUs")
            self.model = nn.DataParallel(self.model)

        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.ema   = EMA(base_model, cfg.ema_decay) if cfg.use_ema else None

        # ── Data loaders ──────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()

        # ── Optimizer and scheduler ───────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
        )

        bpe             = len(self.train_loader)
        accum           = cfg.gradient_accumulation_steps
        upd_per_epoch   = int(math.ceil(bpe / accum))
        total_upd       = cfg.epochs * upd_per_epoch
        warmup_upd      = min(cfg.warmup_epochs * upd_per_epoch,
                               max(0, total_upd - 1))

        if warmup_upd > 0:
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(self.optimizer,
                              start_factor=cfg.warmup_lr_init / cfg.lr,
                              end_factor=1.0, total_iters=warmup_upd),
                    CosineAnnealingLR(self.optimizer,
                                      T_max=max(1, total_upd - warmup_upd),
                                      eta_min=cfg.min_lr),
                ],
                milestones=[warmup_upd],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_upd), eta_min=cfg.min_lr
            )

        self.scaler      = make_grad_scaler(cfg.use_amp)
        self.best_val_f1 = -1.0
        self.bad_epochs  = 0
        self.start_epoch = 1
        self.train_log: List[Dict] = []

        self._save_config()
        self._print_data_stats()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_config(self):
        with open(self.config_dir / "config.json", "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2)

    def _print_data_stats(self):
        n_train = len(self.train_loader.dataset)
        n_val   = len(self.val_loader.dataset)
        n_test  = len(self.test_loader.dataset)
        print(f"\n Data: Train={n_train} | Val={n_val} | Test={n_test}")
        if self.cfg.use_seg and self.cfg.mask_dir:
            mask_count = len(list(Path(self.cfg.mask_dir).rglob("*_mask.png")))
            print(f"   Pseudo-masks: {mask_count}")

    def _build_loaders(self):
        root     = Path(self.cfg.data_root)
        mask_dir = self.cfg.mask_dir if self.cfg.use_seg else None
        use_p    = (self.cfg.num_workers > 0) and (self.cfg.epochs >= 2)
        g        = torch.Generator().manual_seed(self.cfg.seed)

        # Fair-comparison rule: the hue-jitter policy is explicit and is not
        # silently changed as a function of architecture/use_hsv.
        hj = float(self.cfg.hue_jitter)

        train_ds = TeaLeafDataset(root / "train", mask_dir,
                                   split="train", img_size=self.cfg.input_size,
                                   augment=True, hue_jitter=hj)
        val_ds   = TeaLeafDataset(root / "val",   None,
                                   split="val",   img_size=self.cfg.input_size,
                                   augment=False, hue_jitter=hj)
        test_ds  = TeaLeafDataset(root / "test",  None,
                                   split="test",   img_size=self.cfg.input_size,
                                   augment=False, hue_jitter=hj)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collate_with_masks,
            worker_init_fn=worker_init_fn, generator=g,
            persistent_workers=use_p,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collate_with_masks, persistent_workers=use_p,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collate_with_masks, persistent_workers=use_p,
        )
        return train_loader, val_loader, test_loader

    def _gate_alpha(self, epoch: int) -> float:
        w = max(1, self.cfg.gate_warmup_epochs)
        return float(min(1.0, epoch / w))

    def _base_model(self) -> SwinTCCA:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # ── Training ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        accum       = self.cfg.gradient_accumulation_steps
        gate_alpha  = self._gate_alpha(epoch)
        use_seg     = self.cfg.use_seg

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader,
                    desc=f"Train {epoch}/{self.cfg.epochs}", leave=False)

        for i, batch in enumerate(pbar, start=1):
            x      = batch["image"].to(self.cfg.device, non_blocking=True)
            labels = batch["label"].to(self.cfg.device, non_blocking=True)
            masks  = batch["mask"]   # list; kept on CPU until loss computation

            with get_autocast_ctx(self.cfg.use_amp):
                if use_seg:
                    cls_logits, seg_logits = self.model(
                        x, gate_alpha=gate_alpha, return_seg=True
                    )
                else:
                    cls_logits = self.model(x, gate_alpha=gate_alpha)
                    seg_logits = None

                loss, log = multitask_loss(
                    cls_logits      = cls_logits,
                    seg_logits      = seg_logits,
                    labels          = labels,
                    masks           = masks,
                    lambda_cls      = self.cfg.lambda_cls,
                    lambda_seg      = self.cfg.lambda_seg,
                    label_smoothing = self.cfg.label_smoothing,
                    device          = self.cfg.device,
                )

            correct    += (cls_logits.argmax(dim=1) == labels).sum().item()
            self.scaler.scale(loss / accum).backward()

            do_step = (i % accum == 0) or (i == len(self.train_loader))
            if do_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update()
                self.optimizer.zero_grad(set_to_none=True)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n          += bs

            if i % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                    ga=f"{gate_alpha:.2f}",
                    seg_n=log["n_seg"],
                )

        return {"loss": total_loss / max(1, n), "acc1": correct / max(1, n)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, use_ema=False,
                 return_arrays=False, compute_auc=True) -> Dict:
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        self.model.eval()

        all_preds, all_targets = [], []
        all_probs = [] if compute_auc else None
        total_loss, correct, n = 0.0, 0, 0

        for batch in tqdm(loader, desc="Eval", leave=False):
            x      = batch["image"].to(self.cfg.device, non_blocking=True)
            labels = batch["label"].to(self.cfg.device, non_blocking=True)

            with get_autocast_ctx(self.cfg.use_amp):
                # Always classification-only during eval (no seg head needed)
                cls_logits = self.model(x, gate_alpha=1.0, return_seg=False)
                loss = F.cross_entropy(cls_logits, labels,
                                        label_smoothing=self.cfg.label_smoothing)

            probs  = F.softmax(cls_logits.float(), dim=1)
            preds  = probs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            bs         = x.size(0)
            total_loss += loss.item() * bs
            n          += bs

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            if compute_auc and all_probs is not None:
                all_probs.append(probs.cpu().numpy())

        if use_ema and self.ema is not None:
            self.ema.restore()

        preds_np   = np.array(all_preds)
        targets_np = np.array(all_targets)

        macro_auc = -1.0
        probs_np  = None
        if compute_auc and all_probs:
            probs_np = np.concatenate(all_probs, 0).astype(np.float64)
            try:
                macro_auc = roc_auc_score(
                    targets_np, probs_np,
                    multi_class="ovr", average="macro",
                    labels=np.arange(self.cfg.num_classes),
                )
            except Exception as e:
                print(f" AUC failed: {e}")

        out = {
            "loss":            float(total_loss / max(1, n)),
            "acc1":            float(correct / max(1, n)),
            "macro_f1":        float(f1_score(targets_np, preds_np, average="macro")),
            "micro_f1":        float(f1_score(targets_np, preds_np, average="micro")),
            "weighted_f1":     float(f1_score(targets_np, preds_np, average="weighted")),
            "macro_precision": float(precision_score(targets_np, preds_np,
                                                      average="macro", zero_division=0)),
            "macro_recall":    float(recall_score(targets_np, preds_np,
                                                   average="macro", zero_division=0)),
            "macro_auc":       float(macro_auc),
        }
        if return_arrays:
            out["preds"]   = preds_np
            out["targets"] = targets_np
            if probs_np is not None:
                out["probs"] = probs_np
        return out

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch, is_best):
        base = self._base_model()
        ckpt = {
            "epoch":       int(epoch),
            "model":       base.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self.scaler.state_dict(),
            "config":      self.cfg.__dict__,
            "best_val_f1": float(self.best_val_f1),
            "bad_epochs":  int(self.bad_epochs),
            "train_log":   self.train_log,
        }
        if self.ema is not None:
            ckpt["ema_shadow"] = {
                k: v.detach().clone().cpu() for k, v in self.ema.shadow.items()
            }
        self.ckpt_manager.save_checkpoint(
            ckpt, epoch, is_best,
            save_epoch_ckpt=self.cfg.save_epoch_checkpoints,
            epoch_interval=self.cfg.save_epoch_every,
            keep_n=self.cfg.keep_last_n_checkpoints,
        )

    def _save_train_log(self):
        if not self.train_log:
            return
        with open(self.logs_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.train_log[0].keys())
            w.writeheader(); w.writerows(self.train_log)

    def resume_from(self, ckpt_path: Path):
        print(f"\n Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.cfg.device)
        load_state_dict_robust(self._base_model(), ckpt["model"])
        for name, obj in [("optimizer", self.optimizer),
                           ("scheduler", self.scheduler),
                           ("scaler",    self.scaler)]:
            if name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f" {name} load failed: {e}")
        if self.ema is not None and "ema_shadow" in ckpt:
            for k, v in ckpt["ema_shadow"].items():
                self.ema.shadow[k] = v.to(self.cfg.device, non_blocking=True)
        self.best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        self.bad_epochs  = int(ckpt.get("bad_epochs", 0))
        self.start_epoch = max(1, int(ckpt.get("epoch", 0)) + 1)
        if "train_log" in ckpt:
            self.train_log = ckpt["train_log"]
        print(f" Resumed: next epoch {self.start_epoch}, best F1 {self.best_val_f1:.4f}")

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(self):
        print("\n" + "=" * 80)
        print("START TRAINING — Stage-4 Token Models")
        print("=" * 80)
        print(f"Experiment : {self.cfg.experiment_name}")
        print(f"use_hsv    : {self.cfg.use_hsv}")
        print(f"use_seg    : {self.cfg.use_seg}")
        print(f"use_ema    : {self.cfg.use_ema}")
        print(f"Params     : {self.params_m:.2f}M")
        print("=" * 80)

        eff_ema_val = False

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_m   = self.train_one_epoch(epoch)
            use_ema_v = self.cfg.use_ema and (self.ema is not None)
            val_m     = self.evaluate(self.val_loader, use_ema=use_ema_v,
                                       return_arrays=False,
                                       compute_auc=self.cfg.compute_val_auc)

            lr  = float(self.optimizer.param_groups[0]["lr"])
            tag = " (EMA)" if use_ema_v else ""
            print(f"\nEpoch {epoch}/{self.cfg.epochs} | LR: {lr:.2e}")
            print(f"  Train  Loss: {train_m['loss']:.4f}  Acc@1: {train_m['acc1']:.4f}")
            print(f"  Val{tag} Loss: {val_m['loss']:.4f}  Acc@1: {val_m['acc1']:.4f}  "
                  f"Macro-F1: {val_m['macro_f1']:.4f}")

            self.train_log.append({
                "epoch": int(epoch), "lr": float(lr),
                "train_loss": float(train_m["loss"]),
                "train_acc1": float(train_m["acc1"]),
                "val_loss":   float(val_m["loss"]),
                "val_acc1":   float(val_m["acc1"]),
                "val_macro_f1": float(val_m["macro_f1"]),
                "val_micro_f1": float(val_m["micro_f1"]),
                "val_macro_auc": float(val_m["macro_auc"]),
            })
            self._save_train_log()

            improved = float(val_m["macro_f1"]) > self.best_val_f1
            if improved:
                eff_ema_val      = use_ema_v
                self.best_val_f1 = float(val_m["macro_f1"])
                self.bad_epochs  = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  New best Macro-F1: {self.best_val_f1:.4f}")
            else:
                self.bad_epochs += 1
                self._save_checkpoint(epoch, is_best=False)
                print(f"  No improvement "
                      f"({self.bad_epochs}/{self.cfg.early_stopping_patience})")

            if self.bad_epochs >= self.cfg.early_stopping_patience:
                print("\n Early stopping triggered.")
                break

        # ── Final test evaluation ─────────────────────────────────────────
        print("\n" + "=" * 80)
        print("FINAL TEST EVALUATION")
        print("=" * 80)

        best_path = self.ckpt_manager.temp_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best_model.pth in {self.ckpt_manager.temp_dir}")

        best_ckpt  = torch.load(best_path, map_location=self.cfg.device)
        base_model = self._base_model()

        if self.cfg.use_ema and "ema_shadow" in best_ckpt:
            print(f" Loading EMA weights (epoch {best_ckpt.get('epoch','?')})")
            state = dict(best_ckpt["model"])
            for k, v in best_ckpt["ema_shadow"].items():
                if k in state:
                    state[k] = v.to(self.cfg.device, non_blocking=True)
            load_state_dict_robust(base_model, state)
            used_ema = True
        else:
            load_state_dict_robust(base_model, best_ckpt["model"])
            used_ema = False

        test_m = self.evaluate(self.test_loader, use_ema=False,
                                return_arrays=True, compute_auc=True)
        test_m["used_ema_weights"] = used_ema
        test_m["params_m"]         = float(self.params_m)
        test_m["gflops"]           = float(self.gflops_out)
        test_m["token_module_params"] = int(self.token_module_params)

        # Stable test-image IDs are required for truly paired resampling across
        # the RGB/HSV and token-model harnesses.
        test_root = (Path(self.cfg.data_root) / "test").resolve()
        sample_ids = []
        for path, _label in self.test_loader.dataset.samples:
            pp = Path(path).resolve()
            try:
                sid = pp.relative_to(test_root).as_posix()
            except ValueError:
                sid = pp.name
            sample_ids.append(sid)
        if len(sample_ids) != len(test_m["targets"]):
            raise RuntimeError("test sample ID count does not match evaluated targets")
        test_m["sample_ids"] = sample_ids

        print(f"\n Test Results:")
        print(f"   Acc@1    : {test_m['acc1']:.4f}")
        print(f"   Macro-F1 : {test_m['macro_f1']:.4f}")
        auc = test_m["macro_auc"]
        print(f"   Macro-AUC: {auc:.4f}" if auc >= 0 else "   Macro-AUC: N/A")
        print(f"   Params   : {test_m['params_m']:.2f}M")
        print(f"   GFLOPs   : {test_m['gflops']:.2f}" if test_m["gflops"] >= 0 else "   GFLOPs   : N/A")

        # Save artifacts using CheckpointManager
        self.ckpt_manager.copy_final_artifacts(
            best_ckpt_path = best_path,
            config         = self.cfg,
            train_log      = self.train_log,
            test_metrics   = test_m,
            class_to_idx   = {c: i for i, c in enumerate(TeaLeafDataset.CLASS_NAMES)},
            classes        = TeaLeafDataset.CLASS_NAMES,
            logs_dir       = self.logs_dir,
        )
        return test_m


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Stage-4 token modules and matched controls on Swin-S",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp_name",  type=str, required=True)
    p.add_argument("--run_dir",   type=str,
                   default="/kaggle/working/experiments_tcca")
    p.add_argument("--data_root", type=str,
                   default="/kaggle/working/tea_leaf_clean")
    p.add_argument("--mask_dir",  type=str, default="",
                   help="Path to pseudo-mask directory. "
                        "Required when --use_seg is set.")

    p.add_argument("--arch", type=str, default="tcca",
                   choices=["tcca", "rgb", "adapter", "eca"],
                   help="tcca: token-level attention module (TLA/TCCA). "
                        "rgb: recipe control, backbone+classifier only. "
                        "adapter: parameter-matched non-attention capacity control; eca: lightweight attention comparator.")
    p.add_argument("--adapter_expansion", type=int, default=3,
                   help="MLP expansion for --arch adapter. 3 == exact TCCA param match.")
    p.add_argument("--use_hsv",   action="store_true",
                   help="Enable HSV color branch.")
    p.add_argument("--hsv_raw",   action="store_true",
                   help="Use raw HSV (3-ch). Default: sin/cos (4-ch).")
    p.add_argument("--use_seg",   action="store_true",
                   help="Enable FPN segmentation head (Phase 3c).")

    p.add_argument("--pretrained",         action="store_true")
    p.add_argument("--drop_path_rate",     type=float, default=0.2)
    p.add_argument("--batch_size",         type=int,   default=48)
    p.add_argument("--epochs",             type=int,   default=80)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--num_workers",        type=int,   default=2)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--use_ema",            action="store_true")
    p.add_argument("--gate_warmup_epochs", type=int,   default=5)
    p.add_argument("--hue_jitter", type=float, default=0.0)
    p.add_argument("--lambda_seg",         type=float, default=0.5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--compute_val_auc",    action="store_true")
    p.add_argument("--ckpt_temp_dir",      type=str,   default="/kaggle/working/temp")
    p.add_argument("--resume",             type=str,   default=None)
    p.add_argument("--auto_resume",        action="store_true")
    p.add_argument("--save_epoch_checkpoints", action="store_true")
    p.add_argument("--no_dp",             action="store_true")
    p.add_argument("--strict_final_counts", action="store_true",
                   help="Assert the frozen audited 7714-instance split counts before training.")

    return p.parse_args()


# =============================================================================
# Optional cleaned-protocol sanity check
# =============================================================================
def assert_clean_tealeafbd_counts(data_root: Union[str, Path]) -> None:
    """Assert the frozen audited 7,714-instance experiment partition."""
    root = Path(data_root)
    expected = {
        "train": {
            "Brown Blight": 858, "Gray Blight": 896,
            "Green mirid bug": 897, "Healthy leaf": 893,
            "Helopeltis": 863, "Red spider": 835,
            "Tea algal leaf spot": 848,
        },
        "val": {
            "Brown Blight": 83, "Gray Blight": 163,
            "Green mirid bug": 185, "Healthy leaf": 147,
            "Helopeltis": 86, "Red spider": 76,
            "Tea algal leaf spot": 76,
        },
        "test": {
            "Brown Blight": 86, "Gray Blight": 160,
            "Green mirid bug": 179, "Healthy leaf": 141,
            "Helopeltis": 89, "Red spider": 75,
            "Tea algal leaf spot": 78,
        },
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not root.exists():
        raise FileNotFoundError(f"Audited data_root not found: {root}")
    grand_total = 0
    for split, cls_counts in expected.items():
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        split_total = 0
        for cls, exp_n in cls_counts.items():
            cls_dir = split_root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")
            got_n = sum(1 for q in cls_dir.iterdir() if q.suffix.lower() in exts)
            if got_n != exp_n:
                raise AssertionError(
                    f"Unexpected count for {split}/{cls}: got {got_n}, expected {exp_n}"
                )
            split_total += got_n
        if split_total != sum(cls_counts.values()):
            raise AssertionError(f"Unexpected {split} total: {split_total}")
        grand_total += split_total
    if grand_total != 7714:
        raise AssertionError(f"Unexpected audited dataset total: {grand_total}, expected 7714")
    print("Audited TeaLeaf working-set count check passed: "
          "train=6090, val=816, test=808, total=7714")


def main():
    args = parse_args()

    if getattr(args, "strict_final_counts", False):
        assert_clean_tealeafbd_counts(args.data_root)

    cfg = TCCAConfig(
        pretrained             = args.pretrained,
        arch                   = args.arch,
        adapter_expansion      = args.adapter_expansion,
        drop_path_rate         = args.drop_path_rate,
        use_hsv                = args.use_hsv,
        hsv_use_sincos         = (not args.hsv_raw),
        use_seg                = args.use_seg,
        data_root              = args.data_root,
        mask_dir               = args.mask_dir,
        batch_size             = args.batch_size,
        epochs                 = args.epochs,
        lr                     = args.lr,
        num_workers            = args.num_workers,
        seed                   = args.seed,
        use_ema                = args.use_ema,
        gate_warmup_epochs     = args.gate_warmup_epochs,
        hue_jitter              = args.hue_jitter,
        lambda_seg             = args.lambda_seg,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        compute_val_auc        = args.compute_val_auc,
        run_dir                = args.run_dir,
        experiment_name        = args.exp_name,
        ckpt_temp_dir          = args.ckpt_temp_dir,
        save_epoch_checkpoints = args.save_epoch_checkpoints,
        use_data_parallel      = (not args.no_dp),
    )

    print("=" * 80)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"PyTorch: {torch.__version__}  |  timm: {timm.__version__}")
    print("=" * 80)

    trainer = TCCATrainer(cfg)

    if args.resume:
        trainer.resume_from(Path(args.resume))
    elif args.auto_resume:
        last = trainer.ckpt_manager.temp_dir / "last_checkpoint.pth"
        if last.exists():
            trainer.resume_from(last)

    trainer.fit()


if __name__ == "__main__":
    main()