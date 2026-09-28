
"""
model.py — model architectures for the tea-leaf classifier.

Serves two variants matching the training-code run families:

  SwinTCCA   — arch="tcca", use_hsv=False   (training runs: TLA_*)
               Swin-S + Stage-4 Token-Level Chromatic Cross-Attention.

  SwinRGB    — arch="rgb"                    (training runs: R0-R3, C1)
               Swin-S backbone + classifier head, no token module.
               This is the best-performing checkpoint in the campaign
               (results/v2/runs/C1_s1337) and is what the API serves.

Both produce state-dict key layouts identical to the training code, so
`load_state_dict(sd, strict=True)` succeeds 1:1 against the matching
`ema_model_weights_only.pth` / `model_weights_only.pth`.
"""

import math
import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple


# ── Building blocks (shared) ──────────────────────────────────────────────────

class ChromaticCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_tokens, color_tokens):
        attended, _ = self.attn(rgb_tokens, color_tokens, color_tokens)
        return self.norm(rgb_tokens + attended)


class Stage4TCCAFusion(nn.Module):
    def __init__(self, color_dim: int = 768, dim: int = 768,
                 num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(color_dim, dim)
        self.attn = ChromaticCrossAttention(dim, num_heads, attn_dropout)
        self.gate = nn.Parameter(torch.full((1,), -2.0))

    def forward(self, rgb_tokens, color_feat, gate_alpha=1.0):
        c = color_feat.flatten(2).transpose(1, 2)
        c = self.proj(c).to(dtype=rgb_tokens.dtype)
        g = float(gate_alpha) * torch.sigmoid(self.gate)
        return rgb_tokens + g * self.attn(rgb_tokens, c)


class SwinWithHooks(nn.Module):
    """timm Swin-S wrapper returning the final token sequence."""
    def __init__(self, timm_id: str, pretrained: bool, drop_path_rate: float):
        super().__init__()
        self.swin = timm.create_model(
            timm_id,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            global_pool="",
        )
        self._stage_outputs: Dict[int, torch.Tensor] = {}
        self._register_stage_hooks()

    def _register_stage_hooks(self):
        for stage_idx in range(4):
            stage = self.swin.layers[stage_idx]
            def make_hook(idx):
                def hook(module, input, output):
                    self._stage_outputs[idx] = output
                return hook
            stage.register_forward_hook(make_hook(stage_idx))

    def forward(self, x):
        self._stage_outputs.clear()
        normalized = self.swin(x)

        def normalize_to_tokens(t):
            if t.dim() == 3:
                return t
            elif t.dim() == 4:
                if t.shape[-1] < t.shape[1]:
                    B_, C, H, W = t.shape
                    return t.permute(0, 2, 3, 1).reshape(B_, H * W, C)
                else:
                    B_, H, W, C = t.shape
                    return t.reshape(B_, H * W, C)

        tokens4 = normalize_to_tokens(normalized)
        pooled  = tokens4.mean(dim=1)
        return {"tokens4": tokens4, "pooled": pooled}


# ── Classifier head (shared between the two variants) ─────────────────────────

def _make_classifier(num_classes: int, fuse_dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(fuse_dropout),
        nn.Linear(768, 384),
        nn.GELU(),
        nn.Dropout(fuse_dropout),
        nn.Linear(384, num_classes),
    )


SWIN_ID = "swin_small_patch4_window7_224.ms_in1k"


# ── arch="rgb" — served model (best-performing run: C1_s1337) ─────────────────

class SwinRGB(nn.Module):
    """
    Swin-S backbone + classifier head.  No token module.

    Matches the state_dict of training runs with arch="rgb"
    (family prefixes R0-R3 and C1).  Example checkpoint:
        results/v2/runs/C1_s1337/model/ema_model_weights_only.pth
    Top-level state_dict prefixes: {img_mean, img_std, swin, classifier}.

    forward() accepts gate_alpha for call-site parity with SwinTCCA, so
    the XAI plumbing in logic.py works unchanged.
    """
    SWIN_ID = SWIN_ID

    def __init__(
        self,
        num_classes: int = 7,
        drop_path_rate: float = 0.2,
        fuse_dropout: float = 0.2,
        img_mean: Tuple = (0.485, 0.456, 0.406),
        img_std:  Tuple = (0.229, 0.224, 0.225),
        pretrained: bool = False,
    ):
        super().__init__()
        self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std",  torch.tensor(img_std).view(1, 3, 1, 1))

        self.swin = SwinWithHooks(
            timm_id=self.SWIN_ID,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )
        self.classifier = _make_classifier(num_classes, fuse_dropout)

    # Compatibility alias used by the Grad-CAM target-layer selector.
    @property
    def backbone(self) -> nn.Module:
        return self.swin.swin

    def forward(self, x_norm, gate_alpha: float = 1.0):
        swin_out = self.swin(x_norm)
        return self.classifier(swin_out["tokens4"].mean(dim=1))


# ── arch="tcca", use_hsv=False — TLA runs ─────────────────────────────────────

class SwinTCCA(nn.Module):
    """
    RGB self-attention TLA variant — arch="tcca", use_hsv=False.
    Matches training runs with prefix TLA (e.g. TLA_s2026, TLA_s1337).
    Top-level state_dict prefixes:
        {img_mean, img_std, swin, color_proj4, tcca, classifier}.
    """
    SWIN_ID = SWIN_ID

    def __init__(
        self,
        num_classes: int = 7,
        drop_path_rate: float = 0.2,
        fuse_dropout: float = 0.2,
        img_mean: Tuple = (0.485, 0.456, 0.406),
        img_std:  Tuple = (0.229, 0.224, 0.225),
        pretrained: bool = False,
    ):
        super().__init__()
        self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std",  torch.tensor(img_std).view(1, 3, 1, 1))

        self.swin = SwinWithHooks(
            timm_id=self.SWIN_ID,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )
        self.color_proj4 = nn.Linear(768, 768)
        self.tcca        = Stage4TCCAFusion(color_dim=768, dim=768, num_heads=8)
        self.classifier  = _make_classifier(num_classes, fuse_dropout)

    @property
    def backbone(self) -> nn.Module:
        return self.swin.swin

    def forward(self, x_norm, gate_alpha: float = 1.0):
        swin_out = self.swin(x_norm)
        tokens4  = swin_out["tokens4"]

        B = x_norm.shape[0]
        c4          = self.color_proj4(tokens4)
        color_feat4 = c4.transpose(1, 2).reshape(B, 768, 7, 7)

        tokens4 = self.tcca(tokens4, color_feat4, gate_alpha)
        return self.classifier(tokens4.mean(dim=1))
