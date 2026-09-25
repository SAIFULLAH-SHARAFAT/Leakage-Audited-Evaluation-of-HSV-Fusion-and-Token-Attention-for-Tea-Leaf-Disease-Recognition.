#!/usr/bin/env python3
"""Fail-loud structural validation for every declared v2 model family.

This script deliberately uses ``pretrained=False`` so it never needs network
access.  In addition to parameter-budget checks, it verifies the most important
architecture contract in the corrected campaign: the token-family RGB control
(C1) must reproduce the RGB Swin baseline representation/logits when both are
given identical backbone and classifier weights.  It also verifies that the
final Swin LayerNorm participates in C1 back-propagation.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import train_swin_three_models as S  # noqa: E402
import train_token_models as T  # noqa: E402


def nparams(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _finite_logits(name: str, model: torch.nn.Module, x: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (x.shape[0], 7):
        raise RuntimeError(f"{name}: unexpected output shape {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise RuntimeError(f"{name}: non-finite logits")


def _r0_c1_equivalence(r0: S.SwinRGBHSV, c1: T.SwinTCCA) -> None:
    """Prove that corrected C1 consumes the same final-normalized Swin feature.

    The two timm backbones use different global-pool settings, so equality of
    parameter counts is insufficient.  We copy identical weights, compare the
    pooled final representation and end-to-end logits, then prove the final
    LayerNorm receives gradient in C1.
    """
    # Global pooling changes no parameters.  These state dicts should therefore
    # be exactly compatible; strict=True makes any timm API drift fatal.
    c1.swin.swin.load_state_dict(r0.backbone.state_dict(), strict=True)
    c1.classifier.load_state_dict(r0.classifier.state_dict(), strict=True)

    torch.manual_seed(20260920)
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    r0.eval()
    c1.eval()

    with torch.no_grad():
        feat_r0 = r0.backbone(x)
        c1_out = c1.swin(x)
        feat_c1 = c1_out["pooled"]
        logits_r0 = r0(x)
        logits_c1 = c1(x)

    torch.testing.assert_close(
        feat_r0, feat_c1, rtol=1e-5, atol=1e-6,
        msg="C1 pooled feature does not reproduce the final-normalized RGB Swin feature",
    )
    torch.testing.assert_close(
        logits_r0, logits_c1, rtol=1e-5, atol=1e-6,
        msg="R0 and C1 logits differ under identical backbone/head weights",
    )

    # Eval mode disables dropout/drop-path but still allows gradient tracking.
    c1.zero_grad(set_to_none=True)
    logits = c1(x)
    logits.square().mean().backward()
    norm = getattr(c1.swin.swin, "norm", None)
    if norm is None:
        raise RuntimeError("C1 timm Swin has no final .norm module; API contract changed")
    grads = [p.grad for p in norm.parameters() if p.requires_grad]
    if not grads or any(g is None for g in grads):
        raise RuntimeError("C1 final Swin LayerNorm did not receive gradients")
    if not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("C1 final Swin LayerNorm gradient is non-finite")
    if sum(float(g.abs().sum()) for g in grads) <= 0.0:
        raise RuntimeError("C1 final Swin LayerNorm gradient is identically zero")

    print("PASS: R0/C1 forward equivalence under identical weights")
    print("PASS: C1 final Swin LayerNorm receives finite non-zero gradient")


def main() -> None:
    common_swin = dict(
        model_name="swin_small_patch4_window7_224.ms_in1k",
        pretrained=False,
        drop_path_rate=0.2,
        num_classes=7,
    )

    # --- Primary RGB control equivalence ---
    r0 = S.SwinRGBHSV(**common_swin, use_hsv_branch=False)
    c1 = T.SwinTCCA(
        arch="rgb", pretrained=False, drop_path_rate=0.2, num_classes=7
    )
    if nparams(r0) != nparams(c1):
        raise RuntimeError(
            f"R0/C1 total parameter mismatch: {nparams(r0):,} != {nparams(c1):,}"
        )
    _r0_c1_equivalence(r0, c1)

    counts: dict[str, int] = {"R0": nparams(r0), "C1": nparams(c1)}
    del r0, c1
    gc.collect()

    # --- Instantiate all HSV variants so every declared arm is contract-tested ---
    hsv_specs = {
        "R1": dict(use_hsv_branch=True, hsv_use_sincos=True, gate_vector=True),
        "R2": dict(use_hsv_branch=True, hsv_use_sincos=False, gate_vector=False),
        "R3": dict(use_hsv_branch=True, hsv_use_sincos=True, gate_vector=False),
    }
    torch.manual_seed(20260921)
    x_small = torch.randn(1, 3, 224, 224)
    for name, kw in hsv_specs.items():
        m = S.SwinRGBHSV(**common_swin, **kw)
        counts[name] = nparams(m)
        _finite_logits(name, m, x_small)
        del m
        gc.collect()

    # --- Token mechanisms and exact capacity control ---
    D = 768
    expected_matched = 6 * D * D + 8 * D + 1
    if expected_matched != 3_545_089:
        raise RuntimeError(f"Internal matched-budget formula drifted: {expected_matched:,}")

    tla = T.SwinTCCA(
        arch="tcca", pretrained=False, drop_path_rate=0.2, num_classes=7
    )
    tla_active = nparams(tla.tcca) + nparams(tla.color_proj4)
    counts["TLA"] = nparams(tla)
    if tla_active != expected_matched:
        raise RuntimeError(
            f"TLA active parameter mismatch: {tla_active:,} != {expected_matched:,}"
        )
    _finite_logits("TLA", tla, x_small)
    del tla
    gc.collect()

    c2 = T.SwinTCCA(
        arch="adapter", pretrained=False, drop_path_rate=0.2,
        num_classes=7, adapter_expansion=3,
    )
    c2_active = nparams(c2.adapter)
    counts["C2"] = nparams(c2)
    if c2_active != expected_matched:
        raise RuntimeError(
            f"C2 active parameter mismatch: {c2_active:,} != {expected_matched:,}"
        )
    _finite_logits("C2", c2, x_small)
    del c2
    gc.collect()

    c4 = T.SwinTCCA(
        arch="eca", pretrained=False, drop_path_rate=0.2, num_classes=7
    )
    c4_active = nparams(c4.eca)
    counts["C4"] = nparams(c4)
    if c4_active <= 0:
        raise RuntimeError("C4 ECA comparator has no active parameters")
    if c4_active >= expected_matched:
        raise RuntimeError(
            f"C4 is intended to be lightweight but has {c4_active:,} active parameters"
        )
    _finite_logits("C4", c4, x_small)
    del c4
    gc.collect()

    print("\nTOTAL PARAMETER COUNTS")
    print("-" * 48)
    for name in ("R0", "R1", "R2", "R3", "C1", "TLA", "C2", "C4"):
        print(f"{name:4s} {counts[name]:,}")

    print("\nACTIVE TOKEN-MODULE CONTRACTS")
    print("-" * 48)
    print(f"C1 active      : 0")
    print(f"TLA active     : {tla_active:,}")
    print(f"C2 active      : {c2_active:,}")
    print(f"C4 active      : {c4_active:,}  (lightweight; intentionally unmatched)")

    print("\nPASS: all eight architecture contracts verified.")


if __name__ == "__main__":
    main()
