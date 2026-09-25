#!/usr/bin/env python3
"""Parameter and compute accounting for every future design (F0-F5).

The design documents describe the colour branch as lightweight -- and in
parameters it is (0.024-0.060 M). But every design except F3 runs a second
trajectory through the shared Swin weights, so compute does NOT stay small. This
script puts both numbers side by side so neither can be quoted without the other.

  * Parameters come from the real component classes in
    01_shared_color_components.py, and are asserted to equal the extra_params
    declared in configs/future_designs.yaml, so the config cannot drift.
  * Compute is an analytic multiply-accumulate (MAC) count for Swin-S at 224x224
    (patch 4, window 7, dims 96/192/384/768, depths 2/2/18/2, MLP ratio 4),
    counting linear layers, windowed-attention matmuls, patch merging, the patch
    stem and a shared-head readout. Norms, softmax and element-wise adds are
    omitted except where a design's own element-wise couplings are its cost.
  * The analytic RGB baseline is cross-checked against ptflops on the actual timm
    model when ptflops is installed.

Latency cannot be measured for F0-F5 until the timm block adapter exists; the
script says so rather than estimating it. --measure-latency times the RGB
baseline and the component primitives on the chosen device.

USAGE
    python scripts/fd_scripts/08_cost_accounting.py
    python scripts/fd_scripts/08_cost_accounting.py --measure-latency --device cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DIMS, DEPTHS, MLP, WIN2, PATCH, IMG = [96, 192, 384, 768], [2, 2, 18, 2], 4, 49, 4, 224
TOKENS = [(IMG // PATCH // 2 ** s) ** 2 for s in range(4)]          # 3136, 784, 196, 49
HEAD = 768 * 384 + 384 * 7                                           # shared MLP head


def components():
    spec = importlib.util.spec_from_file_location(
        "components", Path(__file__).parent / "01_shared_color_components.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blocks():
    for n, d, depth in zip(TOKENS, DIMS, DEPTHS):
        for _ in range(depth):
            yield n, d


def merges() -> int:
    return sum(2 * n * d * d for n, d in zip(TOKENS[:3], DIMS[:3]))


def stem(in_ch: int) -> int:
    return TOKENS[0] * DIMS[0] * in_ch * PATCH * PATCH


def attn_matmuls() -> int:
    return sum(2 * WIN2 * n * d for n, d in blocks())


def baseline_macs() -> int:
    """RGB Swin-S trunk: 12*N*D^2 linear + 2*49*N*D windowed attention per block."""
    return sum(12 * n * d * d + 2 * WIN2 * n * d for n, d in blocks()) + merges() + stem(3)


def branch_macs(design: str) -> int:
    """Extra MACs the colour branch adds on top of the RGB baseline."""
    per_block = {
        "F0": lambda n, d: 12 * n * d * d + 2 * WIN2 * n * d,                 # own Q, K, V
        "F1": lambda n, d: 10 * n * d * d + 2 * WIN2 * n * d + 6 * n * d,    # own Q; shared K, V; a/b/c
        "F2": lambda n, d: 10 * n * d * d + 2 * WIN2 * n * d + 2 * n * d,    # own Q; shared K, V; a/c
        "F4": lambda n, d: 8 * n * d * d + 6 * n * d,                         # no attention; g, b, c
    }
    if design in ("F3", "F5"):
        return branch_macs("F1")                                             # F3 reads out F1; F5 == F1 graph
    return sum(per_block[design](n, d) for n, d in blocks()) + merges() + stem(4) + HEAD


def design_params(m) -> dict:
    count = lambda mod: sum(p.numel() for p in mod.parameters())
    s = count(m.HSVPatchStem(4, 96, 4))
    per = [d for n, d in blocks()]
    abc = sum(count(m.LayerCoupling(d, MLP, "abc")) for d in per)
    ac = sum(count(m.LayerCoupling(d, MLP, "ac")) for d in per)
    # F4: g replaces a (both D-sized); the b/c couplings stay.
    g = sum(count(m.GatedAttentionCopy(d)) for d in per)
    bc = sum(d * MLP + d for d in per)
    return {"F0": s, "F1": s + abc, "F2": s + ac, "F3": s + abc, "F4": s + g + bc, "F5": s + abc}


def ptflops_baseline() -> int | None:
    try:
        import timm
        from ptflops import get_model_complexity_info
    except ImportError:
        return None
    model = timm.create_model("swin_small_patch4_window7_224.ms_in1k", pretrained=False,
                              num_classes=0).eval()
    macs, _ = get_model_complexity_info(model, (3, IMG, IMG), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
    return int(macs)


def latency(device: str, reps: int) -> dict:
    import torch
    import timm
    m = components()
    out = {"device": device}
    model = timm.create_model("swin_small_patch4_window7_224.ms_in1k", pretrained=False,
                              num_classes=0).eval().to(device)
    x = torch.randn(1, 3, IMG, IMG, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        out["rgb_swin_s_ms_per_image"] = round((time.perf_counter() - t) / reps * 1000, 2)
        stem_mod = m.HSVPatchStem(4, 96, 4).eval().to(device)
        rep = m.rgb_luma_rep(torch.rand(1, 3, IMG, IMG, device=device))
        t = time.perf_counter()
        for _ in range(reps):
            stem_mod(rep)
        out["branch_stem_ms_per_image"] = round((time.perf_counter() - t) / reps * 1000, 3)
    out["F0_F5_latency"] = "not measurable until the timm block adapter exists"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-latency", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fd = yaml.safe_load((REPO / "configs" / "future_designs.yaml").read_text())
    declared = {k: v["extra_params"] for k, v in fd["future_campaign"]["designs"].items()}
    params = design_params(components())
    mismatch = {k: (params[k], declared.get(k)) for k in params if params[k] != declared.get(k)
                and not (k == "F3" and declared.get(k) == 0)}
    # F3 is declared as 0 extra over F1 (the mixture has no parameters); its total equals F1.
    if mismatch:
        raise SystemExit(f"extra_params in future_designs.yaml disagree with the components: {mismatch}")

    base = baseline_macs()
    rows = {}
    for k in ("F0", "F1", "F2", "F3", "F4", "F5"):
        extra = branch_macs(k)
        rows[k] = {"extra_params": params[k] if k != "F3" else 0,
                   "total_extra_params_vs_rgb": params[k],
                   "attention_passes": fd["future_campaign"]["designs"][k].get("attention_passes", 2),
                   "extra_gmacs": round(extra / 1e9, 3),
                   "extra_compute_pct_of_rgb": round(100 * extra / base, 1)}

    pt = ptflops_baseline()
    linear_only = base - attn_matmuls()
    check = None
    if pt is not None:
        check = {"ptflops_gmacs": round(pt / 1e9, 3),
                 "analytic_gmacs": round(base / 1e9, 3),
                 "analytic_without_window_attention_matmuls_gmacs": round(linear_only / 1e9, 3),
                 "closest": "without_attention_matmuls" if abs(pt - linear_only) < abs(pt - base) else "full"}

    result = {"model": "swin_small_patch4_window7_224 @224", "rgb_baseline_gmacs": round(base / 1e9, 3),
              "window_attention_matmul_share_pct": round(100 * attn_matmuls() / base, 2),
              "designs": rows, "ptflops_cross_check": check,
              "latency": latency(args.device, args.reps) if args.measure_latency else None}
    out = Path(args.out) if args.out else REPO / fd["defaults"]["tables_dir"] / "cost_accounting.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"RGB Swin-S baseline: {base/1e9:.3f} GMACs (analytic; windowed-attention matmuls are "
          f"{result['window_attention_matmul_share_pct']:.1f}% of it)")
    if check:
        print(f"ptflops on the timm model: {check['ptflops_gmacs']:.3f} GMACs -> matches the analytic count "
              f"{'WITHOUT' if check['closest'] == 'without_attention_matmuls' else 'WITH'} attention matmuls "
              f"({check['analytic_without_window_attention_matmuls_gmacs']:.3f} vs {check['analytic_gmacs']:.3f})")
    print(f"\n{'design':6s} {'extra params':>13s} {'attn passes':>12s} {'extra GMACs':>12s} {'extra compute':>14s}")
    for k, r in rows.items():
        print(f"{k:6s} {r['total_extra_params_vs_rgb']:>13,} {r['attention_passes']:>12d} "
              f"{r['extra_gmacs']:>12.3f} {r['extra_compute_pct_of_rgb']:>13.1f}%")
    print("\nparameters verified against configs/future_designs.yaml")
    if result["latency"]:
        print(f"latency: {result['latency']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
