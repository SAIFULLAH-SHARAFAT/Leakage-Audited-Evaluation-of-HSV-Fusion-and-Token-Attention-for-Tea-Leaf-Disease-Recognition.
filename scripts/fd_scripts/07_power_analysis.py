#!/usr/bin/env python3
"""Pre-registered power analysis and stopping rule for the future colour campaign.

Answers one question before any future architecture is built: on a given
evaluation set, could a new model's gain even be detected?

From the completed runs of an existing campaign it measures
  * the contestable headroom: oracle macro-F1 (an image counts as correct if any
    arm gets it right) minus the best single arm's macro-F1;
  * the images every arm gets wrong, and how often they agree on the wrong label;
  * how correlated independently trained arms' errors already are -- the ceiling
    on what a two-head confidence mixture (F3) can recover;
  * the paired-bootstrap resolution of the test set: CI half-width, standard
    error, and the minimum detectable effect at 80% power (MDE80 = 2.80 x SE).

It then applies the stopping rule declared in configs/future_designs.yaml
(go_no_go.power): the future campaign is worth running on an evaluation set only
if MDE80 <= max_mde80_fraction_of_headroom x headroom.

Read-only on the runs it is pointed at. The runs directory is a required
argument so this design-only tree hardcodes no current-campaign path.

USAGE
    python scripts/fd_scripts/07_power_analysis.py --runs-dir results/v2/runs --seed 42
    python scripts/fd_scripts/07_power_analysis.py --runs-dir results/v2/runs --pair TLA C1
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
Z80 = 1.959964 + 0.841621          # two-sided alpha = 0.05, power = 0.80


def components():
    spec = importlib.util.spec_from_file_location(
        "components", Path(__file__).parent / "01_shared_color_components.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 over a fixed label set, zero_division=0 -- as 07_paired_bootstrap uses."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def load_runs(runs_dir: Path, seed: int) -> tuple[dict, dict, list[str], np.ndarray, int]:
    arms, family, ids, y, campaign = {}, {}, None, None, set()
    for run in sorted(runs_dir.glob(f"*_s{seed}")):
        if not (run / "metrics" / "test_results.json").exists():
            continue
        cfg = json.loads((run / "config" / "config.json").read_text())
        campaign.add(cfg.get("campaign_id"))
        # Recipe family, read from the run itself: only the token trainer records `arch`.
        family[run.name.rsplit("_s", 1)[0]] = "token" if "arch" in cfg else "hsv"
        raw = run / "raw_outputs"
        rid = json.loads((raw / "test_ids.json").read_text())
        ry = np.load(raw / "test_targets.npy")
        if ids is None:
            ids, y = rid, ry
        elif rid != ids or not np.array_equal(ry, y):
            raise SystemExit(f"{run.name} does not share the test set of the other runs")
        arms[run.name.rsplit("_s", 1)[0]] = np.load(raw / "test_predictions.npy")
    if len(arms) < 2:
        raise SystemExit(f"Need at least two completed runs for seed {seed} in {runs_dir}")
    if len(campaign) != 1:
        raise SystemExit(f"Runs come from more than one campaign: {sorted(map(str, campaign))}")
    k = len(json.loads((next(runs_dir.glob(f'*_s{seed}')) / "config" / "classes.json").read_text()))
    return arms, family, ids, y, k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, help="Completed runs of an existing campaign")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="Arms to resolve; repeatable. Default: every within-family pair "
                         "(arms trained under different recipes are never compared).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fd = yaml.safe_load((REPO / "configs" / "future_designs.yaml").read_text())
    rule = fd["go_no_go"]["power"]
    arms, family, ids, y, k = load_runs(Path(args.runs_dir), args.seed)
    names = sorted(arms)

    from sklearn.metrics import f1_score
    check = names[0]
    ref = f1_score(y, arms[check], labels=np.arange(k), average="macro", zero_division=0)
    if abs(ref - macro_f1(y, arms[check], k)) > 1e-12:
        raise SystemExit("Fast macro-F1 disagrees with scikit-learn; refusing to continue")

    mf1 = {a: macro_f1(y, arms[a], k) for a in names}
    best = max(names, key=mf1.get)
    correct_any = np.zeros(len(y), bool)
    for a in names:
        correct_any |= arms[a] == y
    oracle = np.where(correct_any, y, arms[best])
    oracle_mf1 = macro_f1(y, oracle, k)
    headroom = oracle_mf1 - mf1[best]

    wrong_all = ~correct_any
    unanimous = sum(len({int(arms[a][i]) for a in names}) == 1 for i in np.where(wrong_all)[0])

    div = components().prediction_diversity
    phis = {f"{a}|{b}": div(y, arms[a], arms[b])["error_phi"] for a, b in itertools.combinations(names, 2)}

    pairs = args.pair or [(a, b) for a, b in itertools.combinations(names, 2)
                          if family[a] == family[b]]
    rng = np.random.default_rng(args.rng_seed)
    n = len(y)
    idx = rng.integers(0, n, size=(args.n_boot, n))          # one shared resample set
    resolved = []
    for a, b in pairs:
        if a not in arms or b not in arms:
            raise SystemExit(f"Unknown arm in pair {a} {b}; available: {names}")
        deltas = np.array([macro_f1(y[i], arms[a][i], k) - macro_f1(y[i], arms[b][i], k) for i in idx])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        se = float(deltas.std(ddof=1))
        resolved.append({"a": a, "b": b, "observed_delta": round(mf1[a] - mf1[b], 5),
                         "ci95": [round(float(lo), 5), round(float(hi), 5)],
                         "ci_half_width": round(float(hi - lo) / 2, 5),
                         "bootstrap_se": round(se, 5), "mde80": round(Z80 * se, 5)})

    mde_median = float(np.median([r["mde80"] for r in resolved]))
    threshold = rule["max_mde80_fraction_of_headroom"] * headroom
    decision = "GO" if mde_median <= threshold else "NO-GO"

    result = {
        "runs_dir": str(args.runs_dir), "seed": args.seed, "arms": names, "n_test": n,
        "metric": "macro_f1", "n_boot": args.n_boot,
        "best_arm": best, "best_macro_f1": round(mf1[best], 5),
        "oracle_macro_f1": round(oracle_mf1, 5), "contestable_headroom": round(headroom, 5),
        "images_wrong_in_every_arm": int(wrong_all.sum()),
        "of_which_unanimous_on_the_wrong_label": int(unanimous),
        "error_phi_between_independent_arms": {
            "min": round(min(phis.values()), 4), "median": round(float(np.median(list(phis.values()))), 4),
            "max": round(max(phis.values()), 4), "pairs": {p: round(v, 4) for p, v in phis.items()}},
        "resolution": resolved,
        "stopping_rule": {"rule": rule["statement"],
                          "max_mde80_fraction_of_headroom": rule["max_mde80_fraction_of_headroom"],
                          "median_mde80": round(mde_median, 5), "threshold": round(threshold, 5),
                          "decision": decision},
    }
    out = Path(args.out) if args.out else (
        REPO / fd["defaults"]["tables_dir"] / f"power_analysis_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"arms: {', '.join(names)}  (n_test={n}, {args.n_boot} paired resamples)")
    print(f"best arm {best}: macro-F1 {mf1[best]:.4f} | oracle {oracle_mf1:.4f} | "
          f"contestable headroom {headroom*100:.2f} points")
    print(f"wrong in every arm: {int(wrong_all.sum())} (unanimous on the wrong label: {unanimous})")
    print(f"error phi between independently trained arms: median {result['error_phi_between_independent_arms']['median']:.3f} "
          f"(range {result['error_phi_between_independent_arms']['min']:.3f}-"
          f"{result['error_phi_between_independent_arms']['max']:.3f})")
    for r in resolved:
        print(f"  {r['a']:4s} vs {r['b']:4s}  delta {r['observed_delta']:+.4f}  "
              f"CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  MDE80 {r['mde80']*100:.2f} points")
    print(f"\nstopping rule: median MDE80 {mde_median*100:.2f} points vs threshold "
          f"{threshold*100:.2f} points ({rule['max_mde80_fraction_of_headroom']:.0%} of headroom) -> {decision}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
