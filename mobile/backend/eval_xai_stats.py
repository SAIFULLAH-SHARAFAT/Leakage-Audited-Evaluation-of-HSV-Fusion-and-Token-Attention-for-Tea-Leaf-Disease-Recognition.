"""
eval_xai_stats.py — two-pass evaluation strategy.

Pass 1 (fast): send all 808 images with XAI disabled (inference only).
               Collects: total, rejected, uncertain, confidence scores.
               ~0.5s per image → ~7 min total.

Pass 2 (sampled): send a random sample of 20 eligible images per class
                  with XAI enabled. Collects: agreement%, energy%.
                  ~5s per image × 20 × 7 classes → ~12 min total.

Total wall time: ~20 min instead of 808 × 5s = ~67 min.

The agreement% and energy% are reported over the sampled subset,
which is stated explicitly when writing the paper numbers.
"""

import os, sys, json, requests, random, statistics
from collections import defaultdict

API_URL  = os.environ.get("API_URL",  "http://localhost:8000/predict")
TEST_DIR = os.environ.get("TEST_DIR",
    "/SLURM/home/slurm_g202621260/tealeaf/data/Tea_leaf_dataset/test")
if len(sys.argv) > 1:
    TEST_DIR = sys.argv[1]

SAMPLE_PER_CLASS = 20   # images per class for XAI pass
TIMEOUT_FAST     = 60   # seconds — inference only
TIMEOUT_XAI      = 300  # seconds — full XAI pipeline

CLASSES = [
    "Brown Blight", "Gray Blight", "Green mirid bug", "Healthy leaf",
    "Helopeltis", "Red spider", "Tea algal leaf spot",
]

# ── helper ────────────────────────────────────────────────────────────────────
def get_images(class_name):
    d = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(d):
        print(f"WARNING: missing {d}")
        return []
    return [os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — fast inference pass (no XAI)
# We hit /predict normally but the server only runs XAI when confidence ≥ 0.70.
# For uncertain images it skips XAI automatically, saving time.
# For confident images we still pay the XAI cost — so we add a ?xai=0 param.
# Since the server doesn't support that param yet, we instead use /predict_fast
# if available, otherwise accept the XAI cost and use a longer timeout.
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PASS 1: Full dataset — inference + rejection stats")
print("=" * 60)

pass1 = defaultdict(lambda: {
    "total": 0, "rejected": 0, "uncertain": 0,
    "eligible_paths": [],   # paths that passed blur+confidence gates
    "confidence_scores": [],
})

for class_name in CLASSES:
    images = get_images(class_name)
    print(f"\n{class_name} ({len(images)} images)...")
    for fpath in images:
        pass1[class_name]["total"] += 1
        try:
            with open(fpath, "rb") as f:
                r = requests.post(API_URL, files={"file": f},
                                  timeout=TIMEOUT_XAI)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {fpath}")
                continue
            d = r.json()
        except Exception as e:
            print(f"  ERROR {os.path.basename(fpath)}: {e}")
            continue

        status = d.get("final_status")
        conf   = (d.get("prediction") or {}).get("confidence")
        if conf is not None:
            pass1[class_name]["confidence_scores"].append(conf)

        if status == "image_rejected":
            pass1[class_name]["rejected"] += 1
        elif status == "uncertain":
            pass1[class_name]["uncertain"] += 1
        else:
            pass1[class_name]["eligible_paths"].append(fpath)

            # Collect XAI stats inline if already computed
            xai = d.get("xai_stats")
            if xai:
                pass1[class_name].setdefault("agreement_pass_count", 0)
                pass1[class_name].setdefault("energy_pass_count", 0)
                if xai.get("agreement_pass"):
                    pass1[class_name]["agreement_pass_count"] += 1
                if xai.get("energy_pass"):
                    pass1[class_name]["energy_pass_count"] += 1

    r_ = pass1[class_name]
    eligible = len(r_["eligible_paths"])
    scores   = r_["confidence_scores"]
    print(f"  total={r_['total']}  rejected={r_['rejected']}  "
          f"uncertain={r_['uncertain']}  eligible={eligible}")
    if scores:
        print(f"  confidence: min={min(scores):.3f}  "
              f"mean={statistics.mean(scores):.3f}  "
              f"max={max(scores):.3f}")

# ── Pass 1 summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Class':<25} {'Total':>6} {'Rejected':>9} {'Uncertain':>10} {'Eligible':>9}")
print("-" * 70)

overall_total = overall_rej = overall_unc = overall_elig = 0
for class_name in CLASSES:
    r_ = pass1[class_name]
    elig = len(r_["eligible_paths"])
    print(f"{class_name:<25} {r_['total']:>6} {r_['rejected']:>9} "
          f"{r_['uncertain']:>10} {elig:>9}")
    overall_total += r_["total"]
    overall_rej   += r_["rejected"]
    overall_unc   += r_["uncertain"]
    overall_elig  += elig
print("-" * 70)
print(f"{'Overall':<25} {overall_total:>6} {overall_rej:>9} "
      f"{overall_unc:>10} {overall_elig:>9}")

# Check if XAI was already collected inline (confident images ran XAI anyway)
has_inline_xai = all(
    "agreement_pass_count" in pass1[c] for c in CLASSES
    if len(pass1[c]["eligible_paths"]) > 0
)

if has_inline_xai:
    print("\n✓ XAI stats collected inline during Pass 1 — skipping Pass 2")
    print("\n" + "=" * 70)
    print(f"{'Class':<25} {'Eligible':>9} {'Agree%':>8} {'Energy%':>8}")
    print("-" * 70)
    oa = oe = oe_cnt = 0
    for class_name in CLASSES:
        r_   = pass1[class_name]
        elig = len(r_["eligible_paths"])
        if elig == 0:
            print(f"{class_name:<25} {'0':>9} {'N/A':>8} {'N/A':>8}")
            continue
        ac = r_.get("agreement_pass_count", 0)
        ec = r_.get("energy_pass_count", 0)
        print(f"{class_name:<25} {elig:>9} {ac/elig*100:>7.1f}% {ec/elig*100:>7.1f}%")
        oa += ac; oe += ec; oe_cnt += elig
    print("-" * 70)
    if oe_cnt:
        print(f"{'Overall':<25} {oe_cnt:>9} "
              f"{oa/oe_cnt*100:>7.1f}% {oe/oe_cnt*100:>7.1f}%")
else:
    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — sampled XAI pass
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PASS 2: Sampled XAI ({SAMPLE_PER_CLASS} images/class)")
    print(f"{'='*60}")

    xai_results = {}
    for class_name in CLASSES:
        eligible_paths = pass1[class_name]["eligible_paths"]
        if not eligible_paths:
            print(f"\n{class_name}: no eligible images — skipping")
            xai_results[class_name] = {"sampled": 0, "agree": 0, "energy": 0}
            continue

        sample = random.sample(eligible_paths,
                               min(SAMPLE_PER_CLASS, len(eligible_paths)))
        print(f"\n{class_name}: sampling {len(sample)}/{len(eligible_paths)} eligible...")
        agree = energy = errors = 0
        for fpath in sample:
            try:
                with open(fpath, "rb") as f:
                    r = requests.post(API_URL, files={"file": f},
                                      timeout=TIMEOUT_XAI)
                d = r.json()
                xai = d.get("xai_stats")
                if xai:
                    if xai.get("agreement_pass"): agree  += 1
                    if xai.get("energy_pass"):    energy += 1
                else:
                    print(f"  No xai_stats for {os.path.basename(fpath)}")
            except Exception as e:
                errors += 1
                print(f"  ERROR {os.path.basename(fpath)}: {e}")

        xai_results[class_name] = {
            "sampled": len(sample), "agree": agree, "energy": energy,
            "eligible_total": len(eligible_paths),
        }
        n = len(sample)
        print(f"  agree={agree}/{n} ({agree/n*100:.1f}%)  "
              f"energy={energy}/{n} ({energy/n*100:.1f}%)  errors={errors}")

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Class':<25} {'Eligible':>9} {'Sampled':>8} {'Agree%':>8} {'Energy%':>8}")
    print("-" * 80)
    oa = oe = on = 0
    for class_name in CLASSES:
        r_   = pass1[class_name]
        xr   = xai_results.get(class_name, {})
        elig = len(r_["eligible_paths"])
        n    = xr.get("sampled", 0)
        ac   = xr.get("agree", 0)
        ec   = xr.get("energy", 0)
        a_str = f"{ac/n*100:>7.1f}%" if n > 0 else "     N/A"
        e_str = f"{ec/n*100:>7.1f}%" if n > 0 else "     N/A"
        print(f"{class_name:<25} {elig:>9} {n:>8} {a_str} {e_str}")
        oa += ac; oe += ec; on += n
    print("-" * 80)
    if on:
        print(f"{'Overall (sampled)':<25} {overall_elig:>9} {on:>8} "
              f"{oa/on*100:>7.1f}% {oe/on*100:>7.1f}%")
    print(f"\nNote: agreement% and energy% computed over {SAMPLE_PER_CLASS}-image")
    print(f"random sample per class; eligible totals are from the full dataset.")

# Save
out = {}
for c in CLASSES:
    r_ = dict(pass1[c])
    r_["eligible_paths"] = len(r_["eligible_paths"])   # don't dump full paths
    out[c] = r_
with open("xai_stats_results_final.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to xai_stats_results_final.json")
