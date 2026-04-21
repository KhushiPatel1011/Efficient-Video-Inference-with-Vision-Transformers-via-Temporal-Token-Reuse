"""
Sweep cls_threshold values for block-skip reuse.
Shows accuracy vs speedup tradeoff at different stability thresholds.
"""

import subprocess
import sys
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
frames_dir = "data/raw/medical_frames"
max_frames = 60
cache_after_block = 7
results = []

print(f"\n{'threshold':>10} | {'skip_rate':>9} | {'speedup':>8} | {'top1':>6} | {'top3':>6} | {'top5':>6} | {'avg_kl':>10} | {'avg_cd':>8}")
print("-" * 85)

for t in thresholds:
    out_csv = f"results/block_skip_thresh_{str(t).replace('.','_')}.csv"

    subprocess.run([
        sys.executable,
        "scripts/run_block_skip_reuse.py",
        "--frames", frames_dir,
        "--max-frames", str(max_frames),
        "--cache-after-block", str(cache_after_block),
        "--cls-threshold", str(t),
        "--out-csv", out_csv,
    ], capture_output=True)

    rows = []
    try:
        with open(out_csv, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        print(f"{t:>10.2f} | ERROR reading CSV")
        continue

    skip_rows = [r for r in rows if r["decision"] == "skip"]
    full_rows = [r for r in rows if r["decision"] == "full"]
    n_skip = len(skip_rows)
    n_total = len(rows)
    skip_rate = n_skip / (n_total - 1) if n_total > 1 else 0.0

    if n_skip == 0:
        print(f"{t:>10.2f} | {skip_rate:>9.3f} | {'no skips':>8}")
        continue

    avg_base = sum(float(r["baseline_ms"]) for r in skip_rows) / n_skip
    avg_skip = sum(float(r["skip_ms"]) for r in skip_rows) / n_skip
    speedup = avg_base / avg_skip if avg_skip > 0 else 0.0

    t1 = sum(int(r["top1_match"]) for r in skip_rows) / n_skip
    t3 = sum(int(r["top3_match"]) for r in skip_rows) / n_skip
    t5 = sum(int(r["top5_match"]) for r in skip_rows) / n_skip
    avg_kl = sum(float(r["kl_divergence"]) for r in skip_rows) / n_skip
    avg_cd = sum(float(r["conf_delta"]) for r in skip_rows) / n_skip

    print(
        f"{t:>10.2f} | {skip_rate:>9.3f} | {speedup:>8.3f} | "
        f"{t1:>6.3f} | {t3:>6.3f} | {t5:>6.3f} | "
        f"{avg_kl:>10.6f} | {avg_cd:>8.6f}"
    )

    results.append({
        "cls_threshold": t,
        "skip_rate": round(skip_rate, 4),
        "speedup": round(speedup, 4),
        "top1_match_rate": round(t1, 4),
        "top3_match_rate": round(t3, 4),
        "top5_match_rate": round(t5, 4),
        "avg_kl_divergence": round(avg_kl, 6),
        "avg_conf_delta": round(avg_cd, 6),
    })

# Save summary
out_summary = Path("results/block_skip_threshold_sweep.csv")
out_summary.parent.mkdir(parents=True, exist_ok=True)
if results:
    with open(out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {out_summary}")

print("Done.")