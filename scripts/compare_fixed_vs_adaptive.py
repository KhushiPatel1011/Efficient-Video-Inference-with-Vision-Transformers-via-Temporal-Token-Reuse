from pathlib import Path

import pandas as pd


def main():
    fixed_csv = Path("results/kv_motion_comparison.csv")
    adaptive_csv = Path("results/kv_motion_adaptive.csv")
    out_csv = Path("results/kv_motion_fixed_vs_adaptive.csv")

    if not fixed_csv.exists():
        raise FileNotFoundError(f"Missing fixed CSV: {fixed_csv}")
    if not adaptive_csv.exists():
        raise FileNotFoundError(f"Missing adaptive CSV: {adaptive_csv}")

    fixed_df = pd.read_csv(fixed_csv).copy()
    adaptive_df = pd.read_csv(adaptive_csv).copy()

    fixed_df["policy"] = "fixed"
    if "policy" not in adaptive_df.columns:
        adaptive_df["policy"] = "adaptive"

    common_cols = [
        "video_id",
        "motion_bucket",
        "policy",
        "r_match",
        "frames_used",
        "reuse_frames",
        "avg_baseline_ms",
        "avg_reuse_ms",
        "avg_speedup",
        "top1_match_rate",
    ]

    missing_fixed = set(common_cols) - set(fixed_df.columns)
    missing_adaptive = set(common_cols) - set(adaptive_df.columns)

    if missing_fixed:
        raise ValueError(f"Fixed CSV missing columns: {missing_fixed}")
    if missing_adaptive:
        raise ValueError(f"Adaptive CSV missing columns: {missing_adaptive}")

    merged = pd.concat(
        [fixed_df[common_cols], adaptive_df[common_cols]],
        ignore_index=True
    )

    merged.to_csv(out_csv, index=False)

    print("\nSaved comparison CSV:")
    print(out_csv)
    print("Done.")


if __name__ == "__main__":
    main()