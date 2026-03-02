import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/mask_comparison.csv")
    parser.add_argument("--out", type=str, default="results/mask_comparison_plot.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic sanity
    if "t" not in df.columns or "iou" not in df.columns or "agreement" not in df.columns:
        raise ValueError("CSV missing required columns: t, iou, agreement")

    # IoU Plotting
    plt.figure()
    plt.plot(df["t"], df["iou"])
    plt.xlabel("Frame pair index (t)")
    plt.ylabel("IoU (changed-token overlap)")
    plt.title("Pixel vs Embedding Mask: IoU over Time")
    plt.savefig(out_path.as_posix(), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot Agreement 
    out2 = out_path.with_name(out_path.stem + "_agreement.png")
    plt.figure()
    plt.plot(df["t"], df["agreement"])
    plt.xlabel("Frame pair index (t)")
    plt.ylabel("Agreement (all tokens)")
    plt.title("Pixel vs Embedding Mask: Agreement over Time")
    plt.savefig(out2.as_posix(), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plots:\n- {out_path.as_posix()}\n- {out2.as_posix()}")


if __name__ == "__main__":
    main()
