import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit
from src.methods.temporal_change import patch_change_scores, change_mask_from_scores


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _save_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # stable column order
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder of sequential frames")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to use (including first)")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size (ViT-B/16 uses 16)")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Top fraction patches marked as changed")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup frames (baseline inference only)")
    parser.add_argument("--out-csv", type=str, default="results/temporal_simulation.csv", help="Output CSV path")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load model + preprocess (we are using preprocess for both inference + patch scoring)
    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    frames_dir = Path(args.frames)
    rows: List[Dict] = []

    print("\nTEMPORAL TOKEN REUSE")
    print("Note: no model internals modified yet")
    print("We estimate compute savings using temporal patch stability.\n")

    print("t, changed_patches, total_patches, stable_ratio, baseline_ms, simulated_ms, threshold")

    # Warmup: run a few forward passes to reduce one-time overhead noise
    # Still CPU, still honest — just avoids first-run spikes
    if args.warmup > 0:
        it = iter_frame_pairs(frames_dir, max_frames=min(args.max_frames, args.warmup + 1))
        for _, _, curr_img in it:
            x = transform(curr_img).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(x)

    total_patches_known = None
    baseline_times: List[float] = []
    simulated_times: List[float] = []
    stable_ratios: List[float] = []

    # Main loop: for each pair (t-1, t), compute:
    # - baseline inference time for frame t
    # - patch change mask between (t-1, t)
    # - simulated latency assuming we only "recompute" changed patches (proxy)
    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        prev = transform(prev_img).to(device)      # [3,H,W]
        curr = transform(curr_img).to(device)      # [3,H,W]
        x = curr.unsqueeze(0)                      # [1,3,H,W]

        # Baseline inference timing (real)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()
        baseline_ms = _ms(t0, t1)

        # Patch-level temporal change scoring (ViT-consistent)
        scores = patch_change_scores(prev, curr, patch_size=args.patch_size)
        changed_mask, thresh = change_mask_from_scores(scores, keep_ratio=args.keep_ratio)
        total_patches = int(scores.numel())
        changed = int(changed_mask.sum().item())
        stable = total_patches - changed
        stable_ratio = stable / float(total_patches)

        if total_patches_known is None:
            total_patches_known = total_patches

        # Simulation: estimate time if we only recompute changed patches
        # Assumption: ViT compute roughly scales with number of spatial tokens processed.
        # Simulated ratio uses changed/total (CLS excluded since constant).
        compute_ratio = changed / float(total_patches)
        simulated_ms = baseline_ms * compute_ratio

        baseline_times.append(baseline_ms)
        simulated_times.append(simulated_ms)
        stable_ratios.append(stable_ratio)

        row = {
            "t": t,
            "changed_patches": changed,
            "total_patches": total_patches,
            "stable_ratio": round(stable_ratio, 6),
            "baseline_ms": round(baseline_ms, 3),
            "simulated_ms": round(simulated_ms, 3),
            "threshold": round(float(thresh), 6),
            "keep_ratio": args.keep_ratio,
            "model": args.model,
        }
        rows.append(row)

        print(
            f"{t}, {changed}, {total_patches}, {stable_ratio:.3f}, "
            f"{baseline_ms:.2f}, {simulated_ms:.2f}, {float(thresh):.6f}"
        )

    # Summary
    if rows:
        avg_base = sum(baseline_times) / len(baseline_times)
        avg_sim = sum(simulated_times) / len(simulated_times)
        avg_stable = sum(stable_ratios) / len(stable_ratios)

        # ms -> fps
        base_fps = 1000.0 / avg_base if avg_base > 0 else 0.0
        sim_fps = 1000.0 / avg_sim if avg_sim > 0 else 0.0

        # "Estimated speedup" is simulated_fps / baseline_fps 
        est_speedup = (avg_base / avg_sim) if avg_sim > 0 else 0.0

        print("\n    SUMMARY   ")
        print(f"frames_used (pairs): {len(rows)}")
        print(f"avg_baseline_ms: {avg_base:.2f}")
        print(f"avg_baseline_fps: {base_fps:.2f}")
        print(f"avg_stable_ratio: {avg_stable:.3f}  (fraction stable patches)")
        print(f"avg_simulated_ms: {avg_sim:.2f}")
        print(f"avg_simulated_fps: {sim_fps:.2f}")
        print(f"estimated_speedup: {est_speedup:.2f}x")
        print(
            "note: simulated_ms assumes compute scales with changed patch fraction; "
            "no ViT internals modified yet."
        )

    # Save CSV
    out_csv = Path(args.out_csv)
    _save_csv(rows, out_csv)
    print(f"\nsaved_csv: {out_csv.as_posix()}\nDone.")


if __name__ == "__main__":
    main()