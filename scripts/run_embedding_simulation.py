import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.data.frame_pairs import iter_frame_pairs
from src.models.timm_vit import load_timm_vit
from src.utils.token_extract import extract_patch_tokens_pre_blocks
from src.methods.embedding_change import compute_embedding_change


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _save_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Top fraction tokens marked as changed")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup frames (baseline inference only)")
    parser.add_argument("--out-csv", type=str, default="results/embedding_simulation.csv", help="Output CSV path")
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device("cpu")

    model, transform, _ = load_timm_vit(model_name=args.model, pretrained=True)
    model.to(device)
    model.eval()

    frames_dir = Path(args.frames)
    rows: List[Dict] = []

    print("\n   TEMPORAL TOKEN REUSE (EMBEDDING SIMULATION)    ")
    print("Mask source: token embedding similarity (pre-block patch tokens).")
    print("Note: no ViT internals modified; simulated_ms uses changed-token fraction.\n")
    print("t, changed_tokens, total_tokens, stable_ratio, baseline_ms, simulated_ms, threshold")

    # Warmup to reduce first-run overhead noise
    if args.warmup > 0:
        it = iter_frame_pairs(frames_dir, max_frames=min(args.max_frames, args.warmup + 1))
        for _, _, curr_img in it:
            x = transform(curr_img).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(x)

    baseline_times: List[float] = []
    simulated_times: List[float] = []
    stable_ratios: List[float] = []

    for t, prev_img, curr_img in iter_frame_pairs(frames_dir, max_frames=args.max_frames):
        # Preparing inputs
        prev_x = transform(prev_img).unsqueeze(0).to(device)
        curr_x = transform(curr_img).unsqueeze(0).to(device)

        # Baseline inference timing (real)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(curr_x)
        t1 = time.perf_counter()
        baseline_ms = _ms(t0, t1)

        # Extracting pre-block patch tokens (exclude CLS), shape (196, C)
        with torch.no_grad():
            prev_tokens, _ = extract_patch_tokens_pre_blocks(model, prev_x, return_batch=False)
            curr_tokens, _ = extract_patch_tokens_pre_blocks(model, curr_x, return_batch=False)

        # Embedding-level change detection + top-k changed tokens
        change = compute_embedding_change(prev_tokens, curr_tokens, keep_ratio=args.keep_ratio)

        total = change.total
        changed = change.changed
        stable_ratio = change.stable_ratio

        # Simulation: assuming compute scales with changed-token fraction
        compute_ratio = changed / float(total)
        simulated_ms = baseline_ms * compute_ratio

        baseline_times.append(baseline_ms)
        simulated_times.append(simulated_ms)
        stable_ratios.append(stable_ratio)

        row = {
            "t": t,
            "changed_tokens": changed,
            "total_tokens": total,
            "stable_ratio": round(stable_ratio, 6),
            "baseline_ms": round(baseline_ms, 3),
            "simulated_ms": round(simulated_ms, 3),
            "threshold": round(float(change.threshold), 6),
            "keep_ratio": args.keep_ratio,
            "model": args.model,
        }
        rows.append(row)

        print(
            f"{t}, {changed}, {total}, {stable_ratio:.3f}, "
            f"{baseline_ms:.2f}, {simulated_ms:.2f}, {float(change.threshold):.6f}"
        )

    # Summary
    if rows:
        avg_base = sum(baseline_times) / len(baseline_times)
        avg_sim = sum(simulated_times) / len(simulated_times)
        avg_stable = sum(stable_ratios) / len(stable_ratios)

        base_fps = 1000.0 / avg_base if avg_base > 0 else 0.0
        sim_fps = 1000.0 / avg_sim if avg_sim > 0 else 0.0
        est_speedup = (avg_base / avg_sim) if avg_sim > 0 else 0.0

        print("\n    SUMMARY (Embedding Simulation)")
        print(f"frames_used (pairs): {len(rows)}")
        print(f"avg_baseline_ms: {avg_base:.2f}")
        print(f"avg_baseline_fps: {base_fps:.2f}")
        print(f"avg_stable_ratio: {avg_stable:.3f}  (fraction stable tokens)")
        print(f"avg_simulated_ms: {avg_sim:.2f}")
        print(f"avg_simulated_fps: {sim_fps:.2f}")
        print(f"estimated_speedup: {est_speedup:.2f}x")
        print(
            "note: simulated_ms assumes compute scales with changed-token fraction; "
            "no ViT internals modified yet."
        )

    out_csv = Path(args.out_csv)
    _save_csv(rows, out_csv)
    print(f"\nsaved_csv: {out_csv.as_posix()}\nDone.")


if __name__ == "__main__":
    main()
