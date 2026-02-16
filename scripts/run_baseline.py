import argparse
from pathlib import Path

import torch

from src.models.timm_vit import load_timm_vit
from src.data.frames_dataset import load_frames_from_folder
from src.utils.timer import Timer
from src.evaluation.predictions import topk_from_logits
from src.utils.io import save_rows_to_csv
from src.evaluation.report import summarize_run
from src.utils.hooks import register_vit_block_hooks



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True, help="Folder containing frame images (jpg/png/...)")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--max-frames", type=int, default=30, help="Limit number of frames")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to compute")
    parser.add_argument("--out", type=str, default="results/baseline_run.csv", help="CSV output path")
    parser.add_argument("--use-hooks", type=int, default=1, help="1=log token shapes per block, 0=off")

    args = parser.parse_args()

    device = torch.device("cpu")

    # Loading model and preprocessing
    model, transform, class_names = load_timm_vit(model_name=args.model, pretrained=True)
    
    hook_state = None
    if args.use_hooks:
        hook_state = register_vit_block_hooks(model)


    # Loading frames
    frames = load_frames_from_folder(Path(args.frames), max_frames=args.max_frames)

    rows = []
    timer = Timer()

    for i, img in enumerate(frames):
        x = transform(img).unsqueeze(0).to(device)

        timer.start()
        with torch.no_grad():
            logits = model(x)
        dt_ms = timer.stop_ms()

        topk = topk_from_logits(logits, k=args.topk, class_names=class_names)

        top1 = topk[0]
        print(f"[{i:03d}] {dt_ms:7.2f} ms | top1={top1['label']} ({top1['prob']:.3f})")

        rows.append({
            "frame_idx": i,
            "latency_ms": dt_ms,
            "top1_label": top1["label"],
            "top1_prob": top1["prob"],
        })

    # Saving CSV
    save_rows_to_csv(args.out, rows)

    # Summary
    summary = summarize_run(rows)
    print("\n   SUMMARY    ")
    print(f"frames: {int(summary['num_frames'])}")
    print(f"avg_latency_ms: {summary['avg_latency_ms']:.2f}")
    print(f"fps: {summary['fps']:.2f}")
    print(f"saved_csv: {args.out}")
    
    if hook_state is not None:
        print("\n    TOKEN SHAPES (per block)    ")
        hook_state.pretty_print()



if __name__ == "__main__":
    main()
