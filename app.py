# imports

import gradio as gr
import sys
import time
import torch
import cv2
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.timm_vit import load_timm_vit
from src.models.patching import apply_patch, set_caching_mode, reset_cache

# Model loading
print("Loading model...")
model, transform, _ = load_timm_vit(model_name="vit_base_patch16_224", pretrained=True)
model.eval()
print("Model loaded.")

def extract_frames(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok or count >= max_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        count += 1
    cap.release()
    return frames

def top1_pred(logits):
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return int(idx.item()), float(conf.item())

def run_inference(video, max_frames, stable_ratio):
    if video is None:
        return "Please upload a video."

    try:
        device = torch.device("cpu")
        max_frames = int(max_frames)
        stable_ratio = float(stable_ratio)

        frames = extract_frames(video, max_frames=max_frames)
        if len(frames) == 0:
            return "No frames could be extracted from the video."

        # Baseline inference
        model_baseline, transform_b, _ = load_timm_vit(
            model_name="vit_base_patch16_224", pretrained=True
        )
        model_baseline.eval().to(device)

        baseline_latencies = []
        baseline_preds = []

        for img in frames:
            x = transform_b(img).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model_baseline(x)
            t1 = time.perf_counter()
            baseline_latencies.append((t1 - t0) * 1000)
            baseline_preds.append(top1_pred(logits))

        avg_base = sum(baseline_latencies) / len(baseline_latencies)
        fps_base = 1000.0 / avg_base

        # KV Caching and reusing
        model_reuse, transform_r, _ = load_timm_vit(
            model_name="vit_base_patch16_224", pretrained=True
        )
        model_reuse.eval().to(device)
        apply_patch(model_reuse, stable_ratio=stable_ratio)
        reset_cache(model_reuse)

        reuse_latencies = []
        reuse_preds = []
        match_count = 0

        for i, img in enumerate(frames):
            x = transform_r(img).unsqueeze(0).to(device)
            if i == 0:
                set_caching_mode(model_reuse, caching=True)
            else:
                set_caching_mode(model_reuse, caching=False)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model_reuse(x)
            t1 = time.perf_counter()
            reuse_latencies.append((t1 - t0) * 1000)
            reuse_preds.append(top1_pred(logits))

            if i > 0:
                if reuse_preds[i][0] == baseline_preds[i][0]:
                    match_count += 1

        avg_reuse = sum(reuse_latencies) / len(reuse_latencies)
        fps_reuse = 1000.0 / avg_reuse
        speedup = avg_base / avg_reuse
        match_rate = match_count / (len(frames) - 1)

        # results
        result = "=== BASELINE vs KV CACHE REUSE ===\n\n"
        result += f"Frames processed:       {len(frames)}\n"
        result += f"Stable ratio:           {stable_ratio}\n\n"
        result += f"--- Baseline ---\n"
        result += f"Avg latency:            {avg_base:.2f} ms/frame\n"
        result += f"Throughput:             {fps_base:.2f} FPS\n\n"
        result += f"--- KV Cache Reuse ---\n"
        result += f"Avg latency:            {avg_reuse:.2f} ms/frame\n"
        result += f"Throughput:             {fps_reuse:.2f} FPS\n\n"
        result += f"--- Comparison ---\n"
        result += f"Speedup:                {speedup:.3f}x\n"
        result += f"Top-1 Match Rate:       {match_rate:.3f}\n\n"
        result += f"--- Per-frame predictions (first 5) ---\n"
        for i in range(min(5, len(frames))):
            match = "YES" if i > 0 and reuse_preds[i][0] == baseline_preds[i][0] else ("N/A" if i == 0 else "NO")
            result += f"  Frame {i:03d}: baseline={baseline_preds[i][0]} reuse={reuse_preds[i][0]} match={match}\n"

        return result

    except Exception as e:
        import traceback
        return f"Error:\n{traceback.format_exc()}"

with gr.Blocks(title="Efficient Video Inference with Vision Transformers via Temporal Token Reuse") as demo:

    gr.Markdown(
        """
        # Efficient Video Inference with ViT via Temporal Token Reuse
        **Khevna Vadaliya & Darshil Prajapati | AI Capstone, Spring 2026 | Saint Louis University**

        Upload a video to compare baseline ViT inference against our
        Temporal Token Reuse approach.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            max_frames = gr.Slider(
                minimum=10,
                maximum=120,
                value=30,
                step=10,
                label="Max Frames to Process"
            )
            stable_ratio = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.75,
                step=0.05,
                label="Stable Ratio (fraction of tokens reusing cached K/V)"
            )
            run_button = gr.Button("Run Inference", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Results",
                lines=25,
                placeholder="Results will appear here after running inference..."
            )

    run_button.click(
        fn=run_inference,
        inputs=[video_input, max_frames, stable_ratio],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch()