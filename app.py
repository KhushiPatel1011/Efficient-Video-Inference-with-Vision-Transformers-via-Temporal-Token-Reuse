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

# Loading both the models once at the beginning
print("Loading baseline model...")
baseline_model, transform, _ = load_timm_vit(
    model_name="vit_base_patch16_224", pretrained=True
)
baseline_model.eval()

print("Loading reuse model...")
reuse_model, _, _ = load_timm_vit(
    model_name="vit_base_patch16_224", pretrained=True
)
reuse_model.eval()
apply_patch(reuse_model, stable_ratio=0.75)
print("Models ready.")

DEVICE = torch.device("cpu")
baseline_model.to(DEVICE)
reuse_model.to(DEVICE)


# helpers function
def extract_frames(video_path: str, max_frames: int):
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


# Main function
def run_inference(video, max_frames, stable_ratio):
    if video is None:
        return "Please upload a video."

    try:
        max_frames = int(max_frames)
        stable_ratio = float(stable_ratio)

        # Updating stable ratio on reuse model blocks
        from src.models.patching import TBKVBlock
        for module in reuse_model.modules():
            if isinstance(module, TBKVBlock):
                module._tbkv_stable_ratio = stable_ratio

        frames = extract_frames(video, max_frames=max_frames)
        if len(frames) == 0:
            return "No frames could be extracted from the video."

        # Baseline inference
        baseline_latencies = []
        baseline_preds = []

        for img in frames:
            x = transform(img).unsqueeze(0).to(DEVICE)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = baseline_model(x)
            t1 = time.perf_counter()
            baseline_latencies.append((t1 - t0) * 1000)
            baseline_preds.append(top1_pred(logits))

        avg_base = sum(baseline_latencies) / len(baseline_latencies)
        fps_base = 1000.0 / avg_base

        # KV caching 
        reset_cache(reuse_model)
        reuse_latencies = []
        reuse_preds = []
        match_count = 0

        for i, img in enumerate(frames):
            x = transform(img).unsqueeze(0).to(DEVICE)

            if i == 0:
                set_caching_mode(reuse_model, caching=True)
            else:
                set_caching_mode(reuse_model, caching=False)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = reuse_model(x)
            t1 = time.perf_counter()
            reuse_latencies.append((t1 - t0) * 1000)
            reuse_preds.append(top1_pred(logits))

            if i > 0:
                if reuse_preds[i][0] == baseline_preds[i][0]:
                    match_count += 1

        avg_reuse = sum(reuse_latencies) / len(reuse_latencies)
        fps_reuse = 1000.0 / avg_reuse
        speedup = avg_base / avg_reuse
        n_reuse = len(frames) - 1
        match_rate = match_count / n_reuse if n_reuse > 0 else 0.0

        # ---- Format results ----
        result  = "=== BASELINE vs KV CACHE REUSE ===\n\n"
        result += f"Frames processed:     {len(frames)}\n"
        result += f"Stable ratio:         {stable_ratio}\n\n"
        result += f"--- Baseline ---\n"
        result += f"Avg latency:          {avg_base:.2f} ms/frame\n"
        result += f"Throughput:           {fps_base:.2f} FPS\n\n"
        result += f"--- KV Cache Reuse ---\n"
        result += f"Avg latency:          {avg_reuse:.2f} ms/frame\n"
        result += f"Throughput:           {fps_reuse:.2f} FPS\n\n"
        result += f"--- Comparison ---\n"
        result += f"Speedup:              {speedup:.3f}x\n"
        result += f"Top-1 Match Rate:     {match_rate:.3f}\n\n"
        result += f"--- Per-frame predictions (first 10) ---\n"
        for i in range(min(10, len(frames))):
            if i == 0:
                match_str = "N/A (cache frame)"
            else:
                match_str = "YES" if reuse_preds[i][0] == baseline_preds[i][0] else "NO"
            result += (
                f"  Frame {i:03d}: "
                f"baseline={baseline_preds[i][0]} ({baseline_preds[i][1]:.3f})  "
                f"reuse={reuse_preds[i][0]} ({reuse_preds[i][1]:.3f})  "
                f"match={match_str}\n"
            )

        return result

    except Exception as e:
        import traceback
        return f"Error:\n{traceback.format_exc()}"


# Layout
with gr.Blocks(title="Efficient Video Inference with Vision Transformers via Temporal Token Reuse") as demo:

    gr.Markdown(
        """
        # Efficient Video Inference with ViT via Temporal Token Reuse
        **Khevna Vadaliya & Darshil Prajapati | AI Capstone, Spring 2026 | Saint Louis University**

        Upload a video to compare standard ViT baseline inference against our
        TBKV-style Key-Value cache temporal token reuse approach.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            max_frames = gr.Slider(
                minimum=10, maximum=120, value=30, step=10,
                label="Max Frames to Process"
            )
            stable_ratio = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.75, step=0.05,
                label="Stable Ratio (fraction of tokens reusing cached K/V)"
            )
            run_button = gr.Button("Run Inference", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Results",
                lines=28,
                placeholder="Results will appear here after running inference..."
            )

    run_button.click(
        fn=run_inference,
        inputs=[video_input, max_frames, stable_ratio],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch()