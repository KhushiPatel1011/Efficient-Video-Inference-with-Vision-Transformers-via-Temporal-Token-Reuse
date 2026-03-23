import gradio as gr
import sys
from pathlib import Path

# importable repo roots
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def placeholder(video):
    return "Loading UI..."

with gr.Blocks(title="Efficient Video Inference with Vision Transformers via Temporal Token Reuse") as demo:

    gr.Markdown(
        """
        # Efficient Video Inference via Temporal Token Reuse
        **Khevna Vadaliya & Darshil Prajapati | AI Capstone, Spring 2026 | Saint Louis University**
        
        This demo compares standard ViT inference against our 
        Temporal Token Reuse approach.
        Upload a video and see the speedup and accuracy results in real time.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            max_frames = gr.Slider(
                minimum=10,
                maximum=120,
                value=60,
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
                lines=10,
                placeholder="Results will appear here after running inference..."
            )

    run_button.click(
        fn=placeholder,
        inputs=[video_input],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch()