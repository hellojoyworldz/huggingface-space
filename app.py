import os
import tempfile
import gradio as gr
import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
from constants import KIND

repo = "openai/shap-e"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

pipeline = ShapEPipeline.from_pretrained(repo).to(device)

def on_generate(kind):
    if not kind:
        return None, "동물을 선택해 주세요."

    prompt =  f"a {KIND[kind]}"

    steps = 64 if device != "cpu" else 32

    images = pipeline(
        prompt,
        guidance_scale=15.0,
        num_inference_steps=steps,
        frame_size=256,
    ).images

    path = os.path.join(tempfile.mkdtemp(), "result.gif")
    export_to_gif(images[0], path)

    status = (
        f"**종류** {kind}\n\n"
        f"**명령어 (영어):**\n```\n{prompt}\n```"
    )
    return path, status


with gr.Blocks(title="3D 동물") as demo:
    gr.Markdown("## 🐾 3D 동물")
    gr.Markdown("종류를 고르면 3D GIF를 만들어줘요.")

    kind = gr.Radio(
        choices=list(KIND.keys()),
        value="",
        label="동물을 골라라",
        info="누구로 만들까?",
    )

    make_btn = gr.Button("3D 만들기", variant="primary", size="lg")

    with gr.Row():
        status_out = gr.Markdown()
        result_out = gr.Image(label="3D MODEL")

    make_btn.click(on_generate, [kind], [result_out, status_out])


demo.launch()
