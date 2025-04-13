# views.py

import gradio as gr

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input Image")
        mask_img = gr.Image(label="Segmented Image")
        output_img = gr.Image(label="Output Image")
    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
