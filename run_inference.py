import os
import argparse

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler
import torch
import gradio as gr

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--model_dir", type=str, default='./dreambooth_model')
    parser.add_argument("--prompt", type=str, default='a zwx cat in mad max fury road')
    parser.add_argument("--steps", type=int, default=25)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

args = parse_args()


pipe = StableDiffusionPipeline.from_pretrained(
        args.model_dir,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_dir, subfolder="scheduler"),
        torch_dtype=torch.float32,
    ).to("cuda")

#@title Run the Stable Diffusion pipeline with interactive UI Demo on Gradio
#@markdown Run this cell to get an interactive demo where you can run the model using Gradio

#@markdown ![](https://i.imgur.com/2ACLWu2.png)

def inference(prompt, num_samples):
    all_images = [] 
    images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=args.steps).images
    all_images.extend(images)
    return all_images

results_path = './results'
images = inference(args.prompt,2)
if not os.path.exists(results_path):
  os.mkdir(results_path)
[image.save(f"{results_path}/{i}.jpeg") for i, image in enumerate(images)]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="prompt")
            samples = gr.Slider(label="Samples",value=1)
            run = gr.Button(value="Run")
        with gr.Column():
            gallery = gr.Gallery(show_label=False)

    run.click(inference, inputs=[prompt,samples], outputs=gallery)
    gr.Examples([["a photo of sks toy riding a bicycle", 1,1]], [prompt,samples], gallery, inference, cache_examples=False)


demo.launch(share=True)