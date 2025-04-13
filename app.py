import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO
from PIL import Image
import gradio as gr

model_type = "vit_h"
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"
yolo_path = "yolov8n.pt"
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                      torch_dtype=torch.float16,
                                                      )


def show_anns(anns):
    """Show annotations on the image."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return img


def generate_mask(image):
    """Generate mask for the input image."""
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks


def segmented_images(image, masks):
    """Return the segmented images from the masks to a numpy array."""
    segmented_images = []
    for i, mask in enumerate(masks):
        masked_image = image.copy()
        masked_image[~mask['segmentation']] = 0
        segmented_images.append(masked_image)
    return segmented_images


def detect_objects(image_path):
    """Detect objects in the input image."""
    detector = YOLO(yolo_path)
    results = detector(image_path)
    return results


def inpaint(image, mask, prompt):
    """Inpaint the input image."""
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    # image = image.resize((512, 512))
    # mask = mask.resize((512, 512))

    output = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return output


def main():
    """Main function."""
    with gr.Blocks() as demo:
        with gr.Row():
            input_img = gr.Image(label="Input Image")
            masked_img = gr.Image(label="Segmented Image")
            output_img = gr.Image(label="Output Image")
        with gr.Row():
            prompt = gr.Textbox(lines=1, label="Prompt")
        with gr.Row():
            submit = gr.Button("Run")


if __name__ == "__main__":
    main()
