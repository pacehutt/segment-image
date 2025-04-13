import gradio as gr
import numpy as np
from PIL import Image
from app.views.view import selected_pixels
from app.entities.entity import predictor, pipe
import numpy as np

def generate_mask(image, evt: gr.SelectData):
        selected_pixels.append(evt.index)
        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])

        mask, _, _ = predictor.predict(
            point_coords=input_points, point_labels=input_labels, multimask_output=False)
        mask = Image.fromarray(mask[0, :, :])
        return mask

def inpaint(image, mask, prompt):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    # image = image.resize((512, 512))
    # mask = mask.resize((512, 512))
    output = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return output