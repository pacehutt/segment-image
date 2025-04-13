from app.routers.router import *
from app.views.view import *
from app.interactors.interactor import *

with demo:
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(inpaint, inputs=[
        input_img, mask_img, prompt_text], outputs=[output_img])
