import torch
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry

device = "cuda"
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                      torch_dtype=torch.float16,
                                                      )
pipe = pipe.to(device)
