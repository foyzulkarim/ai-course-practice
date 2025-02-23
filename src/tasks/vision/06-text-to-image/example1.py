import torch
from enum import Enum
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

class ModelName(Enum):
    STABLE_DIFFUSION_2 = "stabilityai/stable-diffusion-2"

model_id = ModelName.STABLE_DIFFUSION_2.value
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)

device = "cpu"
# Ensure CUDA is available and PyTorch is compiled with CUDA support before moving to GPU
if torch.version.cuda is not None and torch.cuda.is_available():
    device = "cuda"
    pipe = pipe.to(device)
elif torch.backends.mps.is_available():
    device = "mps"
    pipe = pipe.to(device)

prompt = """Vast golden desert under a bright midday sun, two tiny figures riding camels in the far distance, shimmering heat waves, soft sand dunes, pale blue sky, serene and epic atmosphere."""
image = pipe(prompt).images[0]
image.save("desert1.png")