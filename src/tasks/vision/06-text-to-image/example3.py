import time
import torch
from enum import Enum
from diffusers import DiffusionPipeline

# Start time measurement
start_time = time.time()

class ModelName(Enum):
    IONET_OFFICIAL_BC8_ALPHA = "ionet-official/bc8-alpha"
    CAGLIOSTROLAB_ANIMAGINE_XL_3_0 = "cagliostrolab/animagine-xl-3.0"   

model_id = ModelName.CAGLIOSTROLAB_ANIMAGINE_XL_3_0.value

# For the Black Forest model, use DiffusionPipeline instead of StableDiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        safety_checker=None,
        feature_extractor=None
    )

# Select device
device = "cpu"
if torch.version.cuda is not None and torch.cuda.is_available():
    device = "cuda"
    pipe = pipe.to(device)
elif torch.backends.mps.is_available():
    device = "mps"
    pipe = pipe.to(device)

prompt = """Vast golden desert under a bright midday sun, two tiny figures riding camels in the far distance, shimmering heat waves, soft sand dunes, pale blue sky, serene and epic atmosphere."""
output = pipe(prompt)
image = output.images[0]
image.save("desert3.png")
# End time measurement and print duration
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script executed in {elapsed_time:.2f} seconds")