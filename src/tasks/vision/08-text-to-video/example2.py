import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("mps")

prompt = """Vast golden desert under a bright midday sun, two tiny figures riding 
                  camels in the far distance moving very slowly, shimmering heat waves, soft sand dunes, 
                  pale blue sky, serene and epic atmosphere"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=240,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "exo.mp4", fps=24)
