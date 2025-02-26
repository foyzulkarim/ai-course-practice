import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "mps" # Options: ["cpu", "cuda", "mps"]
dtype = torch.float16

step = 2  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

# base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

# ToonYou - worked [step=1,2]
base = "frankjoshua/toonyou_beta6"
fileName = "e1_toonyou_2step.gif"

# SG161222/Realistic_Vision_V5.1_noVAE
# base = "SG161222/Realistic_Vision_V5.1_noVAE"
# fileName = "e1_realistic_1step.gif"

# Yntec/ResidentCNZCartoon3D
# base = "Yntec/ResidentCNZCartoon3D"
# fileName = "e1_animatediff_1step.gif"

# guoyww/animatediff-motion-adapter-v1-5-2
# base = "guoyww/animatediff-motion-adapter-v1-5-2"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

prompts = """
Generate an anime-style video of a dragon flying over a vibrant jungle. Style references: Hayao Miyazaki. Output resolution: 1080p, 60fps.
"""

output = pipe(prompt=prompts, guidance_scale=0.8, num_inference_steps=step)
export_to_gif(output.frames[0], fileName)
