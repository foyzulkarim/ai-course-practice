import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif, export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "mps" # Options: ["cpu", "cuda", "mps"]
dtype = torch.float16

step = 1  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

# epiCRealism - worked [step=1,2,4]
base = "emilianJR/epiCRealism" 
fileName = "e1_animatediff_4step.gif"

# ToonYou - worked [step=1,2]
# base = "frankjoshua/toonyou_beta6"
# fileName = "e1_toonyou_2step.gif"

# SG161222/Realistic_Vision_V5.1_noVAE
# base = "SG161222/Realistic_Vision_V5.1_noVAE"
# fileName = "e1_realistic_1step.gif"

# Yntec/ResidentCNZCartoon3D
# base = "Yntec/ResidentCNZCartoon3D"
# fileName = "e1_animatediff_1step.gif"


adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

prompts = """
A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes.Sunlight filters through the tall bamboo, casting a gentle glow on the scene.
"""

output = pipe(prompt=prompts, guidance_scale=0.8, num_inference_steps=step)
export_to_gif(output.frames[0], fileName)
# export_to_video(output.frames[0], "output.mp4")
