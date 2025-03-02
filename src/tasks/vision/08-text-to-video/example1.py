import os
import sys
import torch
from enum import Enum
from dataclasses import dataclass
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Import utility functions from shared vision utils
try:
    # Try relative import first (when run from parent directory)
    from ..utils import get_device, export_video_robust
except ImportError:
    try:
        # Try absolute import (when run from project root)
        from src.tasks.vision.utils import get_device, export_video_robust
    except ImportError:
        # Fallback for direct execution
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import get_device, export_video_robust

# Define a dataclass to hold model configuration
@dataclass
class ModelConfig:
    base: str
    steps: int
    file_prefix: str

# Enum for different base models
class BaseModelName(Enum):
    EPIC_REALISM = ModelConfig(base="emilianJR/epiCRealism", steps=8, file_prefix="e1_epic")
    TOONYOU = ModelConfig(base="frankjoshua/toonyou_beta6", steps=8, file_prefix="e1_toonyou")
    REALISTIC_VISION = ModelConfig(base="SG161222/Realistic_Vision_V5.1_noVAE", steps=8, file_prefix="e1_realistic")
    RESIDENT_CARTOON = ModelConfig(base="Yntec/ResidentCNZCartoon3D", steps=1, file_prefix="e1_animatediff")

def main():
    # Use our utility function to get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    dtype = torch.float16
    
    # Select model config from enum (easy to change)
    model_config = BaseModelName.TOONYOU.value
    
    # Set up motion adapter and repository
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{model_config.steps}step_diffusers.safetensors"
    fileName = f"{model_config.file_prefix}_{model_config.steps}step.gif"
    
    # Create the adapter and load state dict
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    
    # Create the pipeline with the selected base model
    pipe = AnimateDiffPipeline.from_pretrained(
        model_config.base, 
        motion_adapter=adapter, 
        torch_dtype=dtype
    ).to(device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing", 
        beta_schedule="linear"
    )
    
    # Define the prompt
    prompt = """
    Vast golden desert under a bright midday sun, two tiny figures riding camels in the far distance, shimmering heat waves, soft sand dunes, pale blue sky, serene and epic atmosphere.
    """
    
    # Generate the video frames
    output = pipe(
        prompt=prompt, 
        guidance_scale=0.8, 
        num_inference_steps=model_config.steps
    )
    
    # Export the result
    export_to_gif(output.frames[0], fileName)
    print(f"Video exported to {fileName}")
    
    # Export as MP4
    # Uncomment to export as MP4 as well
    # export_video_robust(output.frames[0], f"{model_config.file_prefix}_{model_config.steps}step.mp4")

if __name__ == "__main__":
    main()
