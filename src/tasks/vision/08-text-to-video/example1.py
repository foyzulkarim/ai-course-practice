import os
import sys
import torch
from enum import Enum
from dataclasses import dataclass
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Import utility functions
import sys
import os

# Add parent directory to path to simplify imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Standard import when running from proper package structure
    from ..utils import get_device, export_video_robust
except ImportError:
    # Fallback for direct script execution
    from utils import get_device, export_video_robust

# Define a dataclass to hold model configuration
@dataclass
class ModelConfig:
    """Configuration class for text-to-video generation models.
    
    Attributes:
        base: HuggingFace model repository path
        steps: Number of inference steps to use
        file_prefix: Prefix for output filenames
    """
    base: str
    steps: int
    file_prefix: str

# Enum for different base models
class BaseModelName(Enum):
    """Available pretrained models for text-to-video generation.
    
    Each entry contains the model repository, preferred number of steps,
    and naming convention for output files.
    """
    EPIC_REALISM = ModelConfig(base="emilianJR/epiCRealism", steps=4, file_prefix="e1_epic")
    TOONYOU = ModelConfig(base="frankjoshua/toonyou_beta6", steps=4, file_prefix="e1_toonyou")
    REALISTIC_VISION = ModelConfig(base="SG161222/Realistic_Vision_V5.1_noVAE", steps=4, file_prefix="e1_realistic")
    RESIDENT_CARTOON = ModelConfig(base="Yntec/ResidentCNZCartoon3D", steps=1, file_prefix="e1_animatediff") # WARNING: this uses CPU, hence too slow

def main():    
    # Select model config from enum (easy to change)
    model_config = BaseModelName.EPIC_REALISM.value
    
    # Constants for repositories and file formats
    REPOSITORIES = {
        "animatediff": "ByteDance/AnimateDiff-Lightning"
    }

    # Use our utility function to get the best available device
    # if model_config used RESIDENT_CARTOON, then device is "cpu" else get the best available device using the function
    device = "cpu" if model_config == BaseModelName.RESIDENT_CARTOON.value else get_device()
    # device = get_device()
    # device = "cpu"
    print(f"Using device: {device}")
    
    dtype = torch.float16
    
    # Set up motion adapter and repository
    repo = REPOSITORIES["animatediff"]
    ckpt = f"animatediff_lightning_{model_config.steps}step_diffusers.safetensors"
    file_name = f"{model_config.file_prefix}_{model_config.steps}step.gif"
    
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

    pipe.enable_attention_slicing()
    
    # Example prompts for text-to-video generation
    PROMPTS = {
        "desert": """Vast golden desert under a bright midday sun, two tiny figures riding 
                  camels in the far distance moving very slowly, shimmering heat waves, soft sand dunes, 
                  pale blue sky, serene and epic atmosphere.""",
        "forest": """Lush green forest with tall trees, sunlight filtering through the canopy,
                   a small stream with clear water flowing over rocks, birds flying between branches.""",
        "ocean": """Deep blue ocean waves with white foam, a small boat rocking on the surface,
                  seagulls flying overhead, cloudy sky with rays of sunlight breaking through."""
    }
    
    # Select a prompt to use
    prompt = PROMPTS["desert"]
    
    # Generate the video frames
    output = pipe(
        prompt=prompt, 
        guidance_scale=0.8, 
        num_inference_steps=model_config.steps
    )
    
    # Export the result
    export_to_gif(output.frames[0], file_name)
    print(f"Video exported to {file_name}")
    
    # Export as MP4
    # Uncomment to export as MP4 as well
    # export_video_robust(output.frames[0], f"{model_config.file_prefix}_{model_config.steps}step.mp4")
    # print(f"Video exported to {model_config.file_prefix}_{model_config.steps}step.mp4")

if __name__ == "__main__":
    main()
