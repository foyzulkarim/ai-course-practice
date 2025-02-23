# https://huggingface.co/tasks/image-to-image#inference

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

def main():
    # Load an example input image (ensure the path points to a valid image file)
    init_image = load_image("../images/astronaut.png")
    
    # Define your text prompt as per the official example
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    
    # Load the pre-trained model using the official Stable Diffusion XL refiner model
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, # remove this line to use the default torch.float32
        variant="fp16", # remove this line to use the default fp32
        use_safetensors=True
    )
    
    # Enable CPU offloading to reduce memory usage
    # pipe.enable_model_cpu_offload()
    device = "cpu"
    # Ensure CUDA is available and PyTorch is compiled with CUDA support before moving to GPU
    if torch.version.cuda is not None and torch.cuda.is_available():
        device = "cuda"
        pipe = pipe.to(device)
    elif torch.backends.mps.is_available():
        device = "mps"
        pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Generate a new image from the input image and prompt
    result = pipe(prompt, image=init_image, strength=0.7)
    gen_image = result.images[0]
    
    # Create a grid to display the input and output images side-by-side
    grid = make_image_grid([init_image, gen_image], rows=1, cols=2)
    
    # Save the resulting grid image
    grid.save("output_image_grid_2.png")
    print("Generated image grid saved as output_image_grid.png")

if __name__ == "__main__":
    main()