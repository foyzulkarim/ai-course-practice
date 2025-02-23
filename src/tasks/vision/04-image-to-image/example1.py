import torch
from diffusers import DiffusionPipeline
from PIL import Image

def main():
    # Load an example input image (ensure the path points to a valid image file)
    init_image = Image.open("../images/bird.jpg").convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Define your text prompt
    prompt = """A heavenly paradise with towering golden and silver palaces adorned with gemstones. Crystal-clear rivers flow below, surrounded by lush gardens with oversized fruit trees. Soft divine light illuminates the serene landscape, where silk-covered couches under cool shade offer peace and eternal comfort. The scene exudes tranquility, bliss, and ethereal beauty."""
    
    # Load the pre-trained model using the actively maintained identifier
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    
    device = "cpu"
    # Ensure CUDA is available and PyTorch is compiled with CUDA support before moving to GPU
    if torch.version.cuda is not None and torch.cuda.is_available():
        device = "cuda"
        pipe = pipe.to(device)
    elif torch.backends.mps.is_available():
        device = "mps"
        pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Generate a new image from the initial image and prompt
    result = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)
    result_image = result.images[0]
    
    # Save the resulting image
    result_image.save("output_image.png")
    print("Generated image saved as output_image.png")

if __name__ == "__main__":
    main()