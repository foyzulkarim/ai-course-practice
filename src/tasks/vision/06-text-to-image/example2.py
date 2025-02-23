import torch
from diffusers import DiffusionPipeline
import psutil 

# Check total system memory in GB
total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"Total system memory: {total_memory_gb:.2f} GB")

# Only run the model if the machine has 30+ GB of RAM
if total_memory_gb < 30:
    raise RuntimeError(f"Insufficient memory: {total_memory_gb:.2f} GB detected. At least 30 GB of RAM is required.")

device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipe.to(device)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
result = pipe(prompt)
image = result.images[0]

# Save the generated image
image.save("astronaut.png")