# Text-to-Video Generation

This example demonstrates text-to-video generation using various pre-trained models with AnimateDiff.

## Features

- Automatic device detection (CUDA, MPS, CPU)
- Multiple pre-trained model options
- Robust video export with multiple fallbacks
- Modular code organization

## Dependencies

The following packages are required:

- torch
- diffusers
- transformers
- safetensors
- huggingface-hub
- accelerate
- imageio
- imageio-ffmpeg
- opencv-python (as fallback for video export)

Install with: `pip install -r requirements.txt`

## Usage

```bash
python example1.py
```

## Models

The example includes several pre-trained models:

- EPIC_REALISM: Realistic style images
- TOONYOU: Cartoon style images
- REALISTIC_VISION: Photorealistic style images
- RESIDENT_CARTOON: Cartoon 3D style images

## Customization

- Change the model by modifying the `model_config` variable
- Edit prompts in the `PROMPTS` dictionary
- Adjust inference parameters like guidance scale
- Uncomment the video export line to save as MP4