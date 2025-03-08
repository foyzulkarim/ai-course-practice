import torch
from transformers import pipeline
import requests
from io import BytesIO
import time
import numpy as np
import librosa
import soundfile as sf

def download_audio_file(url):
    """Download an audio file from a URL"""
    response = requests.get(url)
    return BytesIO(response.content)

def load_audio_from_bytesio(audio_bytes):
    """Convert BytesIO to numpy array for the pipeline"""
    # Create a temporary file-like object for soundfile
    audio_bytes.seek(0)  # Reset pointer to beginning of file
    
    # Load audio using soundfile
    audio_data, sampling_rate = sf.read(audio_bytes)
    
    # Convert to mono if needed
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    
    return audio_data, sampling_rate

def main():
    # Check for available devices (CUDA, MPS, or CPU)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Initialize the ASR pipeline with Whisper Turbo
    print("Loading Whisper Turbo model...")
    pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-large-v3-turbo",
        device=device
    )
    
    # Option 1: Use a local audio file
    # audio_file_path = "path/to/your/audio/file.wav"  # Uncomment and replace with your file path
    # audio_data, sampling_rate = librosa.load(audio_file_path, sr=16000)
    
    # # Option 2: Download a sample audio file from the web
    # audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    # print(f"Downloading sample audio from {audio_url}")
    # response = requests.get(audio_url)
    # response.raise_for_status()
    # audio_bytes_io = BytesIO(response.content)  # Save a copy for later

    # Option 2: Use a local audio file (mlk.flac)
    audio_file_path = "mlk.flac"
    audio_bytes = open(audio_file_path, "rb").read()
    audio_bytes_io = BytesIO(audio_bytes)
    
    audio_data, sampling_rate = load_audio_from_bytesio(audio_bytes_io)
    
    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sampling_rate != 16000:
        print(f"Resampling audio from {sampling_rate}Hz to 16000Hz")
        audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    # Transcribe the audio
    print("Transcribing audio...")
    start_time = time.time()
    
    result = pipe(
        audio_data,  # Now using numpy array instead of BytesIO
        generate_kwargs={"max_new_tokens": 256},
        return_timestamps=True  # Get word-level timestamps
    )
    
    end_time = time.time()
    
    # Print results
    print("\nTranscription Result:")
    print(result["text"])
    
    if "chunks" in result:
        print("\nChunks with timestamps:")
        for chunk in result["chunks"]:
            print(f"{chunk['timestamp']}: {chunk['text']}")
    
    print(f"\nTranscription took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
