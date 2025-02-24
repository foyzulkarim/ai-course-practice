from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch
import cv2
from pathlib import Path

# Load and preprocess video
video_path = Path('../videos/waterfalls.mp4').resolve()
cap = cv2.VideoCapture(str(video_path))
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize frame to 224x224
    frame = cv2.resize(frame, (224, 224))
    frames.append(frame)
cap.release()

# Take 8 evenly spaced frames from the video
if len(frames) > 8:
    indices = np.linspace(0, len(frames)-1, 8, dtype=int)
    frames = [frames[i] for i in indices]
else:
    # If video is too short, duplicate frames
    frames = (frames * (8 // len(frames) + 1))[:8]

# Convert to numpy array and normalize
video = np.array(frames).transpose(0, 3, 1, 2) / 255.0

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

inputs = processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
