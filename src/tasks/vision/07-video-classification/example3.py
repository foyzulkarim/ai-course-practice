from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch
import cv2

# Load the video and extract frames
video_path = '../videos/bird.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize frame to 224x224 and convert from BGR to RGB
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()

# Convert frames to a list of numpy arrays
frames = [np.array(frame) for frame in frames]

# Ensure we have exactly 16 frames (as required by the model)
if len(frames) > 16:
    frames = frames[:16]  # Truncate to 16 frames
elif len(frames) < 16:
    # Pad with the last frame if there are fewer than 16 frames
    frames.extend([frames[-1]] * (16 - len(frames)))

# Convert frames to a list of torch tensors
video = [torch.tensor(frame).permute(2, 0, 1) for frame in frames]  # Shape: [3, 224, 224]

# Load the processor and model
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# Preprocess the video frames
inputs = processor(video, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])