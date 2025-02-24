from enum import Enum
from transformers import pipeline

class ModelName(Enum):
    MCG_NJU_VIDEOME ="MCG-NJU/videomae-base-finetuned-kinetics"
    # facebook/timesformer-base-finetuned-k400
    FACEBOOK_TIMESFORMER_BASE_FINETUNED_K400 = "facebook/timesformer-base-finetuned-k400"

# Initialize the pipeline with candidate labels
pipe = pipeline("video-classification", model=ModelName.FACEBOOK_TIMESFORMER_BASE_FINETUNED_K400.value)

# Provide the path to your video file
video_path = "../videos/bird.mp4"  # Replace with your video file path

# Run video classification
results = pipe(video_path)
print(results)
