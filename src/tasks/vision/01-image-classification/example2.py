from enum import Enum
from transformers import pipeline

class ModelName(Enum):
    RESNET50 = "microsoft/resnet-50"
    VIT_TINY_PATCH16_224 = "WinKawaks/vit-tiny-patch16-224"

def infer_generic_category(label):
    pipe = pipeline(model="facebook/bart-large-mnli")
    result = pipe(label,
        candidate_labels=['vehicle', 'animal', 'food', 'furniture', 'clothing', 'accessory', 'electronic', 'appliance', 'personal care', 'tool', 'indoor', 'outdoor'])
    
    print("Inference results:", result)
    # Get the index of max score
    max_score_index = result['scores'].index(max(result['scores']))
    
    # Get the corresponding label
    predicted_category = result['labels'][max_score_index]
    
    print(f'Category: {predicted_category} (confidence: {result["scores"][max_score_index]:.2f})')
    return predicted_category

def main():
    # Initialize the image classification pipeline
    clf = pipeline("image-classification", model=ModelName.RESNET50.value)
    
    results = clf("bird.jpg")
    print("Classification results:", results)
    
    identified_label = infer_generic_category(results[0]["label"])
    print(f"Generic category: {identified_label}")

if __name__ == "__main__":
    main()