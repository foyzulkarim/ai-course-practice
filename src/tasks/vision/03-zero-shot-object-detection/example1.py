from enum import Enum
from PIL import Image
from transformers import pipeline

# Enum of the model names
class ModelName(Enum):
    OWLVIT_BASE = "google/owlvit-base-patch32"
    OWLVI2 = 'google/owlv2-base-patch16-ensemble'

def main():
    # Initialize the zero-shot object detection pipeline
    detector = pipeline("zero-shot-object-detection", model=ModelName.OWLVI2.value)

    # print model name
    print('Model name:')
    print(detector.model.config.name_or_path)
    
    # Load an example image from a local file
    image_path = "../images/bird.jpg"
    image = Image.open(image_path)

    # Define candidate labels for detection
    # labels = [
    #   'bread',
    #   'meat',
    #   'salad',
    #   'burger',
    #   'vegitables',
    #   'sandwich',
    #   'vegetable burger'
    # ]

    labels = [
        'bird',
        'cat',
        'dog',       
    ]

    # Perform zero-shot object detection
    results = detector(image, labels)

    # Print the detection results
    # print(results)

    # Print the detected objects and their scores
    for result in results:
        print(f"Object: {result['label']}, Score: {result['score']}, Box: {result['box']}")

if __name__ == "__main__":
    main()
