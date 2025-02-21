from enum import Enum
from PIL import Image
from transformers import pipeline

# Enum of the model names
class ModelName(Enum):
    DETR_RESNET_50 = "facebook/detr-resnet-50"
    DETR_TABLE_TRANSFORMER = "microsoft/table-transformer-detection"

def main():
    # Initialize the object detection pipeline, default: facebook/detr-resnet-50
    object_detector = pipeline("object-detection", model=ModelName.DETR_RESNET_50.value)

     # print model name
    print('Model name:')
    print(object_detector.model.config.name_or_path)
    
    # Load an example image from a local file
    image_path = "../images/food.jpg"
    image = Image.open(image_path)

    # Perform object detection on the image
    results = object_detector(image)

    # Print the detection results
    print(results)

    # Print the detected objects and their scores
    for result in results:
        print(f"Object: {result['label']}, Score: {result['score']}")

if __name__ == "__main__":
    main()