from PIL import Image
from transformers import pipeline

def main():
    # Initialize the object detection pipeline
    object_detector = pipeline("object-detection")
    
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