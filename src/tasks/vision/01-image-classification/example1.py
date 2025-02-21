from enum import Enum
from transformers import pipeline

# Enum to store model names
class ModelName(Enum):
    RESNET50 = "microsoft/resnet-50"
    VIT_TINY_PATCH16_224 = "WinKawaks/vit-tiny-patch16-224"
    # Add more models as needed

def main():
    # Initialize the image classification pipeline using the enum value
    clf = pipeline("image-classification", model=ModelName.VIT_TINY_PATCH16_224.value)
    
    results = clf("bird.jpg")
    
    # Output the classification results
    print(results)

if __name__ == "__main__":
    main()

# default model: [{'label': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'score': 0.3391473591327667}, {'label': 'jay', 'score': 0.29072731733322144}, {'label': 'brambling, Fringilla montifringilla', 'score': 0.09743645042181015}, {'label': 'jacamar', 'score': 0.024195769801735878}, {'label': 'robin, American robin, Turdus migratorius', 'score': 0.019107399508357048}]
# microsoft/resnet: [{'label': 'jay', 'score': 0.8956521153450012}, {'label': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'score': 0.042736127972602844}, {'label': 'prairie chicken, prairie grouse, prairie fowl', 'score': 0.007138445507735014}, {'label': 'brambling, Fringilla montifringilla', 'score': 0.005858154501765966}, {'label': 'water ouzel, dipper', 'score': 0.005622026510536671}]
# WinKawaks/vit-tiny-patch16-224: [{'label': 'jay', 'score': 0.4335988163948059}, {'label': 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'score': 0.25469231605529785}, {'label': 'brambling, Fringilla montifringilla', 'score': 0.22699926793575287}, {'label': 'robin, American robin, Turdus migratorius', 'score': 0.01921648532152176}, {'label': 'jacamar', 'score': 0.007132253143936396}]