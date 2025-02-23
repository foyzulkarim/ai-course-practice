from enum import Enum
from transformers import pipeline
from PIL import Image

class ModelName(Enum):
    BLIP_IMAGE_CAPTIONING_BASE = "Salesforce/blip-image-captioning-base" # 'a view of a hill with a blue sky'
    BLIP2_OPT_2_7B = "Salesforce/blip2-opt-2.7b" # 'a view of the hills and valleys from a hilltop\n'

captioner = pipeline("image-to-text", model=ModelName.BLIP2_OPT_2_7B.value)

image = Image.open("../images/bird.jpg")
result = captioner(image)

print(result)

# Output:
# skyline: [{'generated_text': 'a view of the city from the air\n'}]
# coffee: [{'generated_text': 'a wooden tray with four cups of coffee\n'}]
# food: [{'generated_text': 'a plate of food on a table\n'}]
# bird: [{'generated_text': 'a blue bird sits on a piece of wood\n'}]