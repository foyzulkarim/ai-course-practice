from transformers import pipeline

def translate(text):
    translator = pipeline("translation_en_to_fr")
    translation = translator(text, max_length=400)
    return translation[0]['translation_text']

if __name__ == "__main__":
    text_to_translate = "Hello world"  # Hardcoded input text
    result = translate(text_to_translate)
    print("Translated text:", result)