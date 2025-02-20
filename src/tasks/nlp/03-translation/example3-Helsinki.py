# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")


# # Text to translate
# english_text = "Hello world"

# # Tokenize the text
# encoded = tokenizer(english_text, return_tensors="pt", padding=True)

# # Generate translation
# translated = model.generate(**encoded)

# # Decode the generated tokens to text
# arabic_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# print(arabic_text)

# # output: السلام عليكم عليكم

# ENGLIST TO ARABIC
from transformers import MarianTokenizer, MarianMTModel

def translate_to_arabic_marian(text):
    # Load the tokenizer and model for English to Arabic
    model_name = "Helsinki-NLP/opus-mt-en-ar"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode the output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

# Example usage
if __name__ == "__main__":
    english_text = "Hello, how are you today?"
    try:
        arabic_translation = translate_to_arabic_marian(english_text)
        print(f"English: {english_text}")
        print(f"Arabic: {arabic_translation}")  # Should output: مرحبا، كيف حالك اليوم؟
    except Exception as e:
        print(f"An error occurred: {str(e)}")

