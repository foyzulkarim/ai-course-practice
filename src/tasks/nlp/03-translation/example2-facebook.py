# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # Load NLLB model (supports 200+ languages)
# model_name = "facebook/nllb-200-distilled-600M"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Text to translate
# english_text = "Hello, how are you today? I hope you're doing well."

# # Tokenize with source language tag
# inputs = tokenizer(english_text, return_tensors="pt")

# # Set the target language (Arabic)
# translated = model.generate(
#     **inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["ara_Arab"],
#     max_length=100
# )

# # Decode the output
# arabic_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# print(arabic_text)

# First, install the required libraries if you haven't already:
# pip install transformers torch sentencepiece

# Install required libraries if not already installed:
# pip install transformers torch sentencepiece

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# def translate_to_arabic_nllb(text):
#     # Load the tokenizer and model
#     model_name = "facebook/nllb-200-distilled-600M"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Get the language id mapping. First try the tokenizer, then the model config.
#     lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
#     if lang_code_to_id is None:
#         lang_code_to_id = getattr(model.config, "lang_code_to_id", None)
    
#     # Print all of the language codes available
#     if lang_code_to_id:
#         print("Available language codes:")
#         for code, token_id in lang_code_to_id.items():
#             print(f"  {code}: {token_id}")
#     else:
#         print("No language id mapping was found in the tokenizer or model config.")

#     # Then try to find the language id for 'ara_Arab'
#     if lang_code_to_id is None or "ara_Arab" not in lang_code_to_id:
#         raise ValueError(f"Cannot find language id for 'ara_Arab'. Available language codes: {list(lang_code_to_id.keys()) if lang_code_to_id else 'None'}")
    
#     forced_bos_token_id = lang_code_to_id["ara_Arab"]

#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt", padding=True)

#     # Generate the translation with target language set to Arabic
#     translated_tokens = model.generate(
#         **inputs,
#         max_length=100,
#         num_beams=5,
#         early_stopping=True,
#         forced_bos_token_id=forced_bos_token_id
#     )

#     # Decode and return the output
#     translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
#     return translated_text

# # Example usage
# if __name__ == "__main__":
#     english_text = "Hello, how are you today?"
#     try:
#         arabic_translation = translate_to_arabic_nllb(english_text)
#         print(f"English: {english_text}")
#         print(f"Arabic: {arabic_translation}")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")