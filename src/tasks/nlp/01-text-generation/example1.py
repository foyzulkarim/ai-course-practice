from transformers import pipeline

def main():
    # Initialize text generation pipeline
    generator = pipeline("text-generation")

    # Specify a prompt.
    prompt = "Once upon a time"

    # print default model name
    print('Default model name:')
    print(generator.model.config.name_or_path) # openai-community/gpt2, size: ~600MB
    
    # Generate text with a maximum length of 50 tokens.
    results = generator(prompt, max_length=50, num_return_sequences=1)
    
    # Print the generated text.
    print("Generated text:")
    for result in results:
        print(result["generated_text"])

if __name__ == "__main__":
    main()
