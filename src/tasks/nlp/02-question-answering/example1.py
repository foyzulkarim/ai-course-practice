from transformers import pipeline
from transformers.utils import TRANSFORMERS_CACHE
import os

def get_model_size(model):
    # Calculate number of bytes (assuming float32 parameters)
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * 4  # 4 bytes per parameter
    size_mb = size_bytes / (1024 ** 2)
    return size_mb

def main():
    # Print cache directory location
    print(f"Models are cached in: {TRANSFORMERS_CACHE}")
    
    # Initialize the question answering pipeline with a pre-trained model
    qa_pipeline = pipeline("question-answering")

    # print default model name
    print('Default model name:')
    print(qa_pipeline.model.config.name_or_path)
    
    model_size_mb = get_model_size(qa_pipeline.model)
    print("Model size (approx): {:.2f} MB".format(model_size_mb))
    
    # Define a context and a question
    context = (
        "The Transformers library by Hugging Face offers state-of-the-art implementations "
        "of various transformer models, including those for question answering tasks. "
        "These models can be used in chatbots, customer support systems, educational apps, and more."
    )
    question = "What does the Transformers library offer?"
    print("Question:", question)
    
    # Get the answer for the question
    result = qa_pipeline(question=question, context=context)
    
    print("Answer:", result['answer'])

if __name__ == "__main__":
    main()