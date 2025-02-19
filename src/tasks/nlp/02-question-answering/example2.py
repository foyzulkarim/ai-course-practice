from transformers import pipeline

def load_context(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def main():
    # Load the context from text.txt
    context = load_context('./text.txt')
    
    # Initialize the question answering pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    # Get a question from the user
    question = input("Enter your question: ")
    
    # Get answer from the model
    result = qa_pipeline(question=question, context=context)
    
    # Print the answer
    print("Answer:", result['answer'])

if __name__ == "__main__":
    main()