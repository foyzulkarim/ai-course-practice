from transformers import pipeline
from transformers.utils import TRANSFORMERS_CACHE
import os
import subprocess
import platform

def get_directory_raw_size(directory):
    total_size = 0
    seen_inodes = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                # Use lstat to not follow symlinks
                stat = os.lstat(fp)
                # Check if we've already counted this file (via inode)
                if stat.st_ino in seen_inodes:
                    continue
                seen_inodes.add(stat.st_ino)
                total_size += stat.st_size
            except Exception as e:
                total_size += os.path.getsize(fp)
    return total_size

def get_directory_size(directory):
    """
    Get the directory size in bytes using the 'du' command.
    Compatible with macOS, Linux, and other Unix-like systems.
    """
    try:
        # Use 'du -sb' for bytes on Linux, and 'du -sk' for kilobytes on macOS
        if platform.system() == "Linux":
            result = subprocess.run(['du', '-sb', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            size_in_bytes = int(result.stdout.split()[0])
        else:
            # macOS and other Unix-like systems
            result = subprocess.run(['du', '-sk', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            size_in_kb = int(result.stdout.split()[0])
            size_in_bytes = size_in_kb * 1024  # Convert kilobytes to bytes

        return size_in_bytes
    except Exception as e:
        raise Exception(f"Error getting directory size: {e}")

def main():
    # Print cache directory location
    print(f"Models are cached in: {TRANSFORMERS_CACHE}")
    
    # Initialize the question answering pipeline with a pre-trained model
    qa_pipeline = pipeline("question-answering")

    # Print default model name
    print('Default model name:')
    model_name = qa_pipeline.model.config.name_or_path
    print(model_name)
    
    # Derive the model folder name from the model name
    # Example: "distilbert/distilbert-base-cased-distilled-squad" becomes 
    # "models--distilbert--distilbert-base-cased-distilled-squad"
    model_folder_name = "models--" + model_name.replace("/", "--")
    model_folder_path = os.path.join(TRANSFORMERS_CACHE, model_folder_name)
    
    if os.path.exists(model_folder_path):
        # Get the model size in megabytes
        # folder_size_mb = get_directory_size(model_folder_path) / (1024 ** 2)
        folder_size_mb = get_directory_raw_size(model_folder_path) / (1024 ** 2)
        print("Model folder size on disk (approx): {:.2f} MB".format(folder_size_mb))
    else:
        print("Model folder not found:", model_folder_path)
    
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