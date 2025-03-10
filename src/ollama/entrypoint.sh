#!/bin/sh
set -e

# Determine the model name: Command-line argument takes precedence over environment variable
if [ -n "$1" ]; then
  MODEL_NAME=$1
else
  MODEL_NAME=${OLLAMA_MODEL:-llama3:8b}
fi

# Verify we have a model name
if [ -z "$MODEL_NAME" ]; then
  echo "Error: No model name provided."
  echo "Usage: docker run -e OLLAMA_MODEL=<model_name> my-ollama-image"
  echo "   or: docker run my-ollama-image <model_name>"
  exit 1
fi

echo "Starting Ollama service..."
# Start Ollama service in the background
ollama serve > /var/log/ollama-serve.log 2>&1 &
SERVE_PID=$!

# Wait for Ollama service to be ready
echo "Waiting for Ollama service to start..."
sleep 5
max_retries=30
retry_count=0

while ! ollama list > /dev/null 2>&1; do
  # Check if ollama serve is still running
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "Error: Ollama service crashed! Check logs:"
    cat /var/log/ollama-serve.log
    exit 1
  fi

  retry_count=$((retry_count+1))
  if [ $retry_count -ge $max_retries ]; then
    echo "Error: Ollama service failed to start after multiple retries. Logs:"
    cat /var/log/ollama-serve.log
    exit 1
  fi
  echo "Ollama service not ready yet, retrying in 2 seconds..."
  sleep 2
done

# Check if the model is already pulled
echo "Checking if model $MODEL_NAME is already pulled..."
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "Model $MODEL_NAME not found locally. Pulling model..."
  # Pull the model first to avoid issues with running it directly
  ollama pull "$MODEL_NAME" || {
    echo "Error: Failed to pull model $MODEL_NAME"
    # Keep container running for debugging
    echo "Keeping container alive for debugging. Connect with: docker exec -it <container> sh"
    echo "To view ollama server logs: cat /var/log/ollama-serve.log"
    tail -f /dev/null
  }
  echo "Model $MODEL_NAME pulled successfully!"
else
  echo "Model $MODEL_NAME already available locally."
fi

# Set up a trap to ensure we clean up properly
trap 'kill $SERVE_PID; exit' INT TERM

# Run the specified Ollama model
echo "Ollama service is running. Starting model: $MODEL_NAME"
echo "API is accessible at http://localhost:11434/v1/chat/completions"
echo "---------------------------------------------------------------"

# The API server is already running, so we just need to keep the container alive
echo "Ollama API server is already running with model $MODEL_NAME"
echo "To test the API, use: curl -X POST http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, how are you?\"}]}'"
echo "Press Ctrl+C to stop the container."

# Trap signals to clean up properly
trap 'kill $SERVE_PID; exit' INT TERM

# Keep container running indefinitely and monitor Ollama service
while true; do
  # Check if ollama service is still running
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "WARNING: Ollama service exited unexpectedly. Restarting..."
    
    # Restart the serve process
    echo "Restarting Ollama serve..."
    ollama serve > /var/log/ollama-serve.log 2>&1 &
    SERVE_PID=$!
    sleep 5
    
    # Make sure the model is loaded
    echo "Making sure model $MODEL_NAME is loaded..."
    ollama pull "$MODEL_NAME" > /dev/null 2>&1 || echo "WARNING: Failed to pull model $MODEL_NAME"
  fi
  
  # Log a heartbeat message
  echo "Container alive, Ollama is running. Model: $MODEL_NAME available via API. $(date)"
  
  # Sleep for 60 seconds before next check
  sleep 60
done