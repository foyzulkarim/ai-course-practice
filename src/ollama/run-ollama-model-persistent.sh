#!/bin/bash

# Script usage function
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --model MODEL    : AI model to run (default: llama3:8b)"
  echo "  -p, --port PORT      : Host port to map to container port 11434 (default: 11436)"
  echo "  -n, --name NAME      : Name for the container (default: derived from MODEL name)"
  echo "  -h, --help           : Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                           # Runs llama3:8b on port 11436 in container named ollama-llama3-8b"
  echo "  $0 -m mistral:7b             # Runs mistral:7b on port 11436 in container named ollama-mistral-7b"
  echo "  $0 -m llama3:70b -p 11437    # Runs llama3:70b on port 11437 in container named ollama-llama3-70b"
  echo "  $0 -m llama3:8b -n my-llm    # Runs llama3:8b on port 11436 in container named my-llm"
  echo "  $0 -p 11438 -n my-llm -m llama3:8b  # All parameters specified"
}

# Default values
MODEL="llama3:8b"
HOST_PORT="11436"
CONTAINER_NAME=""

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -p|--port)
      HOST_PORT="$2"
      shift 2
      ;;
    -n|--name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Generate container name from model if not provided
if [ -z "$CONTAINER_NAME" ]; then
  # Strip special characters from model name and prefix with "ollama-"
  CONTAINER_NAME="ollama-$(echo $MODEL | sed 's/[:\/]/-/g' | sed 's/[^a-zA-Z0-9-]//g')"
fi

echo "Setting up persistent Ollama container:"
echo "  Model: $MODEL"
echo "  Port: $HOST_PORT"
echo "  Container name: $CONTAINER_NAME"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  echo "Container $CONTAINER_NAME already exists"
  
  # Check if it's running, if not start it
  if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Starting existing container..."
    docker start $CONTAINER_NAME
  fi
else
  echo "Creating new Ollama container..."
  # Create a new container in detached mode with specified port
  docker run -d \
    --name $CONTAINER_NAME \
    -p $HOST_PORT:11434 \
    -v ollama-data:/root/.ollama \
    --restart unless-stopped \
    ollama/ollama
  
  echo "Waiting for container to initialize..."
  sleep 5
fi

echo "Connecting to container and running model: $MODEL"

# Create a script in the container to run the model persistently
docker exec $CONTAINER_NAME bash -c "cat > /root/run_model.sh << 'EOF'
#!/bin/bash
MODEL=\$1
# Kill any existing ollama run processes
pkill -f 'ollama run' || true
# Run the model and redirect output to log file
nohup ollama run \$MODEL > /root/ollama_model.log 2>&1 &
echo \$! > /root/ollama_model.pid
echo \"Started model \$MODEL with PID \$(cat /root/ollama_model.pid)\"
EOF"

# Make the script executable
docker exec $CONTAINER_NAME chmod +x /root/run_model.sh

# Execute the script to run the model
docker exec $CONTAINER_NAME /root/run_model.sh "$MODEL"

# Verify the model is running
sleep 2
docker exec $CONTAINER_NAME bash -c "if ps -p \$(cat /root/ollama_model.pid 2>/dev/null) > /dev/null; then 
  echo 'Model $MODEL is now running in the persistent container with PID '\$(cat /root/ollama_model.pid);
  echo 'Last 5 lines of log:'; 
  tail -5 /root/ollama_model.log;
else 
  echo 'Failed to start model $MODEL. Check log for details:';
  cat /root/ollama_model.log;
  exit 1;
fi"

echo ""
echo "The container will continue running in the background."
echo ""
echo "API is available at: http://localhost:$HOST_PORT/v1/models"
echo "To connect to the container: docker exec -it $CONTAINER_NAME bash"
echo "To check model logs: docker exec $CONTAINER_NAME cat /root/ollama_model.log"
echo "To stop the container: docker stop $CONTAINER_NAME"
echo "To stop just the model: docker exec $CONTAINER_NAME pkill -f 'ollama run'"
