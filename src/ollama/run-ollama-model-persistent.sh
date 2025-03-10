#!/bin/bash

# Exit on error, treat undefined variables as errors, and exit on pipeline failures
set -euo pipefail

# Script usage function
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --model MODEL    : AI model to run (default: llama3:8b)"
  echo "  -p, --port PORT      : Host port to map to container port 11434 (default: 11436)"
  echo "  -n, --name NAME      : Name for the container (default: derived from MODEL name and PORT)"
  echo "  -h, --help           : Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                           # Runs llama3:8b on port 11436 in container named ollama-llama3-8b-11436"
  echo "  $0 -m mistral:7b             # Runs mistral:7b on port 11436 in container named ollama-mistral-7b-11436"
  echo "  $0 -m llama3:70b -p 11437    # Runs llama3:70b on port 11437 in container named ollama-llama3-70b-11437"
  echo "  $0 -m llama3:8b -n my-llm    # Runs llama3:8b on port 11436 in container named my-llm"
  echo "  $0 -p 11438 -n my-llm -m llama3:8b  # All parameters specified"
}

# Default values
MODEL=""
HOST_PORT=""
CONTAINER_NAME=""
CLEANUP_VOLUMES=false

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
    -c|--cleanup-volumes)
      CLEANUP_VOLUMES=true
      shift 1
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

# Validate MODEL
if [ -z "$MODEL" ]; then
  echo "Error: Model must be specified."
  usage
  exit 1
fi

# Validate PORT
if ! [[ "$HOST_PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: Port must be a valid number."
  exit 1
fi

# Generate container name from model if not provided
if [ -z "$CONTAINER_NAME" ]; then
  # Strip special characters from model name and prefix with "ollama-", append port number
  CONTAINER_NAME="ollama-$(echo $MODEL | sed 's/[:\/]/-/g' | sed 's/[^a-zA-Z0-9-]//g')-$HOST_PORT"
fi

# Find a free port starting from the specified HOST_PORT
ORIGINAL_PORT="$HOST_PORT"
while lsof -Pi :$HOST_PORT -sTCP:LISTEN -t > /dev/null ; do
  echo "Port $HOST_PORT is already in use, trying next port..."
  HOST_PORT=$((HOST_PORT + 1))
done

if [ "$HOST_PORT" != "$ORIGINAL_PORT" ]; then
  echo "Found free port: $HOST_PORT"
  # Update container name if it was auto-generated with port number
  if [[ "$CONTAINER_NAME" == *"-$ORIGINAL_PORT" ]]; then
    CONTAINER_NAME="${CONTAINER_NAME/-$ORIGINAL_PORT/-$HOST_PORT}"
    echo "Container name updated to: $CONTAINER_NAME"
  fi
fi

# Generate a unique volume name based on container name
VOLUME_NAME="${CONTAINER_NAME}-volume"
echo "Setting up persistent Ollama container:"
echo "  Model: $MODEL"
echo "  Port: $HOST_PORT"
echo "  Container name: $CONTAINER_NAME"
echo "  Volume name: $VOLUME_NAME"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  echo "Container $CONTAINER_NAME already exists"
  
  # Check if it's running, if not start it
  if ! docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Starting existing container..."
    docker start "$CONTAINER_NAME"
  fi
else
  echo "Creating new Ollama container..."
  # Create a new container in detached mode with specified port and automatic volume management
  if ! docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$HOST_PORT":11434 \
    --restart unless-stopped \
    ollama/ollama; then
    echo "Error: Failed to create container."
    exit 2
  fi

  echo "Container created successfully."
fi

# Generate a unique volume name based on container name (if cleanup is enabled)
if [ "$CLEANUP_VOLUMES" = true ]; then
  VOLUME_NAME=$(docker inspect -f '{{ range .Mounts }}{{.Name}} {{end}}' "$CONTAINER_NAME" | tr ' ' '\n' | grep "^${CONTAINER_NAME}-volume$")
  if [ -n "$VOLUME_NAME" ]; then
    echo "Cleaning up old volume: $VOLUME_NAME"
    docker volume rm "$VOLUME_NAME"
  fi
fi

# Create a script in the container to run the model persistently
docker exec "$CONTAINER_NAME" bash -c "cat > /root/run_model.sh << 'EOF'
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
docker exec "$CONTAINER_NAME" chmod +x /root/run_model.sh

# Execute the script to run the model
docker exec "$CONTAINER_NAME" /root/run_model.sh "$MODEL"

# Verify the model is running
sleep 2
if ! docker exec "$CONTAINER_NAME" bash -c "ps -p \$(cat /root/ollama_model.pid 2>/dev/null) > /dev/null"; then
  echo "Failed to start model $MODEL. Check log for details:"
  docker exec "$CONTAINER_NAME" cat /root/ollama_model.log
  exit 3
else
  echo "Model $MODEL is now running in the persistent container with PID \$(cat /root/ollama_model.pid)"
  echo "Last 5 lines of log:"
  docker exec "$CONTAINER_NAME" tail -5 /root/ollama_model.log
fi

echo ""
echo "The container will continue running in the background."
echo ""
echo "API is available at: http://localhost:$HOST_PORT/v1/models"
echo "To connect to the container: docker exec -it $CONTAINER_NAME bash"
echo "To check model logs: docker exec $CONTAINER_NAME cat /root/ollama_model.log"
echo "To stop the container: docker stop $CONTAINER_NAME"
echo "To stop just the model: docker exec $CONTAINER_NAME pkill -f 'ollama run'"