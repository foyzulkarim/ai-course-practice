#!/bin/bash

# Script to build and run a custom Ollama Docker image that pulls and runs
# a specified model at startup

# Script usage function
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --model MODEL    : AI model to run (default: llama3:8b)"
  echo "  -p, --port PORT      : Host port to map to container port 11434 (default: 11434)"
  echo "  -n, --name NAME      : Name for the container (default: derived from MODEL name)"
  echo "  -b, --build          : Build a new image before running (default: false)"
  echo "  -i, --image NAME     : Custom image name (default: custom-ollama)"
  echo "  -h, --help           : Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                           # Runs llama3:8b on port 11434"
  echo "  $0 -m mistral:7b             # Runs mistral:7b on port 11434"
  echo "  $0 -m llama3:70b -p 11437    # Runs llama3:70b on port 11437"
  echo "  $0 -b -m phi3:mini           # Build new image and run phi3:mini"
  echo "  $0 -b -i myollama:latest -m phi3:mini   # Build with custom image name"
}

# Default values
MODEL="llama3:8b"
HOST_PORT="11434"
CONTAINER_NAME=""
IMAGE_NAME="custom-ollama"
BUILD_IMAGE=false

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
    -b|--build)
      BUILD_IMAGE=true
      shift
      ;;
    -i|--image)
      IMAGE_NAME="$2"
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

# Generate container name from model and port if not provided
if [ -z "$CONTAINER_NAME" ]; then
  # Strip special characters from model name and prefix with "ollama-"
  MODEL_PART=$(echo $MODEL | sed 's/[:\/]/-/g' | sed 's/[^a-zA-Z0-9-]//g')
  CONTAINER_NAME="ollama-${MODEL_PART}-p${HOST_PORT}"
fi

# Build the image if requested
if [ "$BUILD_IMAGE" = true ]; then
  echo "Building Docker image: $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" $(dirname "$0")
  if [ $? -ne 0 ]; then
    echo "Error building Docker image"
    exit 1
  fi
fi

echo "Starting Ollama container:"
echo "  Model: $MODEL"
echo "  Port: $HOST_PORT"
echo "  Container name: $CONTAINER_NAME"
echo "  Image: $IMAGE_NAME"

# Run the container with the specified model
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$HOST_PORT:11434" \
  -v ollama-data:/root/.ollama \
  --restart no \
  "$IMAGE_NAME" "$MODEL"

if [ $? -ne 0 ]; then
  echo "Failed to start container. If the container already exists, try removing it with:"
  echo "  docker rm -f $CONTAINER_NAME"
  exit 1
fi

echo ""
echo "Container started successfully!"
echo "API is available at: http://localhost:$HOST_PORT/v1/chat/completions"
echo "To check container logs: docker logs $CONTAINER_NAME"
echo "To stop the container: docker stop $CONTAINER_NAME"
echo "To restart the container: docker start $CONTAINER_NAME"