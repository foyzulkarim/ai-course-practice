#!/bin/bash

# Script to debug Ollama container issues

if [ -z "$1" ]; then
  echo "Usage: $0 <container_name>"
  echo "Example: $0 ollama-llama3-8b-p11434"
  exit 1
fi

CONTAINER_NAME=$1

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  echo "Error: Container $CONTAINER_NAME does not exist"
  echo "Available containers:"
  docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  exit 1
fi

echo "===== Container Status ====="
docker ps -a --filter "name=$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

echo ""
echo "===== Container Logs ====="
docker logs $CONTAINER_NAME

# If container is running, check ollama status inside
if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  echo ""
  echo "===== Ollama Status ====="
  docker exec $CONTAINER_NAME ollama list

  echo ""
  echo "===== Server Log ====="
  docker exec $CONTAINER_NAME cat /var/log/ollama-serve.log 2>/dev/null || echo "No server log file found"

  echo ""
  echo "===== Container Stats ====="
  docker stats $CONTAINER_NAME --no-stream
else
  echo ""
  echo "⚠️ Container is not running. Cannot execute commands inside it."
  echo "To restart the container:"
  echo "  docker start $CONTAINER_NAME"
  echo "To remove and recreate the container:"
  echo "  docker rm $CONTAINER_NAME"
  echo "  ./run-ollama-model.sh -m <model_name> -p <port>"
fi

echo ""
echo "===== Debug Commands ====="
echo "To access container shell: docker exec -it $CONTAINER_NAME sh"
echo "To stop container: docker stop $CONTAINER_NAME"
echo "To remove container: docker rm -f $CONTAINER_NAME"
echo "To restart container: docker restart $CONTAINER_NAME"