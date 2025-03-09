# Use the official Ollama image as the base
FROM ollama/ollama:latest

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
# Optional environment variable for specifying the model
ENV OLLAMA_MODEL="llama3:8b"

# Copy the entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default model if none is specified (can be overridden at runtime)
CMD ["llama3:8b"]