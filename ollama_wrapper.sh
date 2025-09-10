#!/bin/bash
# Ollama wrapper script to ensure it runs on port 3040

# Set the port
export OLLAMA_HOST=127.0.0.1:3040

# Check if ollama is already running on port 3040
if curl -s http://127.0.0.1:3040/api/tags > /dev/null 2>&1; then
    echo "Ollama is already running on port 3040"
    # Just run the command
    ollama "$@"
else
    echo "Starting Ollama server on port 3040..."
    # Kill any existing ollama processes
    pkill -f "ollama serve" 2>/dev/null || true
    # Start ollama in background
    ollama serve &
    # Wait for it to start
    for i in {1..10}; do
        if curl -s http://127.0.0.1:3040/api/tags > /dev/null 2>&1; then
            echo "Ollama started successfully on port 3040"
            break
        fi
        sleep 1
    done
    # Run the command
    ollama "$@"
fi
