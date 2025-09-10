#!/bin/bash

# Script to stop all processes running on port 3040 (typically Ollama)

echo "Checking for processes running on port 3040..."

# Get all process IDs running on port 3040
PIDS=$(lsof -ti:3040)

if [ -z "$PIDS" ]; then
    echo "No processes found running on port 3040"
    exit 0
fi

echo "Found processes running on port 3040:"
for pid in $PIDS; do
    echo "  PID: $pid - $(ps -p $pid -o command= 2>/dev/null || echo 'Process not found')"
done

echo ""
read -p "Do you want to stop these processes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping processes..."
    for pid in $PIDS; do
        echo "Stopping PID: $pid"
        kill $pid
        sleep 1
        
        # Check if process is still running
        if kill -0 $pid 2>/dev/null; then
            echo "Process $pid still running, force killing..."
            kill -9 $pid
        else
            echo "Process $pid stopped successfully"
        fi
    done
    
    # Verify no processes are still running on port 3040
    sleep 2
    REMAINING_PIDS=$(lsof -ti:3040)
    if [ -z "$REMAINING_PIDS" ]; then
        echo "✅ All processes on port 3040 have been stopped"
    else
        echo "⚠️  Some processes may still be running on port 3040: $REMAINING_PIDS"
    fi
else
    echo "Operation cancelled"
fi

