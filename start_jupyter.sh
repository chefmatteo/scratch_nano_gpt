#!/bin/bash
# Start Jupyter Lab with the scratch-nano-gpt environment

echo "🚀 Starting Jupyter Lab with Scratch Nano GPT environment..."
echo "📦 Available kernels:"
uv run jupyter kernelspec list | grep scratch-nano-gpt

echo ""
echo "🌐 Starting Jupyter Lab..."
echo "💡 Select 'Scratch Nano GPT' kernel when creating new notebooks"
echo ""

uv run jupyter lab
