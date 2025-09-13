#!/usr/bin/env python3
"""
Simple inference script for quick text generation
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from output import load_model, generate_text

def quick_generate(prompt, model_path='trained_gpt_model.pth', max_tokens=200):
    """Quick text generation function"""
    try:
        # Load model
        model, chars, stoi, itos, device = load_model(model_path)
        
        # Generate text
        generated = generate_text(model, prompt, stoi, itos, device, max_new_tokens=max_tokens)
        
        return generated
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Example usage
    prompt = "The future of artificial intelligence"
    result = quick_generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
