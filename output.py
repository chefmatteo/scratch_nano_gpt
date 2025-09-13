#!/usr/bin/env python3
"""
Text Generation Script using Trained GPT Model
Loads a saved model and generates text based on user input
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the model classes from models.bigram_v2
from models.bigram_v2 import BigramLanguageModel, Head, MultiHeadAttention, FeedForward, Block

def load_model(model_path='trained_gpt_model.pth'):
    """Load the trained model and vocabulary"""
    print(f"Loading model from {model_path}...")
    
    # Load the saved data
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract configuration and vocabulary
    config = checkpoint['model_config']
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Create model with saved configuration
    model = BigramLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully on {device}")
    print(f"Model configuration: {config}")
    
    return model, chars, stoi, itos, device

def encode(text, stoi):
    """Encode text to token indices"""
    return [stoi[c] for c in text]

def decode(tokens, itos):
    """Decode token indices to text"""
    return ''.join([itos[i] for i in tokens])

def generate_text(model, prompt, stoi, itos, device, max_new_tokens=500, temperature=1.0):
    """Generate text from a prompt"""
    # Encode the prompt
    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
    
    # Generate new tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = model(context)
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            context = torch.cat((context, idx_next), dim=1)
    
    # Decode and return
    generated_text = decode(context[0].tolist(), itos)
    return generated_text

def interactive_generation():
    """Interactive text generation"""
    print("🤖 GPT Text Generator")
    print("=" * 50)
    
    # Load the model
    try:
        model, chars, stoi, itos, device = load_model()
    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first by running bigram_v2.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n💬 Interactive Text Generation")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            prompt = input("\n📝 Your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                print("Please enter a prompt!")
                continue
            
            # Generate text
            print("🔄 Generating...")
            generated = generate_text(model, prompt, stoi, itos, device, max_new_tokens=300)
            
            # Display result
            print(f"\n🤖 Generated text:")
            print("-" * 30)
            print(generated)
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_generation():
    """Generate text for multiple prompts"""
    print("🤖 Batch Text Generation")
    print("=" * 50)
    
    # Load the model
    try:
        model, chars, stoi, itos, device = load_model()
    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first by running bigram_v2.py")
        return
    
    # Example prompts
    prompts = [
        "To be or not to be",
        "Once upon a time",
        "The future of AI is",
        "In a galaxy far away",
        "The meaning of life is"
    ]
    
    print(f"Generating text for {len(prompts)} prompts...")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        generated = generate_text(model, prompt, stoi, itos, device, max_new_tokens=200)
        print(f"Generated: {generated}")
        print("-" * 30)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_generation()
    else:
        interactive_generation()
