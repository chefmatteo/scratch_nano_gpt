#!/usr/bin/env python3
"""
Quick test script for Ollama Gemma server
Simple example to test your local Gemma setup
"""

import requests
import json

def test_ollama_connection():
    """Test connection to Ollama server"""
    try:
        response = requests.get("http://127.0.0.1:3040/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Connected to Ollama server!")
            print("📋 Available models:")
            for model in models.get("models", []):
                print(f"  - {model['name']}")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server on port 3040")
        print("💡 Make sure Ollama is running: ./ollama_wrapper.sh serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def generate_text(prompt: str, model: str = "gemma2:2b"):
    """Generate text using the specified model"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://127.0.0.1:3040/api/generate",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {e}"

def main():
    print("🚀 Quick Ollama Gemma Test")
    print("=" * 30)
    
    # Test connection
    if not test_ollama_connection():
        return
    
    # Test text generation
    print("\n💬 Testing text generation...")
    prompt = "Hello! Can you tell me a fun fact about machine learning?"
    print(f"Prompt: {prompt}")
    
    response = generate_text(prompt)
    print(f"Response: {response}")
    
    # Interactive test
    print("\n🎮 Interactive test (type 'quit' to exit):")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                response = generate_text(user_input)
                print(f"Gemma: {response}")
                
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
