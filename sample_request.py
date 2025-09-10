#!/usr/bin/env python3
"""
Sample script to interact with local Ollama Gemma server
This script demonstrates how to use the Ollama API with your local Gemma models
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:3040"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return {}
    
    def generate_text(self, 
                     model: str, 
                     prompt: str, 
                     stream: bool = False,
                     options: Optional[Dict] = None) -> str:
        """Generate text using the specified model"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json().get("response", "")
                
        except requests.exceptions.RequestException as e:
            print(f"Error generating text: {e}")
            return ""
    
    def _handle_stream_response(self, response) -> str:
        """Handle streaming response"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        print(data['response'], end='', flush=True)
                        full_response += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        print()  # New line after streaming
        return full_response
    
    def chat(self, 
             model: str, 
             messages: list, 
             stream: bool = False,
             options: Optional[Dict] = None) -> str:
        """Chat with the model using conversation format"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json().get("message", {}).get("content", "")
                
        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return ""

def main():
    """Main function demonstrating various API usage patterns"""
    
    # Initialize client
    client = OllamaClient()
    
    print("🤖 Ollama Gemma Client Demo")
    print("=" * 50)
    
    # 1. List available models
    print("\n📋 Available Models:")
    models = client.list_models()
    if models and "models" in models:
        for model in models["models"]:
            print(f"  - {model['name']} ({model.get('size', 'Unknown size')})")
    else:
        print("  No models found or server not responding")
        return
    
    # 2. Simple text generation
    print("\n💬 Simple Text Generation:")
    print("-" * 30)
    
    prompt = "Explain what machine learning is in simple terms."
    print(f"Prompt: {prompt}")
    print("Response:")
    
    response = client.generate_text(
        model="gemma2:2b",
        prompt=prompt,
        stream=True  # Enable streaming for real-time output
    )
    
    # 3. Text generation with options
    print("\n⚙️  Text Generation with Options:")
    print("-" * 40)
    
    options = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100
    }
    
    prompt2 = "Write a short poem about artificial intelligence."
    print(f"Prompt: {prompt2}")
    print("Response:")
    
    response2 = client.generate_text(
        model="gemma2:2b",
        prompt=prompt2,
        options=options,
        stream=True
    )
    
    # 4. Chat conversation
    print("\n💭 Chat Conversation:")
    print("-" * 25)
    
    messages = [
        {"role": "user", "content": "Hello! What's your name?"},
        {"role": "assistant", "content": "Hello! I'm Gemma, an AI assistant created by Google."},
        {"role": "user", "content": "Can you help me understand transformers in machine learning?"}
    ]
    
    print("Conversation:")
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        print(f"{role}: {msg['content']}")
    
    print("\nAssistant response:")
    chat_response = client.chat(
        model="gemma2:2b",
        messages=messages,
        stream=True
    )
    
    # 5. Non-streaming example
    print("\n📝 Non-Streaming Example:")
    print("-" * 30)
    
    prompt3 = "What are the benefits of using Python for data science?"
    print(f"Prompt: {prompt3}")
    
    response3 = client.generate_text(
        model="gemma2:2b",
        prompt=prompt3,
        stream=False  # Non-streaming
    )
    
    print(f"Response: {response3}")
    
    # 6. Interactive mode
    print("\n🎮 Interactive Mode (type 'quit' to exit):")
    print("-" * 45)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if user_input:
                print("Gemma: ", end="")
                client.generate_text(
                    model="gemma2:2b",
                    prompt=user_input,
                    stream=True
                )
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
