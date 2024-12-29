import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json
import time
import requests
from flask import request
from datetime import datetime
import pytz

interaction_histories = {
    'chats': {},
    'memories': {}
}

def load_knowledge_base(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Knowledge base file '{file_path}' not found.")
        return ""

knowledge_base = load_knowledge_base("knowledge_base.txt")


def verify_response(response: str):
    try:
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{response}"
        result = requests.get(search_url).json()
        if "extract" in result:
            return result["extract"]
        else:
            return response
    except Exception as e:
        print(f"Verification error: {e}")
        return response

def get_current_datetime():
    user_timezone = pytz.timezone('America/New_York') 
    now = datetime.now(user_timezone)
    return now.strftime("%A, %B %d, %Y %I:%M:%S %p")

def check_easter_eggs(prompt: str):
    easter_eggs = {
        "what day is it": f"Today is {get_current_datetime().split(',')[0]}.",
        "what time is it": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}.",
        "what is the time": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}.",
        "what's the time": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}."
    }
    for key, value in easter_eggs.items():
        if key.lower() in prompt.lower():
            return value, False, None  
    return None

# Function to get or create a new chat ID
def get_or_create_chat_id(device_id: str):
    if device_id not in interaction_histories['chats']:
        interaction_histories['chats'][device_id] = {}
    chat_id = f"Chat {len(interaction_histories['chats'][device_id]) + 1}"
    interaction_histories['chats'][device_id][chat_id] = []
    return chat_id

# Function to get the current interaction history
def get_interaction_history(device_id: str, chat_id: str):
    if device_id in interaction_histories['chats'] and chat_id in interaction_histories['chats'][device_id]:
        return interaction_histories['chats'][device_id][chat_id]
    return []

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str, device_id: str, chat_id: str = None):
    try:
        # Initialize memory_updated at the start
        memory_updated = False
        
        selected_model = request.headers.get('X-Selected-Model')
        
        # Handle image analysis for Llava-Llama 3
        if prompt.startswith("Analyze image:") and selected_model == 'llava-llama3:8b':
            image_path = prompt.split(": ")[1]
            with open(image_path, "rb") as image_file:
                response = ollama.chat(
                    model='llava-llama3:8b',
                    messages=[{
                        "role": "user",
                        "content": "Analyze this image and describe what you see:",
                        "images": [image_file.read()]
                    }],
                    stream=False
                )
                return response["message"]["content"], False, chat_id

        # Get the selected model from request headers
        selected_model = request.headers.get('X-Selected-Model')
        
        # If not in headers, try to get from request body
        if not selected_model and request.is_json:
            selected_model = request.json.get('model')
        
        # Default to llama3.2:1b if still no model specified
        if not selected_model:
            selected_model = 'llama3.2:1b'
            
        print(f"Using AI Model: {selected_model}")  # Debug log

        # If model changed during chat, force new chat creation
        if chat_id and 'previous_model' in request.headers:
            previous_model = request.headers.get('previous_model')
            if previous_model != selected_model:
                chat_id = None  # This will trigger creation of new chat

        # Check for easter eggs first
        easter_egg_response = check_easter_eggs(prompt)
        if easter_egg_response:
            return easter_egg_response

        # Retrieve or initialize interaction history for the device ID and chat ID
        if device_id not in interaction_histories['chats']:
            interaction_histories['chats'][device_id] = {}
        if not chat_id:
            chat_id = get_or_create_chat_id(device_id)
        if chat_id not in interaction_histories['chats'][device_id]:
            interaction_histories['chats'][device_id][chat_id] = []

        interaction_history = interaction_histories['chats'][device_id][chat_id]

        # Add response length guidance to system message
        messages = [
            {"role": "system", "content": "You are a helpful AI. Keep responses concise and under 2 sentences when possible. Only provide detailed responses when explicitly asked."}
        ]

        # Analyze prompt complexity
        is_complex_query = any(word in prompt.lower() for word in 
            ['explain', 'detail', 'elaborate', 'how to', 'describe', 'what is'])

        # Optimize system messages for performance
        messages = [
            {"role": "system", "content": "You are a helpful AI. Be concise."}
        ]

        # Only add length hint for simple queries
        if not is_complex_query:
            messages.append({
                "role": "system", 
                "content": "Keep responses under 50 words."
            })

        # Continue with existing message history logic
        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        # Only include last 5 interactions to reduce context
        for interaction in interaction_history[-5:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        # Measure start time
        start_time = time.time()

        # Get the selected model from request headers or localStorage
        selected_model = request.headers.get('X-Selected-Model')
        
        # If no model specified in headers, try to get from request JSON
        if not selected_model and request.is_json:
            selected_model = request.json.get('model')
        
        # Default to gemma2:2b if no model specified
        if not selected_model:
            selected_model = 'gemma2:2b'
            
        print(f"Using AI Model: {selected_model}")  # Log the model being used
        
        # Performance optimization settings
        response = ollama.chat(
            model=selected_model,
            messages=messages,
            options={
                "num_predict": 512 if is_complex_query else 256,  # Increased from 30/75
                "temperature": 0.7,
                "top_k": 40,  # Increased from 20
                "top_p": 0.9,  # Increased from 0.7
                "num_ctx": 4096,  # Increased context window
                "num_thread": 4,
                "repeat_penalty": 1.1,
                "num_gpu": 1,
                "seed": 42,
                "batch_size": 8,
                "mirostat": 0,  # Disabled adaptive sampling for more consistent completions
                "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]  # Updated stop tokens
            }
        )

        # Measure end time
        end_time = time.time()
        print(f"AI response time: {end_time - start_time:.2f} seconds")

        # Validate the response
        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = "Error: No valid response content received."

        # Verify the AI response
        verified_reply = verify_response(ai_reply)

        # Check for memory-related keywords in prompt
        memory_keywords = ["my name is", "remember", "don't forget", "i like", "i am", "i'm"]
        should_remember = any(keyword in prompt.lower() for keyword in memory_keywords)

        # Initialize device memories if not exists
        if device_id not in interaction_histories['memories']:
            interaction_histories['memories'][device_id] = {}

        # Store memory if needed
        if should_remember:
            # Extract the relevant information from the prompt
            memory_content = prompt.lower()
            for keyword in memory_keywords:
                if keyword in memory_content:
                    memory_value = memory_content.split(keyword)[1].strip()
                    interaction_histories['memories'][device_id][keyword] = memory_value
                    memory_updated = True
                    print(f"Memory stored for {device_id}: {keyword} -> {memory_value}")

        # Include memories in context for responses
        if device_id in interaction_histories['memories']:
            memories = interaction_histories['memories'][device_id]
            if memories:
                memory_context = "Previous information: " + ", ".join([f"{k}: {v}" for k, v in memories.items()])
                messages.append({"role": "system", "content": memory_context})

        # Store interaction history
        interaction_history.append({"user": prompt, "ai": verified_reply, "memory_updated": memory_updated})
        return verified_reply, memory_updated, chat_id

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error: The response from the server was not valid JSON.", False
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}", False, chat_id

# Example usage
if __name__ == "__main__":
    while True:
        device_id = input("Enter your device ID: ")
        chat_id = input("Enter your chat ID (or leave blank to create a new chat): ")
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response, memory_updated, chat_id = get_ai_response(prompt, device_id, chat_id)
        print(f"AI Response: {response}")
        if memory_updated:
            print("*Memory Updated*")
        print(f"Chat ID: {chat_id}")