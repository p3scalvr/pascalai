import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json
import time
import requests
from flask import request
from datetime import datetime
import pytz

# Memory storage for interaction history by device ID and chat ID
interaction_histories = {}

# Load external knowledge base from a text file
def load_knowledge_base(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Knowledge base file '{file_path}' not found.")
        return ""

# Initialize knowledge base
knowledge_base = load_knowledge_base("knowledge_base.txt")

# Remove YouTube API configuration and related functions

# Function to verify AI response using an external API (e.g., Wikipedia API)
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

# Function to get the current date and time based on the user's timezone
def get_current_datetime():
    user_timezone = pytz.timezone('America/New_York')  # You can adjust the timezone as needed
    now = datetime.now(user_timezone)
    return now.strftime("%A, %B %d, %Y %I:%M:%S %p")

# Function to check for easter eggs
def check_easter_eggs(prompt: str):
    easter_eggs = {
        "what day is it": f"Today is {get_current_datetime().split(',')[0]}.",
        "what time is it": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}.",
        "what is the time": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}.",
        "what's the time": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}."
    }
    for key, value in easter_eggs.items():
        if key.lower() in prompt.lower():
            return value, False, None  # Return tuple of (response, memory_updated, chat_id)
    return None

# Function to get or create a new chat ID
def get_or_create_chat_id(device_id: str):
    if device_id not in interaction_histories:
        interaction_histories[device_id] = {}
    chat_id = f"Chat {len(interaction_histories[device_id]) + 1}"
    interaction_histories[device_id][chat_id] = []
    return chat_id

# Function to get the current interaction history
def get_interaction_history(device_id: str, chat_id: str):
    if device_id in interaction_histories and chat_id in interaction_histories[device_id]:
        return interaction_histories[device_id][chat_id]
    return []

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str, device_id: str, chat_id: str = None):
    try:
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
        if device_id not in interaction_histories:
            interaction_histories[device_id] = {}
        if not chat_id:
            chat_id = get_or_create_chat_id(device_id)
        if chat_id not in interaction_histories[device_id]:
            interaction_histories[device_id][chat_id] = []

        interaction_history = interaction_histories[device_id][chat_id]

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

        # Check if the response contains important information (e.g., a name) and the user wants it to be remembered
        memory_updated = False
        if "remember" in prompt.lower() or "my name is" in prompt.lower() or "don't forget" in prompt.lower() or "i like" in prompt.lower():
            memory_updated = True

        # Store interaction history
        interaction_history.append({"user": prompt, "ai": verified_reply, "memory_updated": memory_updated})
        return verified_reply, memory_updated, chat_id

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error: The response from the server was not valid JSON.", False
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}", False

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