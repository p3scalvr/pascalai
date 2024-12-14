import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json
import time
import requests
from datetime import datetime
import pytz
from googleapiclient.discovery import build

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

# Add YouTube API configuration
YOUTUBE_API_KEY = 'YOUR_YOUTUBE_API_KEY'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def search_youtube_video(query):
    try:
        request = youtube.search().list(
            part="snippet",
            maxResults=1,
            q=query,
            type="video"
        )
        response = request.execute()
        
        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            title = response['items'][0]['snippet']['title']
            return {
                'videoId': video_id,
                'title': title,
                'url': f"https://www.youtube.com/watch?v={video_id}"
            }
        return None
    except Exception as e:
        print(f"YouTube search error: {e}")
        return None

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
    user_timezone = pytz.timezone('America/New_York')  # Replace with the user's actual timezone
    now = datetime.now(user_timezone)
    return now.strftime("%A, %B %d, %Y %I:%M:%S %p")

# Function to check for easter eggs
def check_easter_eggs(prompt: str):
    easter_eggs = {
        "who is pascal": "Pascal is a very hot man",
        "who is joe biden": "'J' This is a very common question Pascal asked PascalAI for debugging and testing",
        "what day is it": f"Today is {get_current_datetime().split(',')[0]}, {get_current_datetime().split(',')[1].strip()}.",
        "what time is it": f"The current time is {get_current_datetime().split()[-2]} {get_current_datetime().split()[-1]}."
    }
    for key, value in easter_eggs.items():
        if key in prompt.lower():
            return value
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
        # Check for video-related keywords
        video_keywords = ['video', 'youtube', 'watch', 'show me', 'play']
        is_video_request = any(keyword in prompt.lower() for keyword in video_keywords)
        
        if is_video_request:
            # Extract search query from prompt
            search_query = prompt.lower()
            for keyword in video_keywords:
                search_query = search_query.replace(keyword, '').strip()
            
            # Search for video
            video_result = search_youtube_video(search_query)
            if video_result:
                response = f"I found this video for you: {video_result['title']}\n{video_result['url']}\n[VIDEO_EMBED:{video_result['videoId']}]"
                return response, False, chat_id

        # Check for easter eggs
        easter_egg_response = check_easter_eggs(prompt)
        if easter_egg_response:
            return easter_egg_response, False

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

        # Add length hint based on query type
        if not is_complex_query:
            messages.append({
                "role": "system", 
                "content": "Provide a single short sentence response unless specifically asked for more detail."
            })

        # Continue with existing message history logic
        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        # Only include last 3 interactions to reduce context
        for interaction in interaction_history[-10:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        # Measure start time
        start_time = time.time()

        # Send request to Ollama with optimized parameters for faster generation
        response = ollama.chat(
            model="gemma2:2b",
            messages=messages,
            options={
                "num_predict": 50 if not is_complex_query else 100,  # Reduce token limit
                "temperature": 0.5,  # Further reduce randomness
                "top_k": 30,  # Further reduce to focus on more likely tokens
                "top_p": 0.8  # Further reduce to focus on more likely completions
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