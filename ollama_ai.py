import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json
import time
import requests

# Check if GPU is available and set device
def get_device():
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        if gpu_properties.total_memory >= 4 * 1024 * 1024 * 1024:  # Check if GPU has at least 4GB of memory
            return "cuda"
    return "cpu"

device = get_device()
print(f"Using device: {device}")  # Prints "cuda" if GPU is available and sufficient, otherwise "cpu"

# Memory storage for interaction history by device ID
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

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str, device_id: str):
    try:
        # Retrieve or initialize interaction history for the device ID
        if device_id not in interaction_histories:
            interaction_histories[device_id] = []

        interaction_history = interaction_histories[device_id]

        # Include the entire interaction history to maintain context
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        # Add knowledge base content if available
        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        # Add past interactions to the context
        for interaction in interaction_history:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        # Log the payload being sent
        print("Request Payload:", json.dumps(messages, indent=2))

        # Measure start time
        start_time = time.time()

        # Send the request to Ollama for the AI response
        response = ollama.chat(model="llama3.2:1b", messages=messages)

        # Measure end time
        end_time = time.time()
        print(f"AI response time: {end_time - start_time} seconds")

        # Validate the response
        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = "Error: No valid response content received."

        # Verify the AI response
        verified_reply = verify_response(ai_reply)

        # Check if the response contains important information (e.g., a name) and the user wants it to be remembered
        memory_updated = False
        if "remember" in prompt.lower() or "my name is" in prompt.lower():
            memory_updated = True

        # Store interaction history
        interaction_history.append({"user": prompt, "ai": verified_reply, "memory_updated": memory_updated})
        return verified_reply, memory_updated

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
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response, memory_updated = get_ai_response(prompt, device_id)
        print(f"AI Response: {response}")
        if memory_updated:
            print("*Memory Updated*")