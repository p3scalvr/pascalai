import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json
import time

# Check if GPU is available and set device
def get_device():
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        if gpu_properties.total_memory >= 4 * 1024 * 1024 * 1024:  # Check if GPU has at least 4GB of memory
            return "cuda"
    return "cpu"

device = get_device()
print(f"Using device: {device}")  # Prints "cuda" if GPU is available and sufficient, otherwise "cpu"

# Memory storage for interaction history
interaction_history = []

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

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str):
    try:
        # Include the entire interaction history to maintain context
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        # Add knowledge base content if available
        if (knowledge_base):
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
        response = ollama.chat(model="llama3.2:1b", messages=messages, device=device)

        # Measure end time
        end_time = time.time()
        print(f"AI response time: {end_time - start_time} seconds")

        # Validate the response
        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = "Error: No valid response content received."

        # Store interaction history
        interaction_history.append({"user": prompt, "ai": ai_reply})
        return ai_reply

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error: The response from the server was not valid JSON."
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response = get_ai_response(prompt)
        print(f"AI Response: {response}")