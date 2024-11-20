import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow
import json

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Prints "cuda" if GPU is available

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
        # Only include a short context or recent interactions to reduce overhead
        context_window = 2  # Adjust based on desired history depth
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        # Add knowledge base content if available
        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        # Add past interactions to the context
        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        # Log the payload being sent
        print("Request Payload:", json.dumps(messages, indent=2))

        # Send the request to Ollama for the AI response
        response = ollama.chat(model="llama3.2", messages=messages)

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