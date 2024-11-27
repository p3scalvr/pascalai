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
        context_window = 1  # Adjust based on desired history depth
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        print("Request Payload:", json.dumps(messages, indent=2))

        response = ollama.chat(model="llama3.2:3b", messages=messages, stream=True)

        ai_reply = ""
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                ai_reply += chunk.decode('utf-8')
                yield ai_reply

        interaction_history.append({"user": prompt, "ai": ai_reply})
        print(f"User: {prompt}")
        print(f"AI: {ai_reply}")

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        yield "Error: The response from the server was not valid JSON."
    except Exception as e:
        print(f"An error occurred: {e}")
        yield f"Error: {e}"

# Example usage
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response = get_ai_response(prompt)
        print(f"AI Response: {response}")