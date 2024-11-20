import ollama
import torch
import json
import os
import asyncio

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to save interaction history
history_file_path = "interaction_history.json"

# Load interaction history from a file
def load_interaction_history():
    if os.path.exists(history_file_path):
        try:
            with open(history_file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print("Error decoding the history file. Starting fresh.")
            return []
    return []

# Save interaction history to a file
def save_interaction_history(history):
    with open(history_file_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

# Load history into memory
interaction_history = load_interaction_history()

# Load and cache the knowledge base
def load_knowledge_base(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Knowledge base file '{file_path}' not found.")
        return ""

knowledge_base = load_knowledge_base("knowledge_base.txt")

# Function to interact with Ollama's AI model
async def get_ai_response(prompt: str):
    try:
        # Include past interactions and knowledge base
        context_window = 5  # Number of past interactions to include
        messages = [{"role": "system", "content": "You are PascalGPT, a helpful and conversational AI."}]

        # Add knowledge base if available
        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        # Add recent interaction history
        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        # Add the new user input
        messages.append({"role": "user", "content": prompt})

        # Send request to Ollama asynchronously
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: ollama.chat(model="llama3.2", messages=messages, device=device)
        )

        # Validate the response
        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = "Error: No valid response content received."

        # Store the new interaction in memory
        interaction_history.append({"user": prompt, "ai": ai_reply})

        # Save the updated history to the file
        save_interaction_history(interaction_history)

        return ai_reply

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error: The response from the server was not valid JSON."
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"

# Main interactive loop
async def main():
    print("Welcome to PascalGPT! Your conversations will be remembered.")
    while True:
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        response = await get_ai_response(prompt)
        print(f"PascalGPT: {response}")

if __name__ == "__main__":
    asyncio.run(main())