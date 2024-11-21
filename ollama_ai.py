import ollama
import torch  
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}") 

interaction_history = []

def load_knowledge_base(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Knowledge base file '{file_path}' not found.")
        return ""

knowledge_base = load_knowledge_base("knowledge_base.txt")

def get_ai_response(prompt: str):
    try:
        context_window = 1  
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        print("Request Payload:", json.dumps(messages, indent=2))

        response = ollama.chat(model="gemma:7b", messages=messages)

        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = "Error: No valid response content received."

        interaction_history.append({"user": prompt, "ai": ai_reply})
        return ai_reply

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error: The response from the server was not valid JSON."
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response = get_ai_response(prompt)
        print(f"AI Response: {response}")