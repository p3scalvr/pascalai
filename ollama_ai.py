from flask import Flask, request, Response
import ollama
import torch
import json

app = Flask(__name__)

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
        context_window = 1
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        if knowledge_base:
            messages.append({"role": "system", "content": f"Reference information: {knowledge_base}"})

        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model="llama3.2:3b", messages=messages, stream=True)

        def generate():
            ai_reply = ""
            for chunk in response.iter_content(chunk_size=128):
                if chunk:
                    ai_reply += chunk.decode('utf-8')
                    yield f"data: {json.dumps({'text': chunk.decode('utf-8')})}\n\n"
            interaction_history.append({"user": prompt, "ai": ai_reply})

        return Response(generate(), content_type='text/event-stream')

    except json.JSONDecodeError as e:
        return f"data: {json.dumps({'text': 'Error: The response from the server was not valid JSON.'})}\n\n"
    except Exception as e:
        return f"data: {json.dumps({'text': f'Error: {e}'})}\n\n"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    return get_ai_response(prompt)

if __name__ == "__main__":
    app.run(debug=True)