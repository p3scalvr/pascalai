import ollama
import torch  # PyTorch library, if Ollama internally uses PyTorch/TensorFlow

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Prints "cuda" if GPU is available

# Memory storage for interaction history
interaction_history = []

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str):
    try:
        # Only include a short context or recent interactions to reduce overhead
        context_window = 2  # Adjust based on desired history depth
        messages = [{"role": "system", "content": "You are a helpful AI."}]
        
        # Add past interactions to the context
        for interaction in interaction_history[-context_window:]:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})
        
        messages.append({"role": "user", "content": prompt})

        # Send the request to Ollama for the AI response (assuming it uses GPU if available)
        response = ollama.chat(model="llama3.2", messages=messages)

        ai_reply = response.get("message", {}).get("content", "Error: No valid response received.")
        
        # Store interaction history
        interaction_history.append({"user": prompt, "ai": ai_reply})
        
        return ai_reply

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt for the AI (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response = get_ai_response(prompt)
        print(f"AI Response: {response}")