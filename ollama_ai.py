import ollama

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str):
    try:
        # Send a request to the Ollama model and get the response
        response = ollama.chat(model="qwen2.5-coder:3b", messages=[{"role": "user", "content": prompt}])

        # Log the full response for debugging
        print("Full response:", response)

        # Check if the 'message' field is present and contains 'content'
        if "message" in response and "content" in response["message"]:
            ai_reply = response["message"]["content"]
        else:
            # If 'content' is missing, print the whole response for debugging
            ai_reply = f"Error: Missing 'content' field in response. Full response: {response}"
        
        return ai_reply

    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    prompt = input("Enter a prompt for the AI: ")
    response = get_ai_response(prompt)
    print(f"AI Response: {response}")