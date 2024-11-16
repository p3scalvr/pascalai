from flask import Flask, render_template, request, jsonify, send_from_directory
import ollama

app = Flask(__name__)

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str):
    try:
        # Send a request to the Ollama model and get the response
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

        # Log the full response for debugging
        print("Full response:", response)

        # Check if the 'message' field is present and contains 'content'
        if "message" in response and "content" in response["message"]:
            ai_reply = response["message"]["content"]
        else:
            ai_reply = f"Error: Missing 'content' field in response. Full response: {response}"
        
        return ai_reply

    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

@app.route("/")
def home():
    return render_template("index.html")  

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/templates")
def templates():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("prompt")
    if user_input:
        ai_response = get_ai_response(user_input)
        return jsonify({"response": ai_response})
    else:
        return jsonify({"response": "No prompt received."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)