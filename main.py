from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import json
import uuid
import os
from ollama_ai import get_ai_response

app = Flask(__name__, 
    template_folder='C:/Users/z/Documents/GitHub/pascalai', 
    static_folder='C:/Users/z/Documents/GitHub/pascalai/static'
)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_device_id():
    device_id = request.cookies.get('device_id')
    if not device_id:
        device_id = str(uuid.uuid4())
    return device_id

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def home(path):
    device_id = get_device_id()
    response = make_response(render_template("ai.html"))
    response.set_cookie('device_id', device_id, max_age=60*60*24*30)
    return response

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/chat", methods=["POST"])
def chat():
    device_id = get_device_id()
    chat_id = request.json.get("chat_id") if request.is_json else None

    try:
        if request.is_json:
            user_input = request.json.get("prompt")
            if user_input:
                response = get_ai_response(user_input, device_id, chat_id)
                
                if isinstance(response, tuple):
                    return jsonify({
                        "response": response[0],
                        "memory_updated": response[1],
                        "chat_id": response[2] or chat_id
                    })
                
                return jsonify({
                    "response": response,
                    "memory_updated": False,
                    "chat_id": chat_id
                })

        return jsonify({"response": "No input received."})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"response": "An error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)