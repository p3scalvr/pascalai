from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, Response, stream_with_context
import ollama
import json
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ollama_ai import get_ai_response  # Import the updated get_ai_response function

app = Flask(__name__, template_folder='C:/Users/baboo/Documents/GitHub/pascalai', static_folder='C:/Users/baboo/Documents/GitHub/pascalai/static')

# Memory storage for interaction history by device ID
interaction_histories = {}

# Function to get or create a unique device ID
def get_device_id():
    device_id = request.cookies.get('device_id')
    if not device_id:
        device_id = str(uuid.uuid4())
    return device_id

@app.route("/")
def home():
    # Serves the homepage (index.html)
    return render_template("index.html")  

@app.route('/static/<path:path>')
def send_static(path):
    # Serve static files
    return send_from_directory(app.static_folder, path)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("prompt")
    device_id = get_device_id()
    if user_input:
        print(f"\nUser: {user_input}")  # Print user input to terminal
        response, memory_updated = get_ai_response(user_input, device_id)
        print(f"AI: {response}")  # Print AI response to terminal
        response = jsonify({"response": response, "memory_updated": memory_updated})
        response.set_cookie('device_id', device_id, max_age=60*60*24*30)  # Set cookie to expire in 30 days
        return response
    return jsonify({"response": "No prompt received."})

@app.route("/ai-page")
def ai_page():
    # Serves the AI chat page (ai.html)
    return render_template("ai.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    if not name or not email or not message:
        return jsonify({'success': False, 'error': 'Missing required fields'})

    try:
        sender_email = "your-email@gmail.com"
        receiver_email = "contact.pascalai@gmail.com"
        password = "your-email-password"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"New Contact Form Submission from {name}"

        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

        return jsonify({'success': True})
    except smtplib.SMTPException as e:
        print(f"SMTP error occurred: {e}")
        return jsonify({'success': False, 'error': 'SMTP error occurred. Please try again later.'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'success': False, 'error': 'An unexpected error occurred. Please try again later.'})

@app.route('/check-mic-permission', methods=['GET'])
def check_mic_permission():
    return jsonify({"permission": "granted"})

@app.route('/gpu-utilization', methods=['GET'])
def gpu_utilization():
    try:
        # Simulate GPU utilization data
        gpu_utilization = 50  # Example value
        return jsonify({"gpu_utilization": gpu_utilization})
    except Exception as e:
        print(f"An error occurred while fetching GPU utilization: {e}")
        return jsonify({"error": "An error occurred while fetching GPU utilization."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)