from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, Response, stream_with_context
import ollama
import json
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__, template_folder='C:/Users/baboo/Documents/GitHub/pascalai/templates')

# Memory storage for interaction history by device ID
interaction_histories = {}

# Function to get or create a unique device ID
def get_device_id():
    device_id = request.cookies.get('device_id')
    if not device_id:
        device_id = str(uuid.uuid4())
    return device_id

# Function to interact with Ollama's AI model
def get_ai_response(prompt: str, device_id: str):
    try:
        # Retrieve or initialize interaction history for the device ID
        if (device_id not in interaction_histories):
            interaction_histories[device_id] = []

        interaction_history = interaction_histories[device_id]

        # Include the entire interaction history to maintain context
        messages = [{"role": "system", "content": "You are a helpful AI."}]

        # Add past interactions to the context
        for interaction in interaction_history:
            messages.append({"role": "user", "content": interaction["user"]})
            messages.append({"role": "assistant", "content": interaction["ai"]})

        messages.append({"role": "user", "content": prompt})

        # Log the payload being sent
        print("Request Payload:", json.dumps(messages, indent=2))

        # Send the request to Ollama for the AI response
        response = ollama.chat(model="llama3.2:1b", messages=messages)

        # Validate the response
        if response.get("message") and response["message"].get("content"):
            ai_reply = response["message"]["content"]
        else:
            ai_reply = f"Error: Missing 'content' field in response. Full response: {response}"
        
        # Store interaction history
        interaction_history.append({"user": prompt, "ai": ai_reply})
        return ai_reply

    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

@app.route("/")
def home():
    # Serves the homepage (homePage.html)
    return render_template("homePage.html")  

@app.route('/static/<path:path>')
def send_static(path):
    # Serve static files
    return send_from_directory('C:/Users/baboo/Documents/GitHub/pascalai/static', path)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("prompt")
    device_id = get_device_id()
    if user_input:
        print(f"\nUser: {user_input}")  # Print user input to terminal
        response = get_ai_response(user_input, device_id)
        print(f"AI: {response}")  # Print AI response to terminal
        return jsonify({"response": response})
    return jsonify({"response": "No prompt received."})

@app.route("/chat-page")
def chat_page():
    # Serves the AI chat page (index.html)
    return render_template("index.html")

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

def navigateTo(section):
    # ...existing code...
    if section == 'contact':
        url = '/contact'
    # ...existing code...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)