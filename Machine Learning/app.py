import os
from flask import Flask
from flask_cors import CORS
from LSTMFlask import LSTM_bp  # Import the LSTM blueprint
from Face import face_recognition_bp  # Correct import path for face recognition blueprint

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

# Register the LSTM Blueprint
app.register_blueprint(LSTM_bp, url_prefix='/lstm')
app.register_blueprint(face_recognition_bp, url_prefix='/face')

@app.route('/')
def index():
    return "Welcome to the multi-model detection server!"

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5001)
