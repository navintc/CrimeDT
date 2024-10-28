import os
import uuid
import secrets
from flask import Flask, redirect, session, url_for, request, jsonify
from flask_cors import CORS
from LSTMFlask import LSTM_bp  # Import the LSTM blueprint
from Face import face_recognition_bp  # Correct import path for face recognition blueprint
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash
import boto3
from ensemble import ensemble_bp

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

token_store = {}
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

app.secret_key =  os.getenv("APP_SECRET_KEY")
print(app.secret_key)

if not app.secret_key:
    raise ValueError("No secret key set for Flask application")

app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True if using HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax"  # Try "None" if testing across different domains/IPs
)

# Auth0 Configuration
AUTH0_CLIENT_ID = os.getenv('AUTH0_CLIENT_ID')
AUTH0_CLIENT_SECRET = os.getenv('AUTH0_CLIENT_SECRET')
AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN')
AUTH0_CALLBACK_URL = 'http://127.0.0.1:5001/callback'
AUTH0_AUDIENCE = 'https://YOUR_AUTH0_DOMAIN/userinfo'

oauth = OAuth(app)

auth0 = oauth.register(
    'auth0',
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    api_base_url=f'https://{AUTH0_DOMAIN}',
    access_token_url=f'https://{AUTH0_DOMAIN}/oauth/token',
    authorize_url=f'https://{AUTH0_DOMAIN}/authorize',
    client_kwargs={
        'scope': 'openid profile email',
    },
    server_metadata_url=f'https://{os.getenv("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)

# Register the LSTM Blueprint
app.register_blueprint(LSTM_bp, url_prefix='/lstm')
app.register_blueprint(face_recognition_bp, url_prefix='/face')
app.register_blueprint(ensemble_bp, url_prefix='/ensemble')

@app.route('/')
def index():
    return "Welcome to the multi-model detection server!"


# AWS login ---------------------------------------

# Set up DynamoDB connection using Boto3
dynamodb = boto3.resource('dynamodb', region_name='us-east-1',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

# Replace 'your-table-name' with the name of your DynamoDB table
table = dynamodb.Table('crimedt')

@app.route('/register', methods=['POST'])
def register():
    # Endpoint to register a new user
    data = request.json
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    client = data.get('client')
    contact = data.get('contact')
    position = data.get('position')
    role = data.get('role')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    hashed_password = generate_password_hash(password)

    user_id = str(uuid.uuid4())
    try:
        # Add the new user to the DynamoDB table
        table.put_item(Item={
            'id': user_id,
            'email': email,
            'password': hashed_password,
            'first_name': first_name,
            'last_name': last_name,
            'client': client,
            'contact': contact,
            'position': position,
            'role': role
        })
        return jsonify({'message': 'User registered successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/internallogin', methods=['POST'])
def internallogin():
    # Endpoint to login a user
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    try:
        # Query DynamoDB for the user
        response = table.get_item(Key={'email': email})
        user = response.get('Item')

        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        # Check if the password matches
        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid email or password'}), 401

        # Set the user in the session if login is successful
        session['user'] = {
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'client': user['client'],
            'contact': user['contact'],
            'position': user['position']
        }

        return jsonify({'message': 'Login successful', 'user': session['user']}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/internallogout', methods=['POST'])
def internallogout():
    # Clear the user session
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200

# AWS Login end  ------------------------



# AUTH0 ---------------------
@app.route('/login')
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )
    # return "Hey baby"

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token

    # Generate a unique random token for the user and store it in the session
    user_token = secrets.token_hex(32)  # Generates a 64-character hex token
    session["user_token"] = user_token

    # Store the token in the in-memory store
    token_store[user_token] = session["user"]

    print(token_store)

    return jsonify({'message': 'Login successful', 'user_token': user_token})

@app.route('/dashboard')
def dashboard():
    return jsonify(session['user'])

@app.route('/check_token', methods=['POST'])
def check_token():
    data = request.json
    print(data)
    token = data.get('token')

    if not token:
        return jsonify({'error': 'Token is required'}), 400

    # Check if the given token is in the store
    if token in token_store:
        return jsonify({'message': 'Token is valid', 'user_data': token_store[token]}), 200
    else:
        return jsonify({'message': 'Invalid token'}), 404

@app.route('/logout')
def logout():
    # Clear the user session
    token = session.get("user_token")

    # Remove the token from the in-memory store if it exists
    if token and token in token_store:
        del token_store[token]

    print(token_store)

    session.clear()

    return redirect(auth0.api_base_url + '/v2/logout?returnTo=' + url_for('index', _external=True))

# AUTH0 end
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5001)
