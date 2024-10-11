import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from flask import Blueprint, jsonify

# Initialize the Blueprint
LSTM_bp = Blueprint('LSTM', __name__)

# Load the model and scalers (ensure the paths are correct)
model = load_model('crime_lstm_model.h5')
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Load and preprocess the original data to extract scaling and encoding
data = pd.read_csv('LSTM/CrimeLSTM.csv')
data = data.drop(columns=['Town', 'Location', 'EndDate', 'EndTime', 'Crime'])
data['StartTime'] = data['StartTime'].fillna('00:00')
data['StartDateTime'] = pd.to_datetime(data['StartDate'] + ' ' + data['StartTime'], format='%Y-%m-%d %H:%M')
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['Minute'] = data['StartDateTime'].dt.minute
data = data.drop(columns=['StartDate', 'StartTime'])
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])
scaler.fit(data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

seq_length = 10  # Define sequence length

# Helper function to create future data
def create_future_data():
    tomorrow = datetime.now() + timedelta(days=1)
    unique_locations = data[['Longitude', 'Latitude']].drop_duplicates()

    future_data = unique_locations.copy()
    future_data['Year'] = tomorrow.year
    future_data['Month'] = tomorrow.month
    future_data['Day'] = tomorrow.day
    future_data['Hour'] = np.random.randint(0, 24, size=unique_locations.shape[0])
    future_data['Minute'] = np.random.randint(0, 60, size=unique_locations.shape[0])

    return future_data

# Helper function to pad sequences
def pad_sequences(X, seq_length):
    num_sequences = (len(X) + seq_length - 1) // seq_length
    pad_size = num_sequences * seq_length - len(X)
    if pad_size > 0:
        X_padded = np.concatenate([X, np.zeros((pad_size, X.shape[1]))], axis=0)
    else:
        X_padded = X
    return X_padded

# Function to calculate distance using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371e3  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in meters

# Function to convert hour to time range
def get_time_range(hour):
    if 0 <= hour < 6:
        return "00:00 - 06:00"
    elif 6 <= hour < 12:
        return "06:00 - 12:00"
    elif 12 <= hour < 18:
        return "12:00 - 18:00"
    else:
        return "18:00 - 24:00"

# Function to predict future crimes
def predict_future_crimes(future_data, center_long, center_lat, radius):
    # Normalize future data
    normalized_data = scaler.transform(future_data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

    # Prepare sequences
    X_future_padded = pad_sequences(normalized_data, seq_length)
    num_sequences = X_future_padded.shape[0] // seq_length
    X_future_seq = X_future_padded[:num_sequences * seq_length].reshape(num_sequences, seq_length, -1)

    # Predict on sequences
    probabilities = model.predict(X_future_seq)
    predicted_classes = np.argmax(probabilities, axis=-1)
    predicted_probabilities = np.max(probabilities, axis=-1)

    # Denormalize predictions
    denormalized_data = scaler.inverse_transform(normalized_data)
    future_data[['Longitude', 'Latitude']] = denormalized_data[:, :2]

    # Map predictions back to future_data length
    future_data['Predicted CrimeType'] = np.repeat(label_encoder.inverse_transform(predicted_classes), seq_length)[:len(future_data)]
    future_data['Probability'] = np.repeat(predicted_probabilities, seq_length)[:len(future_data)]

    # Filter locations within the specified radius
    future_data['Distance'] = haversine(center_long, center_lat, future_data['Longitude'], future_data['Latitude'])
    filtered_data = future_data[future_data['Distance'] <= radius]

    # Convert hours to time ranges
    filtered_data['Predicted Time'] = filtered_data['Hour'].apply(get_time_range)

    return filtered_data

# Define an API route to trigger predictions
@LSTM_bp.route('/predict', methods=['GET'])
def predict():
    # Define the center point and radius
    center_longitude = 80.6219875521929
    center_latitude = 7.322798069315021
    radius = 3651  # Radius in meters

    future_data = create_future_data()
    predicted_future_data = predict_future_crimes(future_data, center_longitude, center_latitude, radius)

    # Get top 10 locations with highest probability
    top_10_locations = predicted_future_data.nlargest(6, 'Probability')

    # Convert to JSON format to return as API response
    response = top_10_locations.to_dict(orient='records')
    return jsonify(response)
