import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import torch
import torch.nn as nn

# Define the Ensemble blueprint
ensemble_bp = Blueprint('ensemble', __name__)

# Define DQN architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the dataset
data = pd.read_csv('CrimeLSTM.csv')
label_encoder = LabelEncoder()
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])

# Load the scaler and DQN model
scaler = StandardScaler()
input_dim = 7  # Adjust based on your data
output_dim = 17  # Adjust based on your number of crime types
dqn_model = DQN(input_dim, output_dim)
dqn_model.load_state_dict(torch.load('dqn_model.pth'))  # Load the DQN model
dqn_model.eval()

# Load other models
knn_model = joblib.load('best_knn_model.joblib')
rf_model = joblib.load('best_random_forest_model.joblib')
lstm_model = load_model('crime_lstm_model.h5')

# Preprocess data
data = data.drop(columns=['Town', 'Location', 'EndDate', 'EndTime', 'Crime'])
data['StartTime'] = data['StartTime'].fillna('00:00')
data['StartDateTime'] = pd.to_datetime(data['StartDate'] + ' ' + data['StartTime'], format='%Y-%m-%d %H:%M')
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['Minute'] = data['StartDateTime'].dt.minute
data = data.drop(columns=['StartDate', 'StartTime'])
scaler.fit(data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

# Define sequence length for LSTM
seq_length = 10

# Prepare future data for prediction
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

def pad_sequences(X, seq_length):
    num_sequences = (len(X) + seq_length - 1) // seq_length
    pad_size = num_sequences * seq_length - len(X)
    if pad_size > 0:
        X_padded = np.concatenate([X, np.zeros((pad_size, X.shape[1]))], axis=0)
    else:
        X_padded = X
    return X_padded

# Prediction functions
def predict_knn_rf(model, X):
    return model.predict_proba(X)

def predict_lstm(X):
    X_seq = pad_sequences(X, seq_length)
    num_sequences = X_seq.shape[0] // seq_length
    X_seq_reshaped = X_seq.reshape(num_sequences, seq_length, -1)
    probabilities = lstm_model.predict(X_seq_reshaped)

    if probabilities.shape[0] < 30:
        probabilities = np.repeat(probabilities, 30 // probabilities.shape[0], axis=0)
        if probabilities.shape[0] < 30:
            probabilities = np.concatenate([probabilities, np.zeros((30 - probabilities.shape[0], probabilities.shape[1]))], axis=0)

    return probabilities[:30]

def predict_dqn(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        q_values = dqn_model(X_tensor)
    if q_values.shape[1] < 18:
        q_values = np.concatenate([q_values.numpy(), np.zeros((X.shape[0], 18 - q_values.shape[1]))], axis=1)
    return q_values

# Haversine formula for distance calculation
def haversine(lon1, lat1, lon2, lat2):
    R = 6371e3  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Distance in meters

# Convert hour to time range
def get_time_range(hour):
    if 0 <= hour < 6:
        return "00:00 - 06:00"
    elif 6 <= hour < 12:
        return "06:00 - 12:00"
    elif 12 <= hour < 18:
        return "12:00 - 18:00"
    else:
        return "18:00 - 24:00"

# Ensemble method (soft voting)
def predict_ensemble(X, center_long, center_lat, radius):
    normalized_data = scaler.transform(X[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

    knn_proba = predict_knn_rf(knn_model, normalized_data)
    rf_proba = predict_knn_rf(rf_model, normalized_data)
    lstm_proba = predict_lstm(normalized_data)
    dqn_values = predict_dqn(normalized_data)
    dqn_proba = nn.functional.softmax(torch.tensor(dqn_values), dim=-1).numpy()

    # Weighting the models based on validation performance
    weights = np.array([0.2, 0.2, 0.5, 0.1])  # Adjust these weights based on validation results
    weighted_avg_proba = (weights[0] * knn_proba + weights[1] * rf_proba + weights[2] * lstm_proba + weights[
        3] * dqn_proba) / sum(weights)

    final_prediction = np.argmax(weighted_avg_proba, axis=-1)
    predicted_classes = label_encoder.inverse_transform(final_prediction)

    # Calculate distances and filter by radius
    distances = haversine(center_long, center_lat, X['Longitude'], X['Latitude'])
    X['Distance'] = distances
    filtered_data = X[X['Distance'] <= radius]

    # Prepare results with predictions and probabilities for filtered data
    filtered_classes = predicted_classes[X.index.isin(filtered_data.index)]
    filtered_probabilities = weighted_avg_proba[X.index.isin(filtered_data.index), :].max(axis=1)

    filtered_data['Predicted CrimeType'] = filtered_classes
    filtered_data['Probability'] = filtered_probabilities

    # Add predicted time range to the filtered data
    filtered_data['Predicted Time'] = filtered_data['Hour'].apply(get_time_range)

    return filtered_data

# Define a route for predictions
@ensemble_bp.route('/predict', methods=['GET'])
def predict():
    # Define the center point and radius
    center_longitude = 80.6219875521929
    center_latitude = 7.322798069315021
    radius = 3651  # Radius in meters

    future_data = create_future_data()
    predicted_data = predict_ensemble(future_data, center_longitude, center_latitude, radius)

    # Get top 10 locations with highest probability
    top_10_locations = predicted_data.nlargest(6, 'Probability')

    results = []
    for _, row in top_10_locations.iterrows():
        results.append({
            "Day": row['Day'],
            "Distance": row['Distance'],
            "Hour": row['Hour'],
            "Latitude": row['Latitude'],
            "Longitude": row['Longitude'],
            "Minute": row['Minute'],
            "Month": row['Month'],
            "Predicted CrimeType": row['Predicted CrimeType'],
            "Predicted Time": row['Predicted Time'],
            "Probability": round(row['Probability'], 10),  # Use 10 decimal places for consistency
            "Year": row['Year']  # Ensure Year is included in the output
        })

    return jsonify(results)
