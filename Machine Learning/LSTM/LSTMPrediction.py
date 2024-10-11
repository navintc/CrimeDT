import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the model and scalers
model = load_model('crime_lstm_model.h5')
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Load and preprocess the original data to extract scaling and encoding
data = pd.read_csv('CrimeLSTM.csv')
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

# Define sequence length
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


def predict_future_crimes(future_data):
    # Normalize future data
    normalized_data = scaler.transform(
        future_data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
    )

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

    return future_data

# Create future data and predict
future_data = create_future_data()
predicted_future_data = predict_future_crimes(future_data)

# Get top 10 locations with highest probability
top_10_locations = predicted_future_data.nlargest(10, 'Probability')

# Print results
print("Top 10 locations and predicted crimes for tomorrow:")
for _, row in top_10_locations.iterrows():
    print(f"Location: ({row['Longitude']:.6f}, {row['Latitude']:.6f})")
    print(f"Predicted Crime Type: {row['Predicted CrimeType']}")
    print(f"Probability: {row['Probability']:.2f}")
    print()