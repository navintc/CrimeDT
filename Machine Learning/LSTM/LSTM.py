import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Load and preprocess data
data = pd.read_csv('CrimeLSTM.csv')

# Drop unnecessary columns
data = data.drop(columns=['Town', 'Location', 'EndDate', 'EndTime', 'Crime'])

data['StartTime'] = data['StartTime'].fillna('00:00')

# Convert 'StartDate' and 'StartTime' to datetime
data['StartDateTime'] = pd.to_datetime(data['StartDate'] + ' ' + data['StartTime'], format='%Y-%m-%d %H:%M')

# Extract features
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['Minute'] = data['StartDateTime'].dt.minute

# Drop the original 'StartDateTime' column
data = data.drop(columns=['StartDate', 'StartTime'])

# Encode 'CrimeType'
label_encoder = LabelEncoder()
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])

# Normalize features
scaler = StandardScaler()
data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']] = scaler.fit_transform(
    data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

# Prepare sequences
def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length])
    return np.array(sequences), np.array(labels)

# Define sequence length
seq_length = 10
X = data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']].values
y = data['CrimeType'].values

# Create sequences
X_seq, y_seq = create_sequences(X, y, seq_length)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)

# Define the LSTM model with improved architecture
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(seq_length, X_seq.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.8

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model
history = model.fit(X_train, y_train, epochs=23, validation_data=(X_test, y_test), verbose=2, callbacks=[lr_scheduler])

# Save the trained model
model_save_path = 'crime_lstm_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Print classification report and accuracy
# Get the unique class labels present in y_test


# Get the unique class labels present in y_test
unique_classes = np.unique(y_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.inverse_transform(unique_classes), labels=unique_classes)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save classification report and accuracy
performance_save_path = '../model_performance.txt'
with open(performance_save_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n\n')
    f.write('Classification Report:\n')
    f.write(report)
print(f"Performance indicators saved to {performance_save_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.inverse_transform(unique_classes))
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (LSTM)')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
confusion_matrix_path = 'LSTM_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.show()

plt.show()

