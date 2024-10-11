import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from joblib import dump
import seaborn as sns

# Load your data
data = pd.read_csv('CrimeKNN.csv')

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

# Features and target variable
X = data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
y = data['CrimeType']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Print best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
y_pred = best_model.predict(X_test)

# Get the unique class labels in the test data
unique_classes = np.unique(y_test)

# Print classification report and accuracy
classification_report_text = classification_report(y_test, y_pred, labels=unique_classes, target_names=label_encoder.classes_[unique_classes], zero_division=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save classification report and accuracy to a text file
with open('knn_performance_metrics.txt', 'w') as f:
    f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report_text)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_[unique_classes])

# Plot confusion matrix and save it
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (KNN)')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
confusion_matrix_path = 'knn_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')


# Save cross-validation scores to the file
with open('knn_performance_metrics.txt', 'a') as f:
    f.write(f"Cross-Validation Scores: {cv_scores}\n")
    f.write(f"Accuracy: {accuracy}\n")

# Plot cross-validation accuracy scores
plt.figure(figsize=(8, 6))
plt.plot(cv_scores, marker='o', color='b', label='Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy Over Folds (KNN)')
plt.grid(True)
cv_accuracy_plot_path = 'knn_cv_accuracy_plot.png'
plt.savefig(cv_accuracy_plot_path)
plt.show()

# Save the trained model
model_save_path = 'best_knn_model.joblib'
dump(best_model, model_save_path)
print(f"Trained KNN model saved to {model_save_path}")

# Define possible locations based on historical data
possible_locations = data[['Longitude', 'Latitude']].drop_duplicates()

# Prepare for prediction
tomorrow = datetime.now() + timedelta(days=1)
possible_locations['Year'] = tomorrow.year
possible_locations['Month'] = tomorrow.month
possible_locations['Day'] = tomorrow.day
possible_locations['Hour'] = np.random.randint(0, 24, size=possible_locations.shape[0])
possible_locations['Minute'] = np.random.randint(0, 60, size=possible_locations.shape[0])

# Debugging: Print sample possible locations before scaling
print("Sample possible locations before prediction:")
print(possible_locations[['Longitude', 'Latitude']].head())

# Predict crime types and probabilities
X_future = possible_locations[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
X_future = scaler.transform(X_future)  # Apply the same scaling to future data
probabilities = best_model.predict_proba(X_future)

# Create a DataFrame with predictions and probabilities
predictions_df = possible_locations.copy()
predictions_df['Predicted CrimeType'] = best_model.predict(X_future)
predictions_df['Probabilities'] = [max(prob) for prob in probabilities]

# Add crime type names
predictions_df['Predicted CrimeType'] = label_encoder.inverse_transform(predictions_df['Predicted CrimeType'])

# Debugging: Print sample data
print("Sample predictions:")
print(predictions_df[['Longitude', 'Latitude']].head())

# Get top 10 locations with highest probability
top_10_locations = predictions_df.nlargest(10, 'Probabilities')

# Save top 10 locations predictions to a file
with open('knn_top_10_predictions.txt', 'w') as f:
    for index, row in top_10_locations.iterrows():
        f.write(f"Location: ({row['Longitude']:.6f}, {row['Latitude']:.6f}) at {row['Hour']}:{row['Minute']}\n")
        f.write(f"Predicted Crime Type: {row['Predicted CrimeType']}\n")
        f.write(f"Probability: {row['Probabilities']:.2f}\n\n")

# Print the top 10 locations
for index, row in top_10_locations.iterrows():
    print(f"Location: ({row['Longitude']:.6f}, {row['Latitude']:.6f}) at {row['Hour']}:{row['Minute']}")
    print(f"Predicted Crime Type: {row['Predicted CrimeType']}")
    print(f"Probability: {row['Probabilities']:.2f}")
    print()
