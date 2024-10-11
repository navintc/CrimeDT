from collections import Counter

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../ss.csv')

# Drop unnecessary columns
data = data.drop(columns=['Town', 'Location', 'EndDate', 'EndTime', 'Crime'])

data['StartTime'] = data['StartTime'].fillna('00:00')

# Convert 'StartDate' and 'StartTime' to datetime
data['StartDateTime'] = pd.to_datetime(data['StartDate'] + ' ' + data['StartTime'], format='%Y-%m-%d %H:%M')

# Feature extraction
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['Minute'] = data['StartDateTime'].dt.minute
data['DayOfWeek'] = data['StartDateTime'].dt.dayofweek

# Drop 'StartDateTime' column
data = data.drop(columns=['StartDate', 'StartTime'])

# Encode 'CrimeType'
label_encoder = LabelEncoder()
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])

# Features and target variable
X = data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek']]
y = data['CrimeType']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
min_samples = 6
counts = Counter(y)
rare_classes = [cls for cls, count in counts.items() if count < min_samples]

# Filter the data to remove these rare classes
mask = ~y.isin(rare_classes)
X_filtered = X_scaled[mask]
y_filtered = y[mask]

# Now apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define hyperparameter grid for GradientBoostingClassifier and RandomForestClassifier
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)

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
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=label_encoder.classes_[unique_classes], zero_division=1))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_[unique_classes])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of Crime Data')
plt.show()

# Define possible locations based on historical data
possible_locations = data[['Longitude', 'Latitude']].drop_duplicates()

# Prepare for prediction
tomorrow = datetime.now() + timedelta(days=1)
possible_locations['Year'] = tomorrow.year
possible_locations['Month'] = tomorrow.month
possible_locations['Day'] = tomorrow.day
possible_locations['Hour'] = np.random.randint(0, 24, size=possible_locations.shape[0])
possible_locations['Minute'] = np.random.randint(0, 60, size=possible_locations.shape[0])
possible_locations['DayOfWeek'] = tomorrow.weekday()

# Normalize new data using the same scaler
X_future_scaled = scaler.transform(possible_locations[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek']])

# Predict crime types and probabilities
probabilities = best_model.predict_proba(X_future_scaled)

# Create a DataFrame with predictions and probabilities
predictions_df = possible_locations.copy()
predictions_df['Predicted CrimeType'] = best_model.predict(X_future_scaled)
predictions_df['Probabilities'] = [max(prob) for prob in probabilities]

# Add crime type names
predictions_df['Predicted CrimeType'] = label_encoder.inverse_transform(predictions_df['Predicted CrimeType'])

# Debugging: Print sample data
print("Sample predictions:")
print(predictions_df[['Longitude', 'Latitude', 'Predicted CrimeType', 'Probabilities']].head())

# Get top 10 locations with highest probability
top_10_locations = predictions_df.nlargest(10, 'Probabilities')

# Print top 10 locations with the highest probability and crime type
for index, row in top_10_locations.iterrows():
    print(f"Location: ({row['Longitude']:.6f}, {row['Latitude']:.6f}) at {row['Hour']}:{row['Minute']}")
    print(f"Predicted Crime Type: {row['Predicted CrimeType']}")
    print(f"Probability: {row['Probabilities']:.2f}")
    print()
