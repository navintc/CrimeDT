import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from joblib import dump  # For saving the trained model

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('../DQN/CrimeReinforcement.csv')

# Drop unnecessary columns
data = data.drop(columns=['Town', 'Location', 'EndDate', 'EndTime', 'Crime'])

data['StartTime'] = data['StartTime'].fillna('00:00')

# Convert 'StartDate' and 'StartTime' to datetime
data['StartDateTime'] = pd.to_datetime(data['StartDate'] + ' ' + data['StartTime'], format='%Y-%m-%d %H:%M')

# Extract features (additional feature engineering)
data['Year'] = data['StartDateTime'].dt.year
data['Month'] = data['StartDateTime'].dt.month
data['Day'] = data['StartDateTime'].dt.day
data['Hour'] = data['StartDateTime'].dt.hour
data['Minute'] = data['StartDateTime'].dt.minute
data['DayOfWeek'] = data['StartDateTime'].dt.dayofweek  # Monday=0, Sunday=6
data['IsWeekend'] = data['DayOfWeek'] >= 5  # New feature for weekend detection

# Drop the original 'StartDateTime' column
data = data.drop(columns=['StartDate', 'StartTime'])

# Encode 'CrimeType'
print("Encoding 'CrimeType'...")
label_encoder = LabelEncoder()
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek',
      'IsWeekend']] = scaler.fit_transform(
    data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend']])

# Prepare data for Q-Learning
print("Preparing data for Q-Learning...")
X = data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend']].values
y = data['CrimeType'].values

# Check class distribution
class_counts = pd.Series(y).value_counts()
print("Class distribution before filtering:\n", class_counts)

# Filter out low-count classes
min_count = 10
classes_to_remove = class_counts[class_counts < min_count].index
data_filtered = data[~data['CrimeType'].isin(classes_to_remove)]
X_filtered = data_filtered[
    ['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'IsWeekend']].values
y_filtered = data_filtered['CrimeType'].values

# Apply SMOTE for class imbalance
print("Applying SMOTE for class imbalance...")
smote = SMOTE(sampling_strategy='auto', random_state=42)  # Set random_state for reproducibility
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Update parameters based on resampled data
n_actions = len(np.unique(y_resampled))  # Update n_actions based on resampled data
n_states = X_resampled.shape[1]  # Use resampled data

# Define parameters
n_bins = 8  # Increase the number of bins for better granularity
n_episodes = 1000  # Increase number of episodes for better learning
max_steps = 80  # Increase the max steps to allow more exploration
alpha = 0.001  # Smaller learning rate for slower, more stable learning
gamma = 0.99  # Higher discount factor for long-term rewards
epsilon = 0.5  # Start with higher exploration rate to explore more
epsilon_decay = 0.995  # Slower epsilon decay for more exploration
min_epsilon = 0.01  # Lower minimum epsilon for exploration even in late stages

# Create bins for each feature
print(f"Creating {n_bins} bins for each feature...")
bins = [np.linspace(X_resampled[:, i].min(), X_resampled[:, i].max(), n_bins) for i in range(n_states)]

# Initialize Q-table with reduced dimensions to handle large state spaces
max_state_index = n_bins ** n_states
Q = np.zeros((max_state_index, n_actions))

# List to store episode accuracies for plotting
episode_accuracies = []


def discretize_state(state, bins):
    discretized = [np.digitize(state[i], bins[i]) for i in range(len(state))]
    index = 0
    for i in range(len(discretized)):
        index += discretized[i] * (n_bins ** i)
    return min(index, max_state_index - 1)


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(n_actions)  # Explore
    else:
        state_idx = discretize_state(state, bins)  # Discretized state
        action = np.argmax(Q[state_idx])  # Exploit
    return action


# Training with enhanced reward system and adjusted epsilon decay
for episode in range(n_episodes):
    if episode % 100 == 0:
        print(f"Episode {episode}/{n_episodes} - Epsilon: {epsilon:.2f}")
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decaying epsilon after each episode
    episode_predictions = []

    for step in range(len(X_resampled) - max_steps):
        state = X_resampled[step]  # Current state
        state_idx = discretize_state(state, bins)  # Discretized state
        action = choose_action(state, epsilon)
        reward = 10 if y_resampled[step] == action else -5  # Stronger penalty for wrong predictions
        next_state = X_resampled[step + 1]  # Next state
        next_state_idx = discretize_state(next_state, bins)
        best_next_action = np.argmax(Q[next_state_idx])
        td_target = reward + gamma * Q[next_state_idx][best_next_action]
        td_error = td_target - Q[state_idx][action]
        Q[state_idx][action] += alpha * td_error

        episode_predictions.append(action)

    # Calculate accuracy for the episode
    episode_accuracy = accuracy_score(y_resampled[:len(episode_predictions)], episode_predictions)
    episode_accuracies.append(episode_accuracy)

print("Training completed.")


# Evaluate the Q-table
def predict(state):
    state_idx = discretize_state(state, bins)
    return np.argmax(Q[state_idx])


# Evaluate model performance
print("Evaluating model performance...")
predictions = [predict(X_resampled[i]) for i in range(len(X_resampled))]

# Get unique classes in the resampled data
unique_classes = np.unique(y_resampled)

# Get the corresponding target names for these unique classes
target_names = label_encoder.inverse_transform(unique_classes)

# Generate classification report with updated target names
accuracy = accuracy_score(y_resampled, predictions)
report = classification_report(y_resampled, predictions, labels=unique_classes, target_names=target_names)

# Save performance report to file
performance_save_path = 'model_performance_rl.txt'
with open(performance_save_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n\n')
    f.write('Classification Report:\n')
    f.write(report)

print(f"Performance indicators saved to {performance_save_path}")

# Create confusion matrix
cm = confusion_matrix(y_resampled, predictions, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

# Plot confusion matrix
print("Plotting confusion matrix...")
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Q-Learning)')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
confusion_matrix_path = 'QL_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.show()

# Save trained Q-table
model_save_path = 'trained_q_table.joblib'
dump(Q, model_save_path)
print(f"Trained model (Q-table) saved to {model_save_path}")

# Plot episode accuracies over time
plt.figure(figsize=(12, 6))
plt.plot(episode_accuracies, label='Episode Accuracy', color='green')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Episodes')
plt.grid()
accuracy_plot_path = 'episode_accuracy.png'
plt.savefig(accuracy_plot_path)  # Save accuracy plot as an image
plt.show()

print(f"Episode accuracy graph saved to {accuracy_plot_path}")
