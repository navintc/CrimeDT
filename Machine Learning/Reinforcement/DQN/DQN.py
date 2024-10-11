import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
from collections import deque

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('CrimeReinforcement.csv')

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
print("Encoding 'CrimeType'...")
label_encoder = LabelEncoder()
data['CrimeType'] = label_encoder.fit_transform(data['CrimeType'])

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']] = scaler.fit_transform(
    data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']])

# Prepare data for DQN
print("Preparing data for DQN...")
X = data[['Longitude', 'Latitude', 'Year', 'Month', 'Day', 'Hour', 'Minute']].values
y = data['CrimeType'].values

# Define DQN neural network architecture
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

# Hyperparameters
n_actions = len(label_encoder.classes_)
n_states = X.shape[1]
batch_size = 128
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
target_update = 10  # How often to update target network
memory_size = 10000
n_episodes = 1000
max_steps = 100

# Initialize DQN and target network
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network is not trained

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Replay memory for experience replay
memory = deque(maxlen=memory_size)

def choose_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax().item()  # Exploit

def store_transition(state, action, reward, next_state, done):
    """Store a transition in replay memory."""
    memory.append((state, action, reward, next_state, done))

def sample_memory(batch_size):
    """Sample a batch of transitions from replay memory."""
    return random.sample(memory, batch_size)

def update_model():
    """Update the policy network using experience replay."""
    if len(memory) < batch_size:
        return

    batch = sample_memory(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert lists of numpy arrays to a single numpy array before creating PyTorch tensors
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute current Q values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Loss and optimization step
    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize list to store total rewards for each episode
total_rewards = []


# Training loop
print("Training the DQN model...")
for episode in range(n_episodes):
    state = X[0]
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state = X[step + 1]
        reward = 1 if y[step] == action else -1
        done = step == max_steps - 1

        store_transition(state, action, reward, next_state, done)
        update_model()

        state = next_state
        total_reward += reward

        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network periodically
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Append total reward after each episode
    total_rewards.append(total_reward)

    # Print total reward after every episode (no condition needed)
    print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward}")

# Save training reward plot
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, label="Total Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Over Episodes (DQN)")
plt.legend()
training_plot_path = 'dqn_training_rewards.png'
plt.savefig(training_plot_path)
plt.show()

# Save the model
model_save_path = 'dqn_model.pth'
torch.save(policy_net.state_dict(), model_save_path)
print(f"Trained DQN model saved to {model_save_path}")

# Evaluation
def predict(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return policy_net(state).argmax().item()

# Evaluate model performance
print("Evaluating model performance...")
predictions = [predict(X[i]) for i in range(len(X))]
accuracy = accuracy_score(y, predictions)
report = classification_report(y, predictions, target_names=label_encoder.classes_)
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Save performance metrics
performance_save_path = 'dqn_model_performance.txt'
with open(performance_save_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n\n')
    f.write('Classification Report:\n')
    f.write(report)

print(f"Performance metrics saved to {performance_save_path}")

# Plot and save confusion matrix
print("Plotting confusion matrix...")
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (DQN)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
confusion_matrix_path = 'DQN_confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.show()

# Save confusion matrix and training rewards plot paths
print(f"Confusion matrix saved to {confusion_matrix_path}")
print(f"Training reward plot saved to {training_plot_path}")
