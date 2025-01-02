import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
import ale_py

# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 160)
        self.fc2 = nn.Linear(160, 160)
        self.fc3 = nn.Linear(160, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define DQN Agent with Experience Replay Buffer
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        # Determine epsilon greedy strategy and linear or exponential decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)  # Move state to device
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()


    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminated = zip(*minibatch)

        # Combine lists into NumPy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        terminated = np.array(terminated, dtype=np.float32)
        actions = np.array(actions)

        # Move data to the device
        states = torch.tensor(states, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        max_next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (1 - terminated) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    gym.register_envs(ale_py)

    # Initialize Pacman environment
    env = gym.make("LunarLander-v3")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Initialize agent with Experience Replay Buffer
    agent = DQNAgent(state_dim, action_dim, lr=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

    # Train DQN agent
    batch_size = 32
    num_episodes = 1000
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Move to device
        total_reward = 0
        terminated = False

        while not terminated:
            action = agent.act(state.cpu().numpy())
            next_state, reward, terminated, _, _ = env.step(action)
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).to(device)  # Flatten and move to device
            agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), terminated)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")