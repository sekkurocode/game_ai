import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple, deque
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """The DQN model"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 4)

    def forward(self, x):
        """A forward pass through the network"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

model = torch.load("net_lunar.pt", weights_only=False)


policy_net = DQN().to(device)
target_net = DQN().to(device)
policy_net.load_state_dict(model["policy_net_state_dict"])
target_net.load_state_dict(model["target_net_state_dict"])

env = gym.make("LunarLander-v3", render_mode="human")
# state, info = env.reset()

counter = 0
num_episodes = 50
for i in range(num_episodes):
    counter += 1
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state

            
        if done:
            break
    
    
env.close()