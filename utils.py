import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.Q = DQN(state_size, action_size)
        self.target_Q = DQN(state_size, action_size)
        self.target_Q.load_state_dict(self.Q.state_dict())

    def select_action(self, state, epsilon, train=True):
        with torch.no_grad():
            if train:
                action = np.random.choice(self.action_size) if np.random.rand() < epsilon else self.Q(state).argmax().item()
            else:
                action = self.Q(state).detach().argmax().item()
        # You can submit this random agent to evaluate the performance of a purely random strategy.     
        # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
        return action
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.tensor(state, dtype=torch.float32).to(self.device),
            torch.tensor(action, dtype=torch.long).to(self.device),
            torch.tensor(reward, dtype=torch.float32).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.bool).to(self.device),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.stack(state),  # Shape: (batch_size, state_dim)
            torch.stack(action),  # Shape: (batch_size,)
            torch.stack(reward),  # Shape: (batch_size,)
            torch.stack(next_state),  # Shape: (batch_size, state_dim)
            torch.stack(done),  # Shape: (batch_size,)
        )

    def __len__(self):
        return len(self.buffer)
