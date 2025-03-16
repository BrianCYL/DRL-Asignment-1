# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
from utils import DQNAgent
import argparse

agent = DQNAgent(16, 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent.Q.load_state_dict(torch.load("checkpoints/model_256_wo_soft_update.pth", map_location=device))
agent.Q.to(device)

def get_action(obs):
    agent.Q.eval()
    state = torch.tensor(obs, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return agent.select_action(state, 0.0, train=False)
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
