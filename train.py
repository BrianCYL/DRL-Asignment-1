# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from simple_custom_taxi_env import SimpleTaxiEnv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from utils import DQNAgent, ReplayBuffer
import argparse
import os
import wandb

# TODO: Train your own agent
# HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
# NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#       To prevent crashes, implement a fallback strategy for missing keys. 
#       Otherwise, even if your agent performs well in training, it may fail during testing.

class DQNAgentTrainer:
    def __init__(self, args):
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.epsilon = args.eps_start
        self.eps_end = args.eps_end
        self.decay_rate = args.decay_rate
        self.tau = args.tau
        self.agent = DQNAgent(args.state_size, args.action_size)
        self.optimizer = optim.SGD(self.agent.Q.parameters(), lr=args.alpha)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(args.buffer_size, self.device)
        self.batch_size = args.batch_size
        self.n_episodes = args.n_episode
        self.update_step = args.update_step
    
    def state_transform(self, obs):
        # binary_string = ''.join(str(x) for x in obs[10:14])  # Convert tensor to binary string
        # obstacle_value = int(binary_string, 2)  # Convert binary string to integer
        # state = obs[:10] + (obstacle_value,) + obs[14:]
        state = torch.tensor(obs, dtype=torch.float32)
        return state
    
    def update(self, state, action, target):
        self.agent.Q.train()
        self.optimizer.zero_grad()
        action = action.to(self.device).long()
        target = target.detach()
        value = self.agent.Q(state).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.criterion(value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, args):
        if args.use_wandb:
            wandb.login()
            wandb.init(project=args.wandb_project, config=args, name=f"{args.wandb_run_name}")
            wandb.watch(self.agent.Q)

        self.agent.Q.to(self.device)
        self.agent.target_Q.to(self.device)
        reward_per_episode = [] 
        env = SimpleTaxiEnv(grid_size=np.random.randint(5, 11))
        env = SimpleTaxiEnv()
        loss_per_episode = []
        for episode in tqdm(range(self.n_episodes)):
            if (episode + 1) % 100 == 0:
                env = SimpleTaxiEnv(grid_size=np.random.randint(5, 11))
            obs, _ = env.reset()

            with torch.no_grad():
                state = self.state_transform(obs)
                # state = torch.tensor(obs, dtype=torch.float32)
            done = False
            total_reward = 0          
            total_loss = 0
            steps = 0
            while not done:
                action = self.agent.select_action(state.to(self.device), epsilon=self.epsilon)
                next_obs, reward, done, _ = env.step(action)
                next_state = self.state_transform(next_obs)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.memory) > self.batch_size:
                    s, a, r, ns, d = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        next_q_values = self.agent.target_Q(ns).max(1)[0]
                        target = r + (1 - d.float()) * self.gamma * next_q_values
                    loss = self.update(s, a, target)
                    total_loss += loss
                        
                if (steps + 1) % self.update_step == 0:
                    target_net_state_dict = self.agent.target_Q.state_dict()
                    Q_net_state_dict = self.agent.Q.state_dict()
                    for key in Q_net_state_dict:
                        target_net_state_dict[key] = Q_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                    self.agent.target_Q.load_state_dict(target_net_state_dict)
                    # self.agent.target_Q.load_state_dict(self.agent.Q.state_dict())

                steps += 1

            if args.use_wandb:
                wandb.log({'loss': total_loss / steps,
                           'reward': total_reward / steps,
                           'epsilon': self.epsilon,
                            })
                
            self.epsilon = max(self.eps_end, self.decay_rate * self.epsilon)
            reward_per_episode.append(total_reward)
            loss_per_episode.append(total_loss)
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(reward_per_episode[-100:])
                avg_loss = np.mean(loss_per_episode[-100:])
                print(f"Episode: {episode + 1}, Average reward: {avg_reward}, Avg Loss: {avg_loss}, Epsilon: {self.epsilon}")
            
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        torch.save(self.agent.Q.state_dict(), f'checkpoints/model_eps_001.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='drl_assignment1')
    parser.add_argument('--wandb_run_name', type=str, default='dqn')
    parser.add_argument('--state_size', type=int, default=16)
    parser.add_argument('--action_size', type=int, default=6)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--n_episode', type=int, default=10000)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--update_step', type=int, default=100)
    parser.add_argument('--decay_rate', type=float, default=0.9995)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--use_wandb', action="store_true")
    args = parser.parse_args()

    trainer = DQNAgentTrainer(args)
    trainer.train(args)


if __name__ == '__main__':
    main()



