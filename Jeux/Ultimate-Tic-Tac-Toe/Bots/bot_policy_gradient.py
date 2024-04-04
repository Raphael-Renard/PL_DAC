import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Game_representation import Morpion

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        print("input:",x)
        print("couche 1:",self.fc1(x))
        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def update(self, rewards, log_probs):
        discounted_rewards = self.calculate_discounted_rewards(rewards)
        policy_loss = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * discounted_reward)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        return discounted_rewards

state_size = 81  
action_size = 81 

agent = PolicyGradient(state_size, action_size)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    env = Morpion(boards=np.zeros((3, 3, 3, 3), dtype=int),
                  big_boards=np.zeros((3, 3), dtype=int),
                  empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                                 [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                                 [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                                [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                                 [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                                 [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                                [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                                 [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                                 [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]],
                  empty_all={(i, j) for i in range(9) for j in range(9)})
    
    rewards = []
    log_probs = []
    done = False
    while not done:
        state = np.reshape(env.boards, (1, 81))
        action, log_prob = agent.select_action(state)
        next_state, reward, done = env.step2(action)
        rewards.append(reward)
        log_probs.append(log_prob)
    print(episode)
    agent.update(rewards, log_probs)
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}")
