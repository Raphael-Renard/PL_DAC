import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Game_representation import Morpion



class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.policy_network = nn.Sequential(
        nn.Linear(state_size, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, action_size),
        nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, env, state):
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
        if discounted_rewards.size(0)>1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (torch.std(discounted_rewards, dim=0) + 1e-8)
        return discounted_rewards
    







state_size = 81  
action_size = 81 

agent = PolicyGradient(state_size, action_size)
env = Morpion()


# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    env.reset()
    
    rewards = []
    log_probs = []
    done = False
    while not done:
        state = np.reshape(env.boards, (1, 81))
        action, log_prob = agent.select_action(env,state)
        next_state, reward, done = env.step2(action)
        rewards.append(reward)
        log_probs.append(log_prob)
    agent.update(rewards, log_probs)
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}")
