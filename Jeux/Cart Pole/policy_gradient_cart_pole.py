import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, action_size)

    def forward(self, x):
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
            policy_loss.append((-log_prob * discounted_reward).reshape(1))
        
        policy_loss = torch.cat(policy_loss).sum()
        #print("policy loss:",policy_loss.item())
        loss = policy_loss.item()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return loss

    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        return discounted_rewards





# Training loop
num_episodes = 700
env = gym.make('CartPole-v1',render_mode = None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 
agent = PolicyGradient(state_size, action_size)

plot_rewards = []
plot_loss = []


for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    
    rewards = []
    log_probs = []
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _,_ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        state=next_state
        if sum(rewards)>1000:
            done=True
    
    loss = agent.update(rewards, log_probs)
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}, Loss: {loss}")
    
    plot_rewards.append(sum(rewards))
    plot_loss.append(loss)

plt.plot(plot_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Score per run')
plt.show()

plt.plot(plot_loss)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Loss per episode')
plt.show()