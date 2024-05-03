import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2 * action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.action_size = action_size

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        state_tensor.required_grad = True

        action_parameters = self.policy_network(state_tensor).reshape((2 * self.action_size,))

        action_mean = action_parameters[::2]
        action_std = torch.exp(action_parameters[1::2])

        action_std = torch.clamp(action_std, 1e-6, 1)

        action_dist = torch.distributions.Normal(action_mean, action_std)

        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.tolist(), log_prob.mean()

    def update(self, rewards, log_probs):
        discounted_rewards = self.calculate_discounted_rewards(rewards)

        policy_loss = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            policy_loss.append((-log_prob * discounted_reward).reshape(1))

        policy_loss = torch.cat(policy_loss).sum()

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

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std(unbiased=(False if len(discounted_rewards)==1 else True)) + 1e-8

        return discounted_rewards
