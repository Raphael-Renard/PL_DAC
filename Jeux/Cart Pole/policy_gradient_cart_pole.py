import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

from Jeux.policy_gradient import PolicyGradient
from continuous_cart_pole import ContinuousCartPoleEnv


class Cart_Pole_PolicyNetwork(nn.Module):
    def __init__(self, state_size):
        super(Cart_Pole_PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class Cart_pole_policy_gradient(PolicyGradient):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.policy_network = Cart_Pole_PolicyNetwork(state_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)


# Training loop
num_episodes = 1000
max_steps_per_episode = 500
env = ContinuousCartPoleEnv()
state_size = 4
action_size = 1
agent = Cart_pole_policy_gradient(state_size, action_size)

plot_rewards = []
plot_loss = []

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    rewards = []
    log_probs = []
    done = False
    while not done and len(rewards) < max_steps_per_episode:
        action, log_prob = agent.select_action(state)

        action = action[0]

        if action < -1:
            action = -1
        elif action > 1:
            action = 1

        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state
        if sum(rewards) > 1000:
            done = True

    loss = agent.update(rewards, log_probs)
    if (episode + 1) % 10 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}, Loss: {loss}")

    plot_rewards.append(sum(rewards))
    plot_loss.append(loss)

    if np.array(plot_rewards).mean() >= 195 and len(plot_rewards) >= 100:
        break

plt.plot(plot_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Score per run')

reg = LinearRegression().fit(np.arange(len(plot_rewards)).reshape(-1, 1), np.array(plot_rewards).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(plot_rewards)).reshape(-1, 1))
plt.plot(y_pred)

plt.savefig('./policy_gradient_score.png')

plt.show()

plt.plot(plot_loss)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Loss per episode')
plt.show()
