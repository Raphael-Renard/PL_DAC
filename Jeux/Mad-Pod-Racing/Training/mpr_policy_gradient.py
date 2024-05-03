import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import mpr_training_env
from Jeux.policy_gradient import PolicyGradient


# Training loop
num_episodes = 2000
env = mpr_training_env.make()
state_size = 4
# action_space = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
agent = PolicyGradient(state_size, 2)

plot_rewards = []
plot_loss = []

for episode in range(num_episodes):
    try:
        state = np.array(env.reset())
        state = np.reshape(state, [1, state_size])

        rewards = []
        log_probs = []
        done = False

        for _ in range(100):
            if done:
                break

            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
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

    except ValueError:
        break

# Save policy network
torch.save(agent.policy_network.state_dict(), '../Resultats/Policy_gradient_policy_network.pth')


# Load policy network
# policy_network = PolicyNetwork(state_size, action_size)
# policy_network.load_state_dict(torch.load('../Resultats/Policy_gradient_policy_network.pth'))

plt.plot(plot_rewards)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Policy gradient : évolution du reward selon les épisodes')

reg = LinearRegression().fit(np.arange(len(plot_rewards)).reshape(-1, 1), np.array(plot_rewards).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(plot_rewards)).reshape(-1, 1))
plt.plot(y_pred)

plt.savefig('../Resultats/Policy_gradient_reward.png')
plt.show()

plt.plot(plot_loss)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Policy gradient : évolution de la loss selon les épisodes')
plt.savefig('../Resultats/Policy_gradient_loss.png')
plt.show()
