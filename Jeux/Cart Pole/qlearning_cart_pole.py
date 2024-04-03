import importlib

import gym
import numpy as np

from Dqn_Cartpole import plot_res


def discretiser_etat(state):
    cart_position, cart_velocity, pole_angle, pole_velocity = state

    # Discretize into shape (1, 10, 10, 10) using log intervals
    cart_position = np.digitize(cart_position, np.logspace(-1, 1, 10))
    cart_velocity = np.digitize(cart_velocity, np.logspace(-1, 1, 10))
    pole_angle = np.digitize(pole_angle, np.linspace(-1, 1, 10))
    pole_velocity = np.digitize(pole_velocity, np.linspace(-1, 1, 10))

    return cart_position, cart_velocity, pole_angle, pole_velocity


if __name__ == "__main__":

    Qlearning = importlib.import_module("Jeux.Mad-Pod-Racing.Training.Qlearning").q_learning
    env = gym.make('CartPole-v1')

    qtable, rewards = Qlearning(env, num_episodes=100000, gamma=.99, disc=discretiser_etat, get_rewards=True)

    plot_res(rewards, filepath="apprentissage_cart_pole_qlearning.png")

    env = gym.make('CartPole-v1', render_mode='human')

    # Display a game with the trained agent
    done = False
    state = env.reset()
    state = discretiser_etat(state[0])
    while not done:
        try:
            action = np.argmax(qtable[state])
        except KeyError:
            action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        state = discretiser_etat(state)
        done = terminated or truncated
        env.render()
    env.close()
