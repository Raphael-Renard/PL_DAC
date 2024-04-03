import pickle

import numpy as np


def q_learning(game, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, verbose=False, qtable_path=None, disc=None, get_rewards=False):
    try:
        nb_actions = game.nb_actions
    except AttributeError:
        nb_actions = game.action_space.n

    if qtable_path:
        with open(f"{qtable_path}.pkl", "rb") as qtable_file:
            q_table = pickle.load(qtable_file)
    else:
        q_table = {}
    if get_rewards:
        rewards = []
    e = 0
    for episode in range(num_episodes):
        if get_rewards:
            total_reward = 0
        e += 1
        if e % 100 == 0:
            print(f"Episode {episode + 1} / {num_episodes}")
        state = game.reset()

        if disc:
            state = disc(state[0])

        done = False
        t = 0
        while not done:
            t += 1
            if verbose:
                print(f"Itération {t}")
            if verbose:
                print("le checkpoint est a la postion :", game.checkpoint_pos)
                print("State : ", state)
                print("actual done", done)

            if state not in q_table:
                q_table[state] = np.zeros(nb_actions)

            if np.random.rand() < epsilon:
                if verbose:
                    print("-------------------EXPLORATION--------------------------")
                action = np.random.randint(nb_actions)
                if verbose:
                    print("action prise : ", action)
            else:
                if verbose:
                    print("---------------------EXPLOITATION-----------------------")
                max_action = np.argmax(q_table[state])
                max_value = q_table[state][max_action]
                max_actions = [action for action, value in enumerate(q_table[state]) if value == max_value]
                action = np.random.choice(max_actions)
                if verbose:
                    print("action prise : ", action)

            # action -= action % 10
            # action += state[0][1]
            try:
                next_state, reward, done = game.step(action)
            except ValueError:
                next_state, reward, terminated, truncated, _ = game.step(action)
                done = terminated or truncated

            if get_rewards:
                total_reward += reward

            if disc:
                next_state = disc(next_state)

            if verbose:
                print("next_state", next_state)
                print("reward ", reward)
                print("new done", done)
                print("\n\n")

            if next_state not in q_table:
                q_table[next_state] = np.zeros(nb_actions)

            # Mise à jour de Q
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            if verbose:
                print("state", state)
                print("q_table maj : ", q_table[state][action])

            state = next_state

            if verbose:
                print(q_table)
                print("\n")

        if e % 100 == 0:
            print(f"{t} itérations")

        if get_rewards:
            rewards.append(total_reward)

    if get_rewards:
        return q_table, rewards

    return q_table
