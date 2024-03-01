import numpy as np


def q_learning(game, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, verbose=False):
    q_table = {}
    for episode in range(num_episodes):
        print(f"Episode {episode + 1} / {num_episodes}")
        state = game.reset()
        done = False
        t=0
        while not done:
            t+=1
            if verbose:
                print(f"Itération {t}")
            if verbose:
                print("le checkpoint est a la postion :", game.checkpoint_pos)
                print("State : ", state)
                print("actual done", done)

            if state not in q_table:
                q_table[state] = np.zeros(game.nb_actions)

            if np.random.rand() < epsilon:
                if verbose:
                    print("-------------------EXPLORATION--------------------------")
                action = np.random.randint(game.nb_actions)
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
            next_state, reward, done = game.step(action, verbose)
            if verbose:
                print("next_state", next_state)
                print("reward ", reward)
                print("new done", done)
                print("\n\n")

            if next_state not in q_table:
                q_table[next_state] = np.zeros(game.nb_actions)

            # Mise à jour de Q
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            if verbose:
                print("state", state)
                print("q_table maj : ", q_table[state][action])

            state = next_state

            if verbose:
                print(q_table)
                print("\n")

    return q_table
