import numpy as np


class Game:
    def __init__(self):
        self.nb_actions = 0  # à modifier
        self.nb_states = 0  # à modifier
        self.goal_state = (0, 0)  # à modifier
        self.agent_state = (0, 0)  # à modifier

    def reset(self):
        self.agent_state = (0, 0)  # à modifier? (position de départ)
        return self.agent_state

    def step(self, action):
        if action == 'action1':  # à compléter
            next_state = "next state"
        elif action == 'action2':
            next_state = "next state"
        # ...

        self.agent_state = next_state

        if self.agent_state == self.goal_state:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        return self.agent_state, reward, done


def q_learning(game, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = {}
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        while not done:
            print("le checkpoint est a la postion :", game.checkpoint_pos)
            print("State : ", state)
            print("actual done", done)

            if state not in q_table:
                q_table[state] = np.zeros(game.nb_actions)

            if np.random.rand() < epsilon:
                print("-------------------EXPLORATION--------------------------")
                action = np.random.randint(game.nb_actions)
                print("action prise : ", action)
            else:
                print("---------------------EXPLOITATION-----------------------")
                max_action = np.argmax(q_table[state])
                max_value = q_table[state][max_action]
                max_actions = [action for action, value in enumerate(q_table[state]) if value == max_value]
                action = np.random.choice(max_actions)
                print("action prise : ", action)

            next_state, reward, done = game.step(action)
            print("next_state", next_state)
            print("reward ", reward)
            print("new done", done)
            print("\n\n")

            if next_state not in q_table:
                q_table[next_state] = np.zeros(game.nb_actions)

            # Mise à jour de Q
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            print("state", state)
            print("q_table maj : ", q_table[state][action])

            state = next_state

            print(q_table)
            print("\n")

    return q_table
