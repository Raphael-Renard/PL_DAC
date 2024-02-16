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
    q_table = np.zeros((game.nb_states, game.nb_actions))

    for episode in range(num_episodes):
        state = game.reset()
        done = False

        while not done:
            # Choix de l'action : epsilon greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(game.nb_actions)
            else:
                action = np.argmax(q_table[state])

            # Exécution de l'action et obtention du nouvel état et de la récompense
            next_state, reward, done = game.step(action)

            # Mise à jour de Q
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

    return q_table
