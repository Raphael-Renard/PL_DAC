import pickle
import codecs
import math

import numpy as np

from Qlearning import q_learning  # , Game
import mpr_training_env

if __name__ == "__main__":
    new_qtable = True
    nb_tests = 10000

    env = mpr_training_env.make((9, 9, 9, 9), (5, 3))

    if new_qtable:
        q_table = q_learning(env, num_episodes=50000, gamma=1)

        # Dump q_table
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)
        print(codecs.encode(pickle.dumps(q_table), "base64").decode())

    else:
        # Load q_table
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)

    # Test de l'agent entraîné
    times = np.array([env.test(q_table, 100) for _ in range(nb_tests)])

    print(f"Pourcentage de timeout : {round(len(times[times == -1]) / nb_tests, 4) * 100}\nPourcentage d'états non reconnus : {round(len(times[times == -2]) / nb_tests, 4) * 100}\nNombre d'itérations moyen : {times[times >= 0].mean()}")
