import pickle
import numpy as np
import importlib

from queue import Queue
from threading import Thread

from matplotlib import pyplot as plt

from Qlearning import (q_learning)

game_module = importlib.import_module("Jeux.Mad-Pod-Racing.Implementation.game")
game_loop_terminal = game_module.game_loop_terminal
generate_checkpoints = game_module.generate_checkpoints
bot_heuristique = importlib.import_module("Jeux.Mad-Pod-Racing.Implementation.Bots.bot_heuristique").bot_heuristique
bot_qlearning = importlib.import_module("Jeux.Mad-Pod-Racing.Implementation.Bots.bot_qlearning").bot_qlearning
import mpr_training_env

nb_intervalles = 10
intervalles = np.arange(1, 100002, 100000 / nb_intervalles)

res_heuristique = []
res_qlearning_1 = []
res_qlearning_2 = []

env = mpr_training_env.make((5, 5, 5, 5), (5, 3))

player_threads = []
player_queues = []

checkpoints_list = [generate_checkpoints() for _ in range(1000)]

for i in range(nb_intervalles+1):
    nb_iterations = max(1, intervalles[i] - intervalles[i-1])
    print(i)
    q_table_1 = q_learning(env, num_episodes=int(nb_iterations), gamma=1, qtable_path=("q_table_1" if i != 0 else None))
    q_table_2 = q_learning(env, num_episodes=int(nb_iterations), gamma=.99, qtable_path=("q_table_2" if i != 0 else None))

    with open('q_table_1.pkl', 'wb') as f:
        pickle.dump(q_table_1, f)
    with open('q_table_2.pkl', 'wb') as f:
        pickle.dump(q_table_2, f)

    player_types = [bot_heuristique, bot_qlearning, bot_qlearning]
    player_names = ["Bot heuristique", "Bot qlearning gamma = 1", "Bot qlearning gamma = .99"]

    if player_threads:
        for queue, _ in player_queues:
            queue.put("end")

    player_threads = []
    player_queues = []

    for j, player in enumerate(player_types):
        player_send_q = Queue()
        player_receive_q = Queue()
        player_queues.append((player_send_q, player_receive_q))
        if j == 0:
            player_threads.append(Thread(target=player, args=(player_send_q, player_receive_q), daemon=True))
        else:
            player_threads.append(Thread(target=player, args=(player_send_q, player_receive_q, f"q_table_{j}"), daemon=True))
        player_threads[-1].start()

    nb_iter_tot = [0] * len(player_names)
    nb_games = 1000
    progress = np.arange(0, nb_games, nb_games // 10)

    print(i)

    for j in range(nb_games):
        nb_iter = game_loop_terminal(player_queues, player_names, first_ends=False, checkpoints=checkpoints_list[j])

        nb_iter_tot = [sum(k) for k in zip(nb_iter_tot, nb_iter)]

        if j in progress:
            print(f"{j / nb_games * 100:.0f}%")

    res_heuristique.append(nb_iter_tot[0] / nb_games)
    res_qlearning_1.append(nb_iter_tot[1] / nb_games)
    res_qlearning_2.append(nb_iter_tot[2] / nb_games)

plt.plot(intervalles, res_heuristique, label="bot heuristique")
plt.plot(intervalles, res_qlearning_1, label="bot qlearning gamma=1")
plt.plot(intervalles, res_qlearning_2, label="bot qlearning gamma=.99")
plt.xlabel("Nombre d'épisodes de train pour Qlearning")
plt.ylabel("Nombre d'itérations par partie moyen")
plt.title("Evolution du temps moyen de courses de divers profils de bots\nen fonction du nombre d'épisodes de train")
plt.legend()
plt.savefig("../Resultats/Qlearning-comparaison-bots.png")
plt.show()


