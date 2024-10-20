from threading import Thread
from queue import Queue

from game import *

from Bots.bot_heuristique import *
from Bots.bot_simple import *
from Bots.bot_qlearning import *
from Bots.bot_policy_gradient import *
from Bots.bot_dqn import *
if __name__ == "__main__":
    gui = True
    # gui = False
    player_verbose = False

    player_types = [bot_simple, bot_qlearning, bot_policy_gradient,bot_dqn]
    player_names = ["Bot simple", "Bot qlearning", "Bot policy gradient","Bot dqn"]
    player_colours = ["blue", "yellow", "green","red"]

    # player_types = [bot_simple]
    # player_names = ["Bot simple"]
    # player_colours = ["blue"]

    player_threads = []
    player_queues = []

    for player in player_types:
        player_send_q = Queue()
        player_receive_q = Queue()
        player_queues.append((player_send_q, player_receive_q))
        player_threads.append(Thread(target=player, args=(player_send_q, player_receive_q), daemon=True))
        player_threads[-1].start()

    if gui:
        game_loop_gui(player_queues, player_names, player_colours, player_verbose)
    else:
        win_count = [0] * len(player_names)
        nb_games = 1000
        progress = np.arange(0, nb_games, nb_games // 10)
        for i in range(nb_games):
            win = game_loop_terminal(player_queues, player_names, player_verbose)
            for w in win:
                win_count[w - 1] += 1

            if i in progress:
                print(f"{i / nb_games * 100:.0f}%")

        print(f"Win count:")
        lenmax = max([len(n) for n in player_names])
        for i, w in enumerate(win_count):
            print(f"\t{player_names[i]} {' ' * (lenmax - len(player_names[i]))}: {w} \t- {w / nb_games * 100:.2f}%")
