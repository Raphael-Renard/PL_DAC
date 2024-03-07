from threading import Thread
from queue import Queue

from game import *

from Bots.bot_heuristique import *
from Bots.bot_simple import *
from Bots.bot_qlearning import *

if __name__ == "__main__":
    gui = False
    player_verbose = False

    player_types = [bot_heuristique, bot_simple, bot_qlearning]
    player_names = ["Bot heuristique", "Bot simple", "Bot qlearning"]
    player_colours = ["red", "blue", "yellow"]

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
        for i in range(nb_games):
            win = game_loop_terminal(player_queues, player_names, player_verbose)
            for w in win:
                win_count[w-1] += 1

        print(f"Win count:")
        lenmax = max([len(n) for n in player_names])
        for i, w in enumerate(win_count):
            print(f"\t{player_names[i]} {' ' * (lenmax - len(player_names[i]))}: {w} \t- {w/nb_games*100:.2f}%")
