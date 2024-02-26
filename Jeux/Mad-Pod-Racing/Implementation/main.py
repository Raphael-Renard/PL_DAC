from threading import Thread
from queue import Queue

from game import *

from Bots.bot_heuristique import *
from Bots.bot_simple import *

if __name__ == "__main__":
    gui = True
    player_verbose = True

    player_types = [bot_heuristique, bot_simple]
    player_names = ["Bot heuristique", "Bot simple"]
    player_colours = ["red", "blue"]

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
        game_loop_terminal(player_queues, player_names, player_verbose)
