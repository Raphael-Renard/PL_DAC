import math
import random
import numpy as np
import copy
import csv

from Game_representation import Morpion



class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def expand(self):
        possible_moves = self.state.get_possible_moves()
        for move in possible_moves:
            new_state = self.state.make_move(move)
            new_node = Node(new_state, parent=self)
            self.children.append(new_node)

    def select_child(self, exploration_factor=1.4):
        selected_child = None
        max_ucb_value = -float('inf')
        for child in self.children:
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                ucb_value = (child.wins / child.visits) + exploration_factor * math.sqrt(math.log(self.visits) / child.visits)
            if ucb_value > max_ucb_value:
                selected_child = child
                max_ucb_value = ucb_value
        return selected_child

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)




class MonteCarloTreeSearch:
    def __init__(self, initial_state,record=True, simulations=100, exploration_factor = 1.4):
        self.record = record
        self.root = Node(initial_state)
        self.played_games = []  # Liste pour stocker les parties jouées
        self.simulations = simulations
        self.exploration_factor = exploration_factor

    def select_move(self):
        for _ in range(self.simulations):
            node = self.root
            game_moves = []  # Liste pour stocker les coups de chaque partie
            if not node.children:
                node.expand()
            selected_child = node.select_child(self.exploration_factor)
            if self.record:
                game_moves.append(selected_child.state.last_move)  # Ajouter le coup à la liste des coups de la partie
            result = self.simulate(selected_child.state,game_moves)
            if self.record:
                self.played_games.append((game_moves, result))  # Ajouter les coups et le résultat à la liste des parties jouées
            selected_child.backpropagate(result)

        best_move = self.root.children[0]

        for child in self.root.children:
            if (child.wins / (child.visits+1e-4)) > (best_move.wins / (best_move.visits+1e-4)):
                best_move = child
        return best_move.state.last_move


    def simulate(self, state, game_moves=[]):
        move = state.last_move
        if not state.is_terminal((move[0]//3,move[1]//3)):
            move = random.choice(list(state.get_possible_moves()))
            if self.record:
                game_moves.append(move)
            state = state.make_move(move)
        terminal = state.is_terminal((move[0]//3,move[1]//3))
        while not terminal:
            move = random.choice(list(state.get_possible_moves()))
            if self.record:
                game_moves.append(move)
            terminal = state.make_move_self(move)
        return state.get_result()
    
    def save_played_games(self, filename):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Check if the file is empty
                writer.writerow(["Moves", "Result"])
            for moves, result in self.played_games:
                writer.writerow([moves, result])




"""
import cProfile

# Exemple
initial_state = Morpion()
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move1:", best_move)

mcts.save_played_games("parties_uttt.csv")

cProfile.run('mcts.select_move()')
"""



# Sur codingame :
"""
current_state = Morpion()

while True:
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())

    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]


    if (opponent_row, opponent_col) != (-1,-1):
        current_state.make_move_self((opponent_row, opponent_col))
    mcts = MonteCarloTreeSearch(current_state)
    best_move = mcts.select_move()
    print(str(best_move[0]),str(best_move[1]))
    

    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    current_state.make_move_self(best_move)


"""