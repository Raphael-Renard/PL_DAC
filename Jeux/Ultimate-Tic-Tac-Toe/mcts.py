import math
import random
import numpy as np
import copy
import csv


class GameState():
    def __init__(self):
        self.player = 1
        self.last_move = None

    def get_possible_moves(self):
        pass

    def make_move(self, move):
        pass

    def is_terminal(self):
        pass

    def get_result(self):
        pass


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
    def __init__(self, initial_state):
        self.root = Node(initial_state)
        self.played_games = []  # Liste pour stocker les parties jouées

    def select_move(self, simulations=10000):
        for _ in range(simulations):
            node = self.root
            game_moves = []  # Liste pour stocker les coups de chaque partie
            if not node.children:
                node.expand()
            selected_child = node.select_child()
            game_moves.append(selected_child.state.last_move)  # Ajouter le coup à la liste des coups de la partie
            result = self.simulate(selected_child.state,game_moves)
            self.played_games.append((game_moves, result))  # Ajouter les coups et le résultat à la liste des parties jouées
            selected_child.backpropagate(result)

        best_move = self.root.children[0]

        for child in self.root.children:
            if (child.wins / (child.visits+1e-4)) > (best_move.wins / (best_move.visits+1e-4)):
                best_move = child
        return best_move.state.last_move


    def simulate(self, state,game_moves=[]):
        move = state.last_move
        if not state.is_terminal((move[0]//3,move[1]//3)):
            move = random.choice(list(state.get_possible_moves()))
            game_moves.append(move)
            state = state.make_move(move)
        terminal = state.is_terminal((move[0]//3,move[1]//3))
        while not terminal:
            move = random.choice(list(state.get_possible_moves()))
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


class Morpion(GameState):
    def __init__(self,
                boards=np.array([[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
                                    [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
                                    [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]),
                empty_all={(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),
                        (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),
                        (0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8),
                        (3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),
                        (3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5),
                        (3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),
                        (6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),
                        (6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),
                        (6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)},
                empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                    [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                    [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                    [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                    [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                    [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                    [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                    [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]],
                big_boards=np.array([[0,0,0],[0,0,0],[0,0,0]])):

        GameState.__init__(self)
        self.boards = boards
        self.big_boards = big_boards # qui a gagné chaque petit board
        self.empty_boards = empty_boards # toutes les cases vides rangées par board
        self.empty_all = empty_all # toutes les cases vides

    def get_possible_moves(self):
        if self.last_move != None:
            board_x = self.last_move[0]%3
            board_y = self.last_move[1]%3
            if self.empty_boards[board_x][board_y]!=[]:
                return self.empty_boards[board_x][board_y]
        return self.empty_all



    def make_move(self, move): # crée un nouvel état
        i, j = move
        big_board_x = i // 3
        big_board_y = j // 3
        x = i % 3
        y = j % 3
        new_state = Morpion(boards=self.boards.copy(),empty_all=self.empty_all.copy(),empty_boards=copy.deepcopy(self.empty_boards),big_boards=self.big_boards.copy()) #
        new_state.boards[big_board_x, big_board_y,x,y] = self.player
        new_state.empty_all.remove((i,j))
        new_state.empty_boards[big_board_x][big_board_y].remove((i,j)) #

        if new_state.is_a_board_completed(big_board_x,big_board_y,x,y):
            new_state.big_boards[big_board_x, big_board_y] = self.player
            new_state.empty_all -= {(x + 3 * big_board_x, y + 3 * big_board_y) for x in range(3) for y in range(3)}
            new_state.empty_boards[big_board_x][big_board_y] = [] #

        new_state.player = -self.player
        new_state.last_move = move
        return new_state


    def is_terminal(self, move): #est-ce que la partie est finie
        # Check ligne et colonne
        if abs(self.big_boards[move[0]].sum()) == 3 or abs(self.big_boards[:, move[1]].sum()) == 3:
            return True

        # Check diagonale
        if move[0] + move[1] % 2 == 0:
            if abs(self.big_boards[0, 0] + self.big_boards[1, 1] + self.big_boards[2, 2]) == 3 or abs(self.big_boards[2, 0] + self.big_boards[1, 1] + self.big_boards[0, 2]) == 3:
                return True

        if not self.empty_all: # check s'il reste des coups jouables
            return True
        return False
    

    def is_a_board_completed(self, board_x,board_y,x,y): # est-ce qu'un petit board a été complété
        # Check ligne et colonne
        if abs(self.boards[board_x][board_y][x].sum()) == 3 or abs(self.boards[board_x][board_y][:, y].sum()) == 3:
            return True

        # Check diagonale
        if (x + y) % 2 == 0:
            if abs(self.boards[board_x][board_y][0, 0] + self.boards[board_x][board_y][1, 1] + self.boards[board_x][board_y][2, 2]) == 3 or abs(self.boards[board_x][board_y][2, 0] + self.boards[board_x][board_y][1, 1] + self.boards[board_x][board_y][0, 2]) == 3:
                return True
        if self.empty_boards[board_x][board_y]==[]:
            return True
        return False

    def make_move_self(self, move): # modifie l'état actuel et retourne si l'état est terminal ou pas
        i, j = move
        big_board_x = i // 3
        big_board_y = j // 3
        x = i % 3
        y = j % 3
        self.boards[big_board_x, big_board_y,x,y] = self.player
        self.empty_all.remove((i,j))
        self.empty_boards[big_board_x][big_board_y].remove((i,j)) #


        if self.is_a_board_completed(big_board_x,big_board_y,x,y):
            self.big_boards[big_board_x, big_board_y] = self.player
            self.empty_all -= {(x + 3 * big_board_x, y + 3 * big_board_y) for x in range(3) for y in range(3)}
            self.empty_boards[big_board_x][big_board_y] = [] #
            self.player = -self.player
            self.last_move = move
            return self.is_terminal((big_board_x,big_board_y))

        self.player = -self.player
        self.last_move = move
        return False


    def get_result(self):
        for i in range(3):
            if self.big_boards[i].sum() == 3*self.player or self.big_boards[:, i].sum() == 3*self.player:
                return self.player
        if  (self.big_boards[0,0]==self.player and self.big_boards[1,1]==self.player and self.big_boards[2,2]==self.player) or\
            (self.big_boards[2,0]==self.player and self.big_boards[1,1]==self.player and self.big_boards[0][2]==self.player):
            return self.player
        else:
            return 0




import cProfile

# Exemple
initial_state = Morpion()
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move1:", best_move)

mcts.save_played_games("parties_uttt.csv")

cProfile.run('mcts.select_move()')

"""
initial_state.make_move_self(best_move)
mcts = MonteCarloTreeSearch(initial_state)
print("possible moves 2",initial_state.get_possible_moves())
best_move = mcts.select_move()
print("Best move2:",best_move)
initial_state.make_move_self(best_move)
mcts = MonteCarloTreeSearch(initial_state)
print("possible moves 3",initial_state.get_possible_moves())
best_move = mcts.select_move()
print("Best move3:",best_move)
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