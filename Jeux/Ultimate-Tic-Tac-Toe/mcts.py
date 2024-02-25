import math
import random
import numpy as np

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

    def select_move(self, simulations=100):
        for _ in range(simulations):
            node = self.root
            while not node.state.is_terminal():
                if not node.children:
                    node.expand()
                node = node.select_child()
            result = self.simulate(node.state)
            node.backpropagate(result)

        best_move = self.root.children[0]
        for child in self.root.children:
            if (child.wins / (child.visits+1e-4)) > (best_move.wins / (best_move.visits+1e-4)):
                best_move = child
        return best_move.state.last_move

    def simulate(self, state):
        while not state.is_terminal():
            move = random.choice(state.get_possible_moves())
            state = state.make_move_self(move)

        return state.get_result()




class Morpion(GameState):
    def __init__(self):
        self.boards = np.array([[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)])
        self.big_boards = np.array([[0 for _ in range(3)] for _ in range(3)]) # qui a gagné chaque board
        self.player = 1
        self.last_move = None
        self.empty_all = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),
                    (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5), 
                    (0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8),
                    (3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),
                    (3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5),
                    (3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),
                    (6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),
                    (6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),
                    (6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]

    def get_possible_moves(self):
        if self.last_move != None:
            board_x = self.last_move[0]%3
            board_y = self.last_move[1]%3
            empty=[]
            for i in range(3):
                for j in range(3):
                    if (i+3*board_x,j+3*board_y) in self.empty_all:
                        empty.append((i+3*board_x,j+3*board_y))
            if empty!=[]:
                return empty
        return self.empty_all

    def make_move(self, move):
        i, j = move
        new_state = Morpion()
        new_state.boards = self.boards.copy()
        new_state.boards[i//3][j//3][i%3][j%3] = self.player
        new_state.empty_all=self.empty_all.copy()
        new_state.empty_all.remove((i,j))

        if np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3],axis=1).any() or np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3].T,axis=1).any() or (new_state.boards[i//3,j//3,0,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][2,2]==self.player) or (new_state.boards[i//3,j//3][2,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][0][2]==self.player):
            new_state.big_boards[i//3,j//3] = self.player
            for x in range(3):
                for y in range(3):
                    if (x+3*(i//3),y+3*(j//3)) in new_state.empty_all:
                        new_state.empty_all.remove((x+3*(i//3),y+3*(j//3)))

        new_state.player = -self.player
        new_state.last_move = move
        return new_state
    
    def make_move_self(self, move):
        i, j = move
        self.boards[i//3][j//3][i%3][j%3] = self.player
        self.empty_all.remove((i,j))

        if np.all([self.player,self.player,self.player] == self.boards[i//3,j//3],axis=1).any() or np.all([self.player,self.player,self.player] == self.boards[i//3,j//3].T,axis=1).any() or (self.boards[i//3,j//3,0,0]==self.player and self.boards[i//3,j//3][1,1]==self.player and self.boards[i//3,j//3][2,2]==self.player) or (self.boards[i//3,j//3][2,0]==self.player and self.boards[i//3,j//3][1,1]==self.player and self.boards[i//3,j//3][0][2]==self.player):
            self.big_boards[i//3,j//3] = self.player
            for x in range(3):
                for y in range(3):
                    if (x+3*(i//3),y+3*(j//3)) in self.empty_all:
                        self.empty_all.remove((x+3*(i//3),y+3*(j//3)))

        self.player = -self.player
        self.last_move = move
        #return self

    def is_terminal(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[1,1,1]) or np.all(np.fliplr(self.big_boards).diagonal()==[1,1,1]) or np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[-1,-1,-1]) or np.all(np.fliplr(self.big_boards).diagonal()==[-1,-1,-1]) or self.empty_all==[]:
            return True
        else:
            return False


    def get_result(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[1,1,1]) or np.all(np.fliplr(self.big_boards).diagonal()==[1,1,1]):
            return 1
        elif np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[-1,-1,-1]) or np.all(np.fliplr(self.big_boards).diagonal()==[-1,-1,-1]):
            return -1
        return 0





import cProfile

# Exemple 1
initial_state = Morpion()
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move:", best_move)

cProfile.run('mcts.select_move()')






# Sur codingame :
"""
import math
import random
import numpy as np

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

    def select_move(self, simulations=1):
        for _ in range(simulations):
            node = self.root
            while not node.state.is_terminal():
                if not node.children:
                    node.expand()
                node = node.select_child()
            result = self.simulate(node.state)
            node.backpropagate(result)

        best_move = self.root.children[0]
        for child in self.root.children:
            if (child.wins / (child.visits+1e-4)) > (best_move.wins / (best_move.visits+1e-4)):
                best_move = child
        return best_move.state.last_move

    def simulate(self, state):
        while not state.is_terminal():
            move = random.choice(state.get_possible_moves())
            state = state.make_move_self(move)

        return state.get_result()




class Morpion(GameState):
    def __init__(self):
        self.boards = np.array([[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)])
        self.big_boards = np.array([[0 for _ in range(3)] for _ in range(3)]) # qui a gagné chaque board
        self.player = 1
        self.last_move = None
        self.empty_all = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),
                    (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5), 
                    (0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8),
                    (3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),
                    (3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5),
                    (3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),
                    (6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),
                    (6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),
                    (6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]

    def get_possible_moves(self):
        if self.last_move != None:
            board_x = self.last_move[0]%3
            board_y = self.last_move[1]%3
            empty=[]
            for i in range(3):
                for j in range(3):
                    if (i+3*board_x,j+3*board_y) in self.empty_all:
                        empty.append((i+3*board_x,j+3*board_y))
            if empty!=[]:
                return empty
        return self.empty_all

    def make_move(self, move):
        i, j = move
        new_state = Morpion()
        new_state.boards = self.boards.copy()
        new_state.boards[i//3][j//3][i%3][j%3] = self.player
        new_state.empty_all=self.empty_all.copy()
        new_state.empty_all.remove((i,j))

        if np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3],axis=1).any() or np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3].T,axis=1).any() or (new_state.boards[i//3,j//3,0,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][2,2]==self.player) or (new_state.boards[i//3,j//3][2,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][0][2]==self.player):
            new_state.big_boards[i//3,j//3] = self.player
            for x in range(3):
                for y in range(3):
                    if (x+3*(i//3),y+3*(j//3)) in new_state.empty_all:
                        new_state.empty_all.remove((x+3*(i//3),y+3*(j//3)))

        new_state.player = -self.player
        new_state.last_move = move
        return new_state
    
    def make_move_self(self, move):
        i, j = move
        self.boards[i//3][j//3][i%3][j%3] = self.player
        self.empty_all.remove((i,j))

        if np.all([self.player,self.player,self.player] == self.boards[i//3,j//3],axis=1).any() or np.all([self.player,self.player,self.player] == self.boards[i//3,j//3].T,axis=1).any() or (self.boards[i//3,j//3,0,0]==self.player and self.boards[i//3,j//3][1,1]==self.player and self.boards[i//3,j//3][2,2]==self.player) or (self.boards[i//3,j//3][2,0]==self.player and self.boards[i//3,j//3][1,1]==self.player and self.boards[i//3,j//3][0][2]==self.player):
            self.big_boards[i//3,j//3] = self.player
            for x in range(3):
                for y in range(3):
                    if (x+3*(i//3),y+3*(j//3)) in self.empty_all:
                        self.empty_all.remove((x+3*(i//3),y+3*(j//3)))

        self.player = -self.player
        self.last_move = move
        #return self

    def is_terminal(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[1,1,1]) or np.all(np.fliplr(self.big_boards).diagonal()==[1,1,1]) or np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[-1,-1,-1]) or np.all(np.fliplr(self.big_boards).diagonal()==[-1,-1,-1]) or self.empty_all==[]:
            return True
        else:
            return False


    def get_result(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[1,1,1]) or np.all(np.fliplr(self.big_boards).diagonal()==[1,1,1]):
            return 1
        elif np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or np.all(np.diagonal(self.big_boards)==[-1,-1,-1]) or np.all(np.fliplr(self.big_boards).diagonal()==[-1,-1,-1]):
            return -1
        return 0


current_state = Morpion()

while True:
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())

    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]


    if (opponent_row, opponent_col) != (-1,-1):
        current_state = current_state.make_move((opponent_row, opponent_col))
     
    mcts = MonteCarloTreeSearch(current_state)
    best_move = mcts.select_move()
    print(str(best_move[0]),str(best_move[1]))
    current_state = current_state.make_move(best_move)

    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

"""