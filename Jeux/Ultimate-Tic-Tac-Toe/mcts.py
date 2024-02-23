import math
import random
import copy
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
                ucb_value = float('inf')  # Set a high value for unvisited nodes
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
            state = state.make_move(move)

        return state.get_result()




class Morpion(GameState):
    def __init__(self,boards):
        self.boards = boards
        self.big_boards = np.array([[0 for _ in range(3)] for _ in range(3)]) # qui a gagné chaque board
        self.player = 1
        self.last_move = None
        self.empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)], 
                    [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                    [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                    [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                    [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                    [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                    [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                    [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]]
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
            if self.empty_boards[board_x][board_y]!=[]:
                return self.empty_boards[board_x][board_y]
        return self.empty_all

    def make_move(self, move):
        i, j = move
        new_state = Morpion(self.boards.copy())
        new_state.boards[i//3][j//3][i%3][j%3] = self.player
        new_state.empty_boards=copy.deepcopy(self.empty_boards)
        new_state.empty_all=self.empty_all.copy()
        new_state.empty_boards[i//3][j//3].remove((i,j))
        new_state.empty_all.remove((i,j))

        if np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3],axis=1).any() or np.all([self.player,self.player,self.player] == new_state.boards[i//3,j//3].T,axis=1).any() or (new_state.boards[i//3,j//3,0,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][2,2]==self.player) or (new_state.boards[i//3,j//3][2,0]==self.player and new_state.boards[i//3,j//3][1,1]==self.player and new_state.boards[i//3,j//3][0][2]==self.player):
            new_state.big_boards[i//3,j//3] = self.player
            new_state.empty_boards[i//3][j//3]=[]
            for x in range(3):
                for y in range(3):
                    if (x+3*(i//3),y+3*(j//3)) in new_state.empty_all:
                        new_state.empty_all.remove((x+3*(i//3),y+3*(j//3)))

        new_state.player = -self.player
        new_state.last_move = move
        return new_state

    def is_terminal(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or (self.big_boards[0,0]==1 and self.big_boards[1,1]==1 and self.big_boards[2,2]==1) or (self.big_boards[2,0]==1 and self.big_boards[1,1]==1 and self.big_boards[0,2]==1):
            return True
        elif np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or (self.big_boards[0,0]==-1 and self.big_boards[1,1]==-1 and self.big_boards[2,2]==-1) or (self.big_boards[2,0]==-1 and self.big_boards[1,1]==-1 and self.big_boards[0,2]==-1):
            return True
        elif self.empty_all==[]:
            return True
        else:
            return False


    def get_result(self):
        if np.all([1,1,1] == self.big_boards, axis=1).any() or np.all([1,1,1] == self.big_boards.T, axis=1).any() or (self.big_boards[0,0]==1 and self.big_boards[1,1]==1 and self.big_boards[2,2]==1) or (self.big_boards[2,0]==1 and self.big_boards[1,1]==1 and self.big_boards[0,2]==1):
            return 1
        elif np.all([-1,-1,-1] == self.big_boards, axis=1).any() or np.all([-1,-1,-1] == self.big_boards.T, axis=1).any() or (self.big_boards[0,0]==-1 and self.big_boards[1,1]==-1 and self.big_boards[2,2]==-1) or (self.big_boards[2,0]==-1 and self.big_boards[1,1]==-1 and self.big_boards[0,2]==-1):
            return -1
        return 0





import cProfile


# Exemple 1
initial_state = Morpion(np.array([[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]))
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move:", best_move)

cProfile.run('mcts.select_move()')

"""
# Exemple 2
boards = np.array([[[[1, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, -1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
          [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, -1, 0], [0, 0, 0], [0, 0, 0]]]])
initial_state = Morpion(boards)
initial_state.last_move=(0,6)
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move:", best_move)
"""
















# Sur codingame :
""""
import sys
import math
import random
import copy

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
                ucb_value = float('inf')  # Set a high value for unvisited nodes
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

    def select_move(self, simulations=3):
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
            state = state.make_move(move)
        return state.get_result()

class GameState:
    def __init__(self,boards):
        self.boards = boards
        self.player = 1
        self.last_move = None

    def get_possible_moves(self):
        moves = []
        if self.last_move != None:
            board_x = self.last_move[0]%3
            board_y = self.last_move[1]%3
            if type(self.boards[board_x][board_y])==list:
                for i in range(3):
                    for j in range(3):
                        if self.boards[board_x][board_y][i][j] == 0:
                            moves.append((i+3*board_x, j+3*board_y))
            else:
                for x in range(3):
                    for y in range(3):
                        if type(self.boards[x][y])==list:
                            for i in range(3):
                                for j in range(3):
                                    if self.boards[x][y][i][j] == 0:
                                        moves.append((i+3*x, j+3*y))
        else:
            for x in range(3):
                    for y in range(3):
                        if type(self.boards[x][y])==list:
                            for i in range(3):
                                for j in range(3):
                                    if self.boards[x][y][i][j] == 0:
                                        moves.append((i+3*x, j+3*y))
        return moves

    def make_move(self, move):
        i, j = move
        new_state = GameState(copy.deepcopy(self.boards))
        new_state.boards[i//3][j//3][i%3][j%3] = self.player

        if [self.player,self.player,self.player] in new_state.boards[i//3][j//3] or [row[0] for row in new_state.boards[i//3][j//3]]==[self.player,self.player,self.player] or [row[1] for row in new_state.boards[i//3][j//3]]==[self.player,self.player,self.player] or [row[2] for row in new_state.boards[i//3][j//3]]==[self.player,self.player,self.player] or (new_state.boards[i//3][j//3][0][0]==self.player and new_state.boards[i//3][j//3][1][1]==self.player and new_state.boards[i//3][j//3][2][2]==self.player) or (new_state.boards[i//3][j//3][2][0]==self.player and new_state.boards[i//3][j//3][1][1]==self.player and new_state.boards[i//3][j//3][0][2]==self.player):
            new_state.boards[i//3][j//3] = self.player
        elif not(0 in new_state.boards[i//3][j//3][0]) and not(0 in new_state.boards[i//3][j//3][1]) and not(0 in new_state.boards[i//3][j//3][2]):
            new_state.boards[i//3][j//3] = 0

        new_state.player = -self.player
        new_state.last_move = move
        return new_state

    def is_terminal(self):
        if [1,1,1] in self.boards or [row[0] for row in self.boards]==[1,1,1] or [row[1] for row in self.boards]==[1,1,1] or [row[2] for row in self.boards]==[1,1,1] or (self.boards[0][0]==1 and self.boards[1][1]==1 and self.boards[2][2]==1) or (self.boards[2][0]==1 and self.boards[1][1]==1 and self.boards[0][2]==1):
            return True
        elif [-1,-1,-1] in self.boards or [row[0] for row in self.boards]==[-1,-1,-1] or [row[1] for row in self.boards]==[-1,-1,-1] or [row[2] for row in self.boards]==[-1,-1,-1] or (self.boards[0][0]==-1 and self.boards[1][1]==-1 and self.boards[2][2]==-1) or (self.boards[2][0]==-1 and self.boards[1][1]==-1 and self.boards[0][2]==-1):
            return True
        elif [[type(self.boards[i][j]) for j in range(3)] for i in range(3)] == [[int,int,int],[int,int,int],[int,int,int]]:
            return True
        else:
            return False


    def get_result(self):
        if [self.player,self.player,self.player] in self.boards or [row[0] for row in self.boards]==[self.player,self.player,self.player] or [row[1] for row in self.boards]==[self.player,self.player,self.player] or [row[2] for row in self.boards]==[self.player,self.player,self.player] or (self.boards[0][0]==self.player and self.boards[1][1]==self.player and self.boards[2][2]==self.player) or (self.boards[2][0]==self.player and self.boards[1][1]==self.player and self.boards[0][2]==self.player):
            return 1
        elif [-self.player,-self.player,-self.player] in self.boards or [row[0] for row in self.boards]==[-self.player,-self.player,-self.player] or [row[1] for row in self.boards]==[-self.player,-self.player,-self.player] or [row[2] for row in self.boards]==[-self.player,-self.player,-self.player] or (self.boards[0][0]==-self.player and self.boards[1][1]==-self.player and self.boards[2][2]==-self.player) or (self.boards[2][0]==-self.player and self.boards[1][1]==-self.player and self.boards[0][2]==-self.player):
            return -1
        elif [[type(self.boards[i][j]) for j in range(3)] for i in range(3)] == [[int,int,int],[int,int,int],[int,int,int]]:
            return 0


boards = [[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
board_x = 0
board_y = 0


while True:

    empty = []
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())

    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]
        empty.append((row,col))

    if (opponent_row, opponent_col)!=(-1,-1):
        x = opponent_row//3
        y = opponent_col//3
        boards[x][y][opponent_row%3][opponent_col%3] = -1
        
        if [-1,-1,-1] in boards[x][y] or [row[0] for row in boards[x][y]]==[-1,-1,-1] or [row[1] for row in boards[x][y]]==[-1,-1,-1] or [row[2] for row in boards[x][y]]==[-1,-1,-1] or (boards[x][y][0][0]==-1 and boards[x][y][1][1]==-1 and boards[x][y][2][2]==-1) or (boards[x][y][2][0]==-1 and boards[x][y][1][1]==-1 and boards[x][y][0][2]==-1):
            boards[x][y] = -1
        elif not(0 in boards[x][y][0]) and not(0 in boards[x][y][1]) and not(0 in boards[x][y][2]):
            boards[x][y] = 0

    current_state = GameState(boards)
    current_state.player=1
    current_state.last_move=(opponent_row,opponent_col)
    mcts = MonteCarloTreeSearch(current_state)
    best_move = mcts.select_move()
    print(str(best_move[0]),str(best_move[1]))
    boards[best_move[0]//3][best_move[1]//3][best_move[0]%3][best_move[1]%3]=1

    if [1,1,1] in boards[best_move[0]//3][best_move[1]//3] or [row[0] for row in boards[best_move[0]//3][best_move[1]//3]]==[1,1,1] or [row[1] for row in boards[best_move[0]//3][best_move[1]//3]]==[1,1,1] or [row[2] for row in boards[best_move[0]//3][best_move[1]//3]]==[1,1,1] or (boards[best_move[0]//3][best_move[1]//3][0][0]==1 and boards[best_move[0]//3][best_move[1]//3][1][1]==1 and boards[best_move[0]//3][best_move[1]//3][2][2]==1) or (boards[best_move[0]//3][best_move[1]//3][2][0]==1 and boards[best_move[0]//3][best_move[1]//3][1][1]==1 and boards[best_move[0]//3][best_move[1]//3][0][2]==1):
        boards[best_move[0]//3][best_move[1]//3] = 1
"""