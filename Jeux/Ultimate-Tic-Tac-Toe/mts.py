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
            if (child.wins / child.visits) > (best_move.wins / best_move.visits):
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
            board_x = self.last_move[0]//3
            board_y = self.last_move[1]//3
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


# Exemple
initial_state = GameState([[[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)])
mcts = MonteCarloTreeSearch(initial_state)
best_move = mcts.select_move()
print("Best move:", best_move)

