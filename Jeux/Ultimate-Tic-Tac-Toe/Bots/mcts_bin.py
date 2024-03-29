import math
import random
import numpy as np
import copy
import sys
from array import array
import pstats

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

    def select_move(self, simulations=1000):
        for _ in range(simulations):
            node = self.root
            if not node.children:
                node.expand()
            selected_child = node.select_child()
            result = self.simulate(selected_child.state)
            selected_child.backpropagate(result)

        best_move = self.root.children[0]

        for child in self.root.children:
            if (child.wins / (child.visits+1e-4)) > (best_move.wins / (best_move.visits+1e-4)):
                best_move = child
        return best_move.state.last_move


    def simulate(self, state):
        move = state.last_move
        if not state.is_terminal((move[0]//3,move[1]//3)):
            move = random.choice(list(state.get_possible_moves()))
            state = state.make_move(move)
        terminal = state.is_terminal((move[0]//3,move[1]//3))
        while not terminal:
            move = random.choice(list(state.get_possible_moves()))
            terminal = state.make_move_self(move)
        return state.get_result()



## Représentation bit d'un board 9x9 : 
##  codage pour un board : 0b_XY_XY_XY_XY_XY_XY_XY_XY_XY , x j1, y j2, 0 ou 1. 
## (bit le plus à droite case 0, le plus à gauche case 9)
## Grille non valide : si X et Y = 1 pour même case.
## Masquage : 
##  Joueur : joueur 1 : 0b01 , joueur 2: 0b10
##  (board >> 2*(j+3*i)) & 0b11 donne l'état de la case (i,j) (0b01 si on veut savoir pour le j1, 0b10 pour le joueur 2)
## pour l'indexation 0 à 9 des cases : board >> 2*i donne l'état de la case
## Jouer un coup sur une case i : board = board | joueur << 2*i (joueur 0b10 ou 0b01)

# Hashage des coups possibles : pour chaque config du board, on pré-caclule les coups possibles 
POSSIBLE_MOVES = [[ i for i in range(9) if 3 & (3-(x >> 2*i)) == 3] for x in range(2**18) ]
# Masques pour détecter config gagnantes (que pour joueur 1, pour le joueur 2 il suffit de *2 le masque)
WIN_MASKS = [
    # Lignes
    0b_01_01_01_00_00_00_00_00_00,
    0b_00_00_00_01_01_01_00_00_00,
    0b_00_00_00_00_00_00_01_01_01,
    #Colonnes
    0b_01_00_00_01_00_00_01_00_00,
    0b_00_01_00_00_01_00_00_01_00,
    0b_00_00_01_00_00_01_00_00_01,
    #Diagonales
    0b_01_00_00_00_01_00_00_00_01,
    0b_00_00_01_00_01_00_01_00_00,
]

def print_board(b):
    # Pour debugage, affichage du board
    s = ""
    for i in range(3):
        for j in range(3):
            if (b >> 2*(j+3*i) & 0b10) != 0:
                s+="X "
            elif (b >> 2*(j+3*i) & 0b01) != 0:
                s+="O "
            else: 
                s+="  "
        s+="\n"
    print(s)


def test_win_bin(board,p):
    ## Test victoire du joueur p (1 ou 2)
    for m in WIN_MASKS:
        if (m * p  & board) == (m*p):
            return True
    return False
# Hashage pour savoir si un état est gagnant (et qui j1->1, j2-> -1), n'indique pas si état terminal
WIN_STATE = [1 if test_win_bin(x,1) else -1 if test_win_bin(x,2) else 0 for x in range(2**18)]
# Hashage pour état terminal (gagnant ou tout rempli)
TERMINAL_STATE = [WIN_STATE[x]!=0 or len(POSSIBLE_MOVES[x])==0 for x in range(2**18) ]

class MorpionBin(GameState):
    def __init__(self,boards=None,big_board=None,last_move=None,player=None):
        # boards sera un tableau d'entiers, chaque entier code pour l'état d'une grille
        self.boards = boards
        # big_board est un entier qui code l'état de la super-grille
        self.big_board = big_board
        # last_move couple (b,i) : b numéro du board, i numéro de la case (0 à 9)
        self.last_move = last_move
        # player 1 ou 2
        self.player = player
        if boards is None:
            self.boards = array('L',[0b0 for _ in range(9)])
            self.big_board = 0b0
            self.last_move = None
            self.player = 1 

    def copy(self):
        return MorpionBin(copy.deepcopy(self.boards),self.big_board,self.last_move,self.player)

    def get_possible_moves(self):
        if self.last_move is not None and not TERMINAL_STATE[self.boards[self.last_move[1]]]:
            # Si c'est pas le dernier coup et si un coup est possible dans le board
            moves = POSSIBLE_MOVES[self.boards[self.last_move[1]]]
            return [(self.last_move[1],i) for i in moves]
        # Sinon on renvoit tout les coups possibles à travers tous les boards non finis
        return [(b,i) for b in range(9) if not TERMINAL_STATE[self.boards[b]] for i in POSSIBLE_MOVES[self.boards[b]]]
    
    def make_move(self, move): # crée un nouvel état
        new_state = self.copy()
        new_state.make_move_self(move)
        return new_state
        
    def make_move_self(self, move): # modifie l'état actuel et retourne si l'état est terminal ou pas
        b, i = move
        # On joue le coup en case i du board b
        self.boards[b] = self.boards[b] | self.player << 2*i 
        if WIN_STATE[self.boards[b]]!=0:
            #si board gagnant pour le joueur, maj du big_board
            self.big_board = self.big_board | self.player << 2*b
        #on change de joueur
        self.player = self.player ^ 0b11
        self.last_move = b,i
        return self.is_terminal()    
        
    def is_terminal(self,dummy1=None,dummy2=None): 
        #est-ce que la partie est finie
        # dummy1 et 2 servent à rien, pour rester compatible
        if WIN_STATE[self.big_board]!=0:
            return True
        for b in range(9):
            if not TERMINAL_STATE[self.boards[b]]:
                return False 
            # Si pas de gagnant sur le board et des coups sont possibles
        return True

    def get_result(self):
        return WIN_STATE[self.big_board]


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
state_bin = MorpionBin()
state_old = Morpion()
mcts_bin = MonteCarloTreeSearch(state_bin)
mcts_old = MonteCarloTreeSearch(state_old)


#%timeit mcts_bin.select_move(simulations=10000)
#%timeit mcts_old.select_move(simulations=10000)         

cProfile.run('mcts_old.select_move(simulations=10000)','statsold')
cProfile.run('mcts_bin.select_move(simulations=10000)','statsbin')

pold = pstats.Stats("statsold")
pbin =  pstats.Stats("statsbin")

pold.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
pbin.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
