import numpy as np
import copy


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




class Morpion(GameState):
    def __init__(self,
                boards=np.zeros((3, 3, 3, 3), dtype=int),
                big_boards=np.zeros((3, 3), dtype=int),
                empty_all={(i, j) for i in range(9) for j in range(9)},
                empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                    [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                    [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                    [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                    [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                    [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                    [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                    [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]]):

        GameState.__init__(self)
        self.boards = boards
        self.big_boards = big_boards # qui a gagné chaque petit board
        self.empty_boards = empty_boards # toutes les cases vides rangées par board
        self.empty_all = empty_all # toutes les cases vides
        
        self.state_size = len(empty_all) * 2
        self.action_size = len(empty_all)

    def get_possible_moves(self):
        if self.last_move != None:
            board_x = self.last_move[0]%3
            board_y = self.last_move[1]%3
            if self.empty_boards[board_x][board_y]!=[]:
                return self.empty_boards[board_x][board_y]
        return list(self.empty_all)



    def make_move(self, move): # crée un nouvel état
        i, j = move
        big_board_x = i // 3
        big_board_y = j // 3
        x = i % 3
        y = j % 3
        new_state = Morpion(boards=self.boards.copy(),
                            empty_all=self.empty_all.copy(),
                            empty_boards=copy.deepcopy(self.empty_boards),
                            big_boards=self.big_boards.copy())
        new_state.boards[big_board_x, big_board_y,x,y] = self.player
        new_state.empty_all.remove((i,j))
        new_state.empty_boards[big_board_x][big_board_y].remove((i,j))

        if new_state.is_a_board_completed(big_board_x,big_board_y,x,y):
            new_state.big_boards[big_board_x, big_board_y] = self.player
            new_state.empty_all -= {(x + 3 * big_board_x, y + 3 * big_board_y) for x in range(3) for y in range(3)}
            new_state.empty_boards[big_board_x][big_board_y] = []

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
        # x,y coup joué (entre 0 et 2)
        # board_x,board_y board dans lequel on a joué

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
    

    ### DQN
        
    def small_board_won_reward(self, board_x,board_y,x,y):
        # x,y coup joué (entre 0 et 2)
        # board_x,board_y board dans lequel on a joué

        if abs(self.boards[board_x][board_y][x].sum())==3 or abs(self.boards[board_x][board_y][:, y].sum()) == 3:
            return self.player

        if (x + y) % 2 == 0:
            if abs(self.boards[board_x][board_y][0, 0] + self.boards[board_x][board_y][1, 1] + self.boards[board_x][board_y][2, 2]) == 3 or abs(self.boards[board_x][board_y][2, 0] + self.boards[board_x][board_y][1, 1] + self.boards[board_x][board_y][0, 2]) == 3:
                return self.player
        return 0
    
    def calculate_reward(self,action):
        board_x, board_y = action[0]//3,action[1]//3

        if self.is_terminal((board_x, board_y)):
            return -self.get_result()*10 - 0.5*self.player
        else:
            return -self.small_board_won_reward(board_x,board_y,action[0]%3,action[1]%3)*0.5 # reward de 0.5 si on gagne un petit board, -0.5 si on en perd
    
    def step(self, action): # utilise representation en 3 channels 3x3
        if action not in self.get_possible_moves():
            reward=-1000
            done = True
            return None, reward, done
        self.make_move_self(action)
        reward = self.calculate_reward(action) 
        done = self.is_terminal((action[0]//3,action[1]//3))
        if done:
            return None, reward, done
        (i,j) = self.get_possible_moves()[0]
        return self.get_grid(i,j), reward, done
    
    

    def step2(self, action): # utilise representation de toute la grille en un vecteur de taille 81
        if action not in self.get_possible_moves():
            reward=-1000
            done = True
            return None, reward, done
        self.make_move_self(action)
        reward = self.calculate_reward(action) 
        done = self.is_terminal((action[0]//3,action[1]//3))
        if done:
            return None, reward*100, done
        return np.reshape(self.boards,(1,81)), reward, done
    

    def get_grid(self,i,j): # state representation : the small grid
        coord_board_x = i//3
        coord_board_y = j//3
        channel1 = (self.boards[coord_board_x,coord_board_y]==1) #player 1
        channel2 = (self.boards[coord_board_x,coord_board_y]==-1) #player 2
        channel3 = (self.boards[coord_board_x,coord_board_y]==0) #empty
        
        return (channel1,channel2,channel3), coord_board_x, coord_board_y
    
    def get_grid2(self,i,j): # state representation : the small grid in one vector of size 9
        coord_board_x = i//3
        coord_board_y = j//3
        return np.reshape(self.boards[coord_board_x,coord_board_y],(1,9)), coord_board_x, coord_board_y
 
    
    def reset(self):
        self.boards=np.zeros((3, 3, 3, 3), dtype=int)
        self.big_boards=np.zeros((3, 3), dtype=int)
        self.empty_all={(i, j) for i in range(9) for j in range(9)}
        self.empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                    [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                    [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                    [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                    [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                    [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                    [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                    [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]] 
        (i,j) = self.get_possible_moves()[0] #
        state,board_x,board_y = self.get_grid(i,j)
        return state,board_x,board_y
