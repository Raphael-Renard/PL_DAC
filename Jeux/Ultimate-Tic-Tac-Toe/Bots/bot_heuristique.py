import random
import numpy as np

class Heuristique():
    def __init__(self, state):
        self.state = state

    def OOorXX(self,board_x,board_y):
        if np.all(self.state.boards[board_x][board_y][0]==[self.state.player,self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[0,self.state.player,self.state.player]) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][2][0]==self.state.player and self.state.boards[board_x][board_y][0][2]==0):
            return(0+3*board_x,2+3*board_y)
        elif np.all(self.state.boards[board_x][board_y][0]==[0,self.state.player,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[0,self.state.player,self.state.player]) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][2][2]==self.state.player and self.state.boards[board_x][board_y][0][0]==0):
            return(0+3*board_x,0+3*board_y)
        elif np.all(self.state.boards[board_x][board_y][1]==[self.state.player,self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[self.state.player,0,self.state.player]):
            return(1+3*board_x, 2+3*board_y)
        elif np.all(self.state.boards[board_x][board_y][1]==[0,self.state.player,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[self.state.player,0,self.state.player]):
            return(1+3*board_x, 0+3*board_y)
        elif np.all( self.state.boards[board_x][board_y][2]==[self.state.player,self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[self.state.player,self.state.player,0]) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][0][0]==self.state.player and self.state.boards[board_x][board_y][2][2]==0):
            return(2+3*board_x, 2+3*board_y)

        elif  np.all(self.state.boards[board_x][board_y][2]==[0,self.state.player,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[self.state.player,self.state.player,0]) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][0][2]==self.state.player and self.state.boards[board_x][board_y][2][0] == 0):
            return(2+3*board_x, 0+3*board_y)
                
        elif np.all(self.state.boards[board_x][board_y][0]==[self.state.player,0,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[0,self.state.player,self.state.player]) :
            return(0+3*board_x, 1+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][2]==[self.state.player,0,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[self.state.player,self.state.player,0]):
            return(2+3*board_x, 1+3*board_y)
        
        elif np.all(self.state.boards[board_x][board_y][1]==[self.state.player,0,self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[self.state.player,0,self.state.player]) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][2][0]==0 and self.state.boards[board_x][board_y][0][2]==self.state.player) or (self.state.boards[board_x][board_y][1][1]==self.state.player and self.state.boards[board_x][board_y][2][2]==0 and self.state.boards[board_x][board_y][0][0]==self.state.player):
            return(1+3*board_x, 1+3*board_y)

        
        elif np.all(self.state.boards[board_x][board_y][0]==[-self.state.player,-self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[0,-self.state.player,-self.state.player]) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][2][0]==-self.state.player and self.state.boards[board_x][board_y][0][2]==0) :
            return(0+3*board_x, 2+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][0]==[0,-self.state.player,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[0,-self.state.player,-self.state.player]) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][2][2]==-self.state.player and self.state.boards[board_x][board_y][0][0]==0):
            return(0+3*board_x, 0+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][1]==[-self.state.player,-self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[-self.state.player,0,-self.state.player]):
            return(1+3*board_x, 2+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][1]==[0,-self.state.player,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[-self.state.player,0,-self.state.player]):
            return(1+3*board_x, 0+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][2]==[-self.state.player,-self.state.player,0]) or np.all(self.state.boards[board_x][board_y][:,2]==[-self.state.player,-self.state.player,0]) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][0][0]==-self.state.player and self.state.boards[board_x][board_y][2][2]==0) :
            return(2+3*board_x, 2+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][2]==[0,-self.state.player,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,0]==[-self.state.player,-self.state.player,0]) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][0][2]==-self.state.player and self.state.boards[board_x][board_y][2][0]==0) :
            return(2+3*board_x, 0+3*board_y)
                
        elif np.all(self.state.boards[board_x][board_y][0]==[-self.state.player,0,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[0,-self.state.player,-self.state.player]):
            return(0+3*board_x, 1+3*board_y)

        elif np.all(self.state.boards[board_x][board_y][2]==[-self.state.player,0,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[-self.state.player,-self.state.player,0]):
            return(2+3*board_x, 1+3*board_y)
        
        elif np.all(self.state.boards[board_x][board_y][1]==[-self.state.player,0,-self.state.player]) or np.all(self.state.boards[board_x][board_y][:,1]==[-self.state.player,0,-self.state.player]) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][2][0]==0 and self.state.boards[board_x][board_y][0][2]==-self.state.player) or (self.state.boards[board_x][board_y][1][1]==-self.state.player and self.state.boards[board_x][board_y][2][2]==0 and self.state.boards[board_x][board_y][0][0]==-self.state.player):
            return(1+3*board_x, 1+3*board_y)


        else:
            return None

    def select_move(self):
        (i,j) = list(self.state.get_possible_moves())[0]
        board_x = i//3
        board_y = j//3
        coup = self.OOorXX(board_x,board_y)
        if coup is None:
            return random.choice(list(self.state.get_possible_moves()))
        else:
            return coup