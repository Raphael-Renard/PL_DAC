# Ultimate tic tac toe


import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.



boards = [[[[' ' for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
board_x = 0
board_y = 0

def OOorXX(board_x,board_y):
    if boards[board_x][board_y][0]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==[" ","X","X"] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][0]=="X"):
        print(str(0+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][0][2]="X"
        return True
    elif boards[board_x][board_y][0]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==[" ","X","X"] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]=="X"):
        print(str(0+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][0][0] = "X"
        return True

    elif boards[board_x][board_y][1]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==["X"," ","X"]:
        print(str(1+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][1][2]="X"
        return True

    elif boards[board_x][board_y][1]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==["X"," ","X"]:
        print(str(1+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][1][0] = "X"
        return True

    elif boards[board_x][board_y][2]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==["X","X"," "] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][0]=="X"):
        print(str(2+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][2][2]="X"
        return True

    elif  boards[board_x][board_y][2]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==["X","X"," "] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][2]=="X"):
        print(str(2+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][2][0] = "X"
        return True
            
    elif boards[board_x][board_y][0]==["X"," ","X"] or [row[1] for row in boards[board_x][board_y]]==[" ","X","X"] :
        print(str(0+3*board_x)+" "+ str(1+3*board_y))
        boards[board_x][board_y][0][1]="X"
        return True

    elif boards[board_x][board_y][2]==["X"," ","X"] or [row[1] for row in boards[board_x][board_y]]==["X","X"," "]:
        print(str(2+3*board_x)+" "+ str(1+3*board_y))
        boards[board_x][board_y][2][1] = "X"
        return True



    
    elif boards[board_x][board_y][0]==["O","O"," "]or [row[2] for row in boards[board_x][board_y]]==[" ","O","O"] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][0]=="O") :
        print(str(0+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][0][2]="X"
        return True

    elif boards[board_x][board_y][0]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==[" ","O","O"] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][2]=="O") :
        print(str(0+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][0][0] = "X"
        return True

    elif boards[board_x][board_y][1]==["O","O"," "] or [row[2] for row in boards[board_x][board_y]]==["O"," ","O"]:
        print(str(1+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][1][2]="X"
        return True

    elif boards[board_x][board_y][1]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==["O"," ","O"]:
        print(str(1+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][1][0] = "X"
        return True

    elif boards[board_x][board_y][2]==["O","O"," "] or [row[2] for row in boards[board_x][board_y]]==["O","O"," "] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][0][0]=="O") :
        print(str(2+3*board_x)+" "+ str(2+3*board_y))
        boards[board_x][board_y][2][2]="X"
        return True

    elif boards[board_x][board_y][2]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==["O","O"," "] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][0][2]=="O") :
        print(str(2+3*board_x)+" "+ str(0+3*board_y))
        boards[board_x][board_y][2][0] = "X"
        return True
            
    elif boards[board_x][board_y][0]==["O"," ","O"] or [row[1] for row in boards[board_x][board_y]]==[" ","O","O"]:
        print(str(0+3*board_x)+" "+ str(1+3*board_y))
        boards[board_x][board_y][0][1]="X"
        return True

    elif boards[board_x][board_y][2]==["O"," ","O"] or [row[1] for row in boards[board_x][board_y]]==["O","O"," "]:
        print(str(2+3*board_x)+" "+ str(1+3*board_y))
        boards[board_x][board_y][2][1] = "X"
        return True
    
    else:
        return False

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
        boards[x][y][opponent_row%3][opponent_col%3] = 'O'
        
        if ["O","O","O"] in boards[x][y] or [row[0] for row in boards[x][y]]==["O","O","O"] or [row[1] for row in boards[x][y]]==["O","O","O"] or [row[2] for row in boards[x][y]]==["O","O","O"] or (boards[x][y][0][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][2][2]=="O") or (boards[x][y][2][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][0][2]=="O"):
            boards[x][y] = "O"


        board_x = opponent_row%3
        board_y = opponent_col%3
        if boards[board_x][board_y]=="X" or boards[board_x][board_y]=="O" :
            for i in range(3):
                for j in range(3):
                    if  boards[i][j]!="X" and boards[i][j]!="O":
                        board_x = i
                        board_y = j
                        break


    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
        

    bo = OOorXX(board_x,board_y)
    if not(bo):
        for (x1,y1) in empty:
            if x1//3==board_x and y1//3 == board_y:
                print(str(x1)+" "+str(y1))
                boards[board_x][board_y][x1%3][y1%3] = "X"
                break


    if ["X","X","X"] in boards[board_x][board_y] or [row[0] for row in boards[board_x][board_y]]==["X","X","X"] or [row[1] for row in boards[board_x][board_y]]==["X","X","X"] or [row[2] for row in boards[board_x][board_y]]==["X","X","X"] or (boards[board_x][board_y][0][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]=="X") or (boards[board_x][board_y][2][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][2]=="X"):
        boards[board_x][board_y] = "X"

    
    
    
    
