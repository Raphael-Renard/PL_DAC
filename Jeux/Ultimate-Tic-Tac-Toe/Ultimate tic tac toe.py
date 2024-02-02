# Ultimate tic tac toe


import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


# game loop
coup=1
commence = False

board = [[' ' for _ in range(3)] for _ in range(3)]

while True:
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())

    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]

    


    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    if (opponent_row,opponent_col) != (-1,-1):
        board[opponent_row][opponent_col] = 'O'

    if board == [[" "," "," "],[" "," "," "],[" "," "," "]]: # coup 1
        print("0 0")
        board[0][0] = "X"
        coup=2
        commence = True

    if commence:
        if ((opponent_row==0 and opponent_col==1) or (opponent_row==0 and opponent_col==2) or (opponent_row==2 and opponent_col==2) or (opponent_row==1 and opponent_col==2)) and coup==2: # coup 2
            print("2 0")
            board[2][0] = "X"
            coup=3

        elif coup==2 and ((opponent_row==1 and opponent_col==0) or (opponent_row==2 and opponent_col==1) or (opponent_row==2 and opponent_col==0)):
            print("0 2")
            board[0][2] = "X"
            coup=3

        elif coup==3 and board[0]==["X"," ","X"]:
            print("0 1")
        elif coup==3 and [row[0] for row in board]==["X"," ","X"]:
            print("1 0")

        elif coup==3 and board[0][0]==" " and board[2][2]==" ":
            print("2 2")
            board[2][2] = "X"
            coup = 4
        elif coup==3 and board[0][0]==" " and board[2][2]=="O":
            print("0 2")
            board[0][2] = "X"
            coup = 4

        elif coup==4 and board[0]==["X"," ","X"]:
            print("0 1")

        elif coup==4 and [row[0] for row in board]==["X"," ","X"]:
            print("1 0")

        elif coup==4 and [row[2] for row in board]==["X"," ","X"]:
            print("1 2")

        elif coup==4 and board[2]==["X"," ","X"]:
            print("2 1")

        elif coup==4 and board[0][0]=="X" and board[1][1]==" " and board[2][2]==" ":
            print("1 1")
        
        elif coup==2 and board[1][1]=="O":
            print("2 2")
            board[2][2] = "X"
            coup=3
        elif coup==3 and board[1][1]=="O" and board[2][0]=="O":
            print("0 2")
            board[0][2]="X"
            coup=4
        elif coup==3 and board[1][1]=="O" and board[0][2]=="O":
            print("2 0")
            board[2][0]="X"
            coup=4

        elif coup==3 and board[1]==["O","O"," "]:
            print("1 2")
            board[1][2] = "X"
            coup=4
        elif coup==3 and board[1]==[" ","O","O"]:
            print("1 0")
            board[1][0] = "X"
            coup=4
        elif coup==3 and [row[1] for row in board]==["O","O"," "]:
            print("2 1")
            board[2][1] = "X"
            coup=4
        elif coup==3 and [row[1] for row in board]==[" ","O","O"]:
            print("0 1")
            board[0][1] = "X"
            coup=4
        elif coup==4 and board[0]==["X","X"," "]:
            print("0 2")
        elif coup==4 and board[2]==[" ","X","X"]:
            print("2 0")
        elif coup==4 and [row[2] for row in board]==[" ","X","X"]:
            print("0 2")
        elif coup==4 and [row[0] for row in board]==["X","X"," "]:
            print("2 0")
        
        elif coup==4 and (board == [["X"," ","O"],["O","O","X"],[" "," ","X"]] or board == [["X","X","O"],[" ","O"," "],[" ","O","X"]]):
            print("2 0")
            board[2][0]="X"
            coup=5
        
        elif coup==4 and (board == [["X"," "," "],["X","O","O"],["O"," ","X"]] or board == [["X","O"," "],[" ","O"," "],["O","X","X"]]):
            print("0 2")
            board[0][2]="X"
            coup=5
        elif coup==5:
            print(str(row)+" "+str(col))
    
    else :
        if coup==1:
            if board[1][1]=="O":
                print("0 0")
                coup = 2
                board[0][0] ="X"
            else:
                print("1 1")
                board[1][1]="X"
                coup=2
        else:

            if board[0]==["O","O"," "] or board[0]==["X","X"," "] or [row[2] for row in board]==[" ","X","X"] or [row[2] for row in board]==[" ","O","O"] or (board[1][1]==board[2][0]):
                print("0 2")
                board[0][2]="X"

            elif board[0]==[" ","O","O"] or board[0]==[" ","X","X"] or [row[0] for row in board]==[" ","X","X"] or [row[0] for row in board]==[" ","O","O"] or (board[1][1]==board[2][2]):
                print("0 0")
                board[0][0] = "X"

            elif board[1]==["O","O"," "] or board[1]==["X","X"," "] or [row[2] for row in board]==["X"," ","X"] or [row[2] for row in board]==["O"," ","O"]:
                print("1 2")
                board[1][2]="X"

            elif board[1]==[" ","O","O"] or board[1]==[" ","X","X"] or [row[0] for row in board]==["X"," ","X"] or [row[0] for row in board]==["O"," ","O"]:
                print("1 0")
                board[1][0] = "X"
            
            elif board[2]==["O","O"," "] or board[2]==["X","X"," "] or [row[2] for row in board]==["X","X"," "] or [row[2] for row in board]==["O","O"," "] or (board[1][1]==board[0][0]):
                print("2 2")
                board[2][2]="X"

            elif board[2]==[" ","O","O"] or board[2]==[" ","X","X"] or [row[0] for row in board]==["X","X"," "] or [row[0] for row in board]==["O","O"," "] or (board[1][1]==board[0][2]):
                print("2 0")
                board[2][0] = "X"
            
            elif board[0]==["O"," ","O"] or board[0]==["X"," ","X"] or [row[1] for row in board]==[" ","X","X"] or [row[1] for row in board]==[" ","O","O"]:
                print("0 1")
                board[0][1]="X"

            elif board[2]==["O"," ","O"] or board[2]==["X"," ","X"] or [row[1] for row in board]==["X","X"," "] or [row[1] for row in board]==["O","O"," "]:
                print("2 1")
                board[2][1] = "X"
            else:
                print(str(row)+" "+str(col))
                board[row][col]="X"
            





        
