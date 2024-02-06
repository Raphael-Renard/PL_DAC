import sys
import time

def game(conn1, conn2):

    boards = [[[[' ' for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
    board_x = 0
    board_y = 0
    stop = False

    conn1.send("-1 -1") # init opponent_row, opponent_col

    conn1.send("81") # init valid_action_count

    for i in range(9): #init row, col
        for j in range(9):
            conn1.send(str(i)+" "+str(j))

    while not(stop):
        choice = False
        
        opponent_row, opponent_col = [int(i) for i in conn1.recv().split()] 
        conn2.send(str(opponent_row)+" " +str(opponent_col))

        empty = []
        for x in range(3):
            for y in range(3):
                for i in range(3):
                    for j in range(3):
                        if len(boards[x][y]>1):
                            if boards[x][y][i][j] ==  " ":
                                empty.append((i+3*x,j+3*y))

        valid_action_count = len(empty)
        conn2.send(str(valid_action_count))
        print("yolo")
        for e1,e2 in empty:
            conn2.send(str(e1)+" "+str(e2))


        if (opponent_row, opponent_col) in empty:
            x = opponent_row//3
            y = opponent_col//3
            boards[x][y][opponent_row%3][opponent_col%3] = 'O'
            
            if ["O","O","O"] in boards[x][y] or [row[0] for row in boards[x][y]]==["O","O","O"] or [row[1] for row in boards[x][y]]==["O","O","O"] or [row[2] for row in boards[x][y]]==["O","O","O"] or (boards[x][y][0][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][2][2]=="O") or (boards[x][y][2][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][0][2]=="O"):
                boards[x][y] = "O"
            elif empty==[]:
                boards[x][y] = "N"


            board_x = opponent_row%3
            board_y = opponent_col%3

            if boards[board_x][board_y]=="X" or boards[board_x][board_y]=="O" or  boards[board_x][board_y]=="N":
                choice = True

            player_row, player_col = [int(i) for i in conn2.recv().split()]
            if (player_row, player_col) in empty:
                if player_row//3==board_x and player_col//3 == board_y and not(choice):
                    boards[board_x][board_y][player_row%3][player_col%3] = "X"
                elif choice and boards[player_row//3][player_col//3]!="N" and  boards[player_row//3][player_col//3]!="O" and  boards[player_row//3][player_col//3]!="X":
                    boards[board_x][board_y][player_row%3][player_col%3] = "X"
                else:
                    print("coup invalide")
                    stop = True


        if ["X","X","X"] in boards[board_x][board_y] or [row[0] for row in boards[board_x][board_y]]==["X","X","X"] or [row[1] for row in boards[board_x][board_y]]==["X","X","X"] or [row[2] for row in boards[board_x][board_y]]==["X","X","X"] or (boards[board_x][board_y][0][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]=="X") or (boards[board_x][board_y][2][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][2]=="X"):
            boards[board_x][board_y] = "X"

        if ["X","X","X"] in boards or [row[0] for row in boards]==["X","X","X"] or [row[1] for row in boards]==["X","X","X"] or [row[2] for row in boards]==["X","X","X"] or (boards[0][0]=="X" and boards[1][1]=="X" and boards[2][2]=="X") or (boards[2][0]=="X" and boards[1][1]=="X" and boards[0][2]=="X"):
            print("player wins")
            stop = True
        elif ["O","O","O"] in boards or [row[0] for row in boards]==["O","O","O"] or [row[1] for row in boards]==["O","O","O"] or [row[2] for row in boards]==["O","O","O"] or (boards[0][0]=="O" and boards[1][1]=="O" and boards[2][2]=="O") or (boards[2][0]=="O" and boards[1][1]=="O" and boards[0][2]=="O"):
            print("opponent wins")
            stop = True
        elif [[len(boards[i][j]) for j in range(3)] for i in range(3)] == [[1,1,1],[1,1,1],[1,1,1]]:
            print("match nul")
            stop = True

            




def bot(conn_in, conn_out):
    boards = [[[[' ' for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
    board_x = 0
    board_y = 0
    def OOorXX(board_x,board_y):
        if boards[board_x][board_y][0]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==[" ","X","X"] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][0]=="X" and boards[board_x][board_y][0][2]==" "):
            conn_in.send(str(0+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][0][2]="X"
            return True
        elif boards[board_x][board_y][0]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==[" ","X","X"] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]=="X" and boards[board_x][board_y][0][0]==" "):
            conn_in.send(str(0+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][0][0] = "X"
            return True
        elif boards[board_x][board_y][1]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==["X"," ","X"]:
            conn_in.send(str(1+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][1][2]="X"
            return True
        elif boards[board_x][board_y][1]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==["X"," ","X"]:
            conn_in.send(str(1+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][1][0] = "X"
            return True
        elif boards[board_x][board_y][2]==["X","X"," "] or [row[2] for row in boards[board_x][board_y]]==["X","X"," "] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][0]=="X" and boards[board_x][board_y][2][2]==" "):
            conn_in.send(str(2+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][2][2]="X"
            return True
        elif  boards[board_x][board_y][2]==[" ","X","X"] or [row[0] for row in boards[board_x][board_y]]==["X","X"," "] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][2]=="X" and boards[board_x][board_y][2][0] == " "):
            conn_in.send(str(2+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][2][0] = "X"
            return True
        elif boards[board_x][board_y][0]==["X"," ","X"] or [row[1] for row in boards[board_x][board_y]]==[" ","X","X"] :
            conn_in.send(str(0+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][0][1]="X"
            return True
        elif boards[board_x][board_y][2]==["X"," ","X"] or [row[1] for row in boards[board_x][board_y]]==["X","X"," "]:
            conn_in.send(str(2+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][2][1] = "X"
            return True
        elif boards[board_x][board_y][1]==["X"," ","X"] or [row[1] for row in boards[board_x][board_y]]==["X"," ","X"] or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][0]==" " and boards[board_x][board_y][0][2]=="X") or (boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]==" " and boards[board_x][board_y][0][0]=="X"):
            conn_in.send(str(1+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][1][1] = "X"
            return True
        elif boards[board_x][board_y][0]==["O","O"," "]or [row[2] for row in boards[board_x][board_y]]==[" ","O","O"] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][0]=="O" and boards[board_x][board_y][0][2]==" ") :
            conn_in.send(str(0+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][0][2]="X"
            return True
        elif boards[board_x][board_y][0]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==[" ","O","O"] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][2]=="O" and boards[board_x][board_y][0][0]==" "):
            conn_in.send(str(0+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][0][0] = "X"
            return True
        elif boards[board_x][board_y][1]==["O","O"," "] or [row[2] for row in boards[board_x][board_y]]==["O"," ","O"]:
            conn_in.send(str(1+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][1][2]="X"
            return True
        elif boards[board_x][board_y][1]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==["O"," ","O"]:
            conn_in.send(str(1+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][1][0] = "X"
            return True
        elif boards[board_x][board_y][2]==["O","O"," "] or [row[2] for row in boards[board_x][board_y]]==["O","O"," "] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][0][0]=="O" and boards[board_x][board_y][2][2]==" ") :
            conn_in.send(str(2+3*board_x)+" "+ str(2+3*board_y))
            boards[board_x][board_y][2][2]="X"
            return True
        elif boards[board_x][board_y][2]==[" ","O","O"] or [row[0] for row in boards[board_x][board_y]]==["O","O"," "] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][0][2]=="O" and boards[board_x][board_y][2][0]==" ") :
            conn_in.send(str(2+3*board_x)+" "+ str(0+3*board_y))
            boards[board_x][board_y][2][0] = "X"
            return True
        elif boards[board_x][board_y][0]==["O"," ","O"] or [row[1] for row in boards[board_x][board_y]]==[" ","O","O"]:
            conn_in.send(str(0+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][0][1]="X"
            return True
        elif boards[board_x][board_y][2]==["O"," ","O"] or [row[1] for row in boards[board_x][board_y]]==["O","O"," "]:
            conn_in.send(str(2+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][2][1] = "X"
            return True
        elif boards[board_x][board_y][1]==["O"," ","O"] or [row[1] for row in boards[board_x][board_y]]==["O"," ","O"] or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][0]==" " and boards[board_x][board_y][0][2]=="O") or (boards[board_x][board_y][1][1]=="O" and boards[board_x][board_y][2][2]==" " and boards[board_x][board_y][0][0]=="O"):
            conn_in.send(str(1+3*board_x)+" "+ str(1+3*board_y))
            boards[board_x][board_y][1][1] = "X"
            return True
        else:
            return False


    while True:
        print("joue1")

        empty = []
        opponent_row, opponent_col = [int(i) for i in conn_out.recv().split()]

        print("joue",opponent_row,opponent_col)
        valid_action_count = int(conn_out.recv())
        print("vaa",valid_action_count)

        for i in range(valid_action_count):
            row, col = [int(j) for j in conn_out.recv().split()]
            empty.append((row,col))


        if (opponent_row, opponent_col)!=(-1,-1):
            x = opponent_row//3
            y = opponent_col//3
            boards[x][y][opponent_row%3][opponent_col%3] = 'O'
            
            if ["O","O","O"] in boards[x][y] or [row[0] for row in boards[x][y]]==["O","O","O"] or [row[1] for row in boards[x][y]]==["O","O","O"] or [row[2] for row in boards[x][y]]==["O","O","O"] or (boards[x][y][0][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][2][2]=="O") or (boards[x][y][2][0]=="O" and boards[x][y][1][1]=="O" and boards[x][y][0][2]=="O"):
                boards[x][y] = "O"
            elif empty==[]:
                boards[x][y] = "N"


            board_x = opponent_row%3
            board_y = opponent_col%3

            if boards[board_x][board_y]=="X" or boards[board_x][board_y]=="O" or  boards[board_x][board_y]=="N":
                for i in range(3):
                    for j in range(3):
                        if  boards[i][j]!="X" and boards[i][j]!="O" and boards[i][j]!="N":
                            board_x = i
                            board_y = j
                            break
            

        bo = OOorXX(board_x,board_y)
        if not(bo):
            for (x1,y1) in empty:
                if x1//3==board_x and y1//3 == board_y:
                    conn_in.send(str(x1)+" "+str(y1))
                    boards[board_x][board_y][x1%3][y1%3] = "X"
                    break


        if ["X","X","X"] in boards[board_x][board_y] or [row[0] for row in boards[board_x][board_y]]==["X","X","X"] or [row[1] for row in boards[board_x][board_y]]==["X","X","X"] or [row[2] for row in boards[board_x][board_y]]==["X","X","X"] or (boards[board_x][board_y][0][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][2][2]=="X") or (boards[board_x][board_y][2][0]=="X" and boards[board_x][board_y][1][1]=="X" and boards[board_x][board_y][0][2]=="X"):
            boards[board_x][board_y] = "X"




from multiprocessing import Process, Pipe

if __name__ == '__main__':
    parent_conn_game1, child_conn_game1 = Pipe()
    parent_conn_game2, child_conn_game2 = Pipe()

    p_game = Process(target=game, args=(child_conn_game1, child_conn_game2))
    p_bot1 = Process(target=bot, args=(child_conn_game1, parent_conn_game1))
    p_bot2 = Process(target=bot, args=(child_conn_game2, parent_conn_game2))

    p_game.start()
    p_bot1.start()
    p_bot2.start()

    while True:
        """"
        try:
            # Receive responses from game and print
            game_response1 = parent_conn_game1.recv()
            print(game_response1)

            # Receive responses from game and print
            game_response2 = parent_conn_game2.recv()
            print(game_response2)

            # Send responses to bots
            parent_conn_game1.send(game_response2)
            parent_conn_game2.send(game_response1)

        except EOFError:
            # If the user ends the input (Ctrl + D), terminate the processes and exit
            p_game.terminate()
            p_bot1.terminate()
            p_bot2.terminate()
            break
        """