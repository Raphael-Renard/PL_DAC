from Game_representation import Morpion
from bot_mcts import MonteCarloTreeSearch
from bot_aleatoire import Aleatoire
import numpy as np
import matplotlib.pyplot as plt

mcts_gagne = []
mcts_neutre = []


#L_explo = [0.5,0.7,1,1.3,1.6,1.8,2]
L_explo = [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
for explo in L_explo:
    gagne_mcts = 0
    perdu_mcts = 0
    neutre_mcts = 0

    # test MCTS contre aléatoire
    for partie in range(50):
        state = Morpion(boards=np.array([[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
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
                    big_boards=np.array([[0,0,0],[0,0,0],[0,0,0]]))
        alea = Aleatoire(state)
        T = False

        
        while not T:
            MCTS = MonteCarloTreeSearch(state,record=False,exploration_factor=explo,simulations=100)
            best_move = MCTS.select_move()
            state.make_move_self(best_move)
            T = state.is_terminal((best_move[0]//3,best_move[1]//3))
            if T:
                break
            opponent_move = alea.give_move()
            state.make_move_self(opponent_move)
            T = state.is_terminal((opponent_move[0]//3,opponent_move[1]//3))

        resultat = state.get_result()
        if resultat == 1:
            gagne_mcts +=1
        elif resultat == -1:
            perdu_mcts +=1
        else:
            neutre_mcts +=1
    

    print("gagne mcts",gagne_mcts)
    mcts_gagne.append(gagne_mcts)
    mcts_neutre.append(neutre_mcts)
    print("perdu mcts",perdu_mcts)
    print("nul mcts",neutre_mcts)


plt.plot(L_explo,mcts_gagne,label="mcts gagné")
plt.plot(L_explo,mcts_neutre,label="mcts nul")
plt.legend()
plt.xlabel('Facteur d\'exploration')
plt.ylabel('Nombre de parties')
plt.title('Sur 50 parties jouées, nombre de parties gagnées et\nde matchs nuls en fonction du facteur d\'exploration')
plt.show()