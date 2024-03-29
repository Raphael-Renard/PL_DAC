# Compte-rendu 8 : 22/03

## Ce qui a été fait

- Ultimate Tic-Tac-Toe :
    - courbes pour comparer le bot MCTS contre random et heuristique contre random
    - courbe pour comparer différentes valeur du facteur d'exploration pour ucb
    - correction DQN : augmenter manuellement les valeurs des coups légaux dans la q-table pour ne plus jouer de coups illégaux
    - simplification du réseau de neurones utilisé

- Cart Pole :
    - implémentation de DQN

- Mad Pod Racing (Q-learning)
    - adaptation du bot q-learning à Codingame
    - résultats peu satisfaisants (comportement erratique &rarr; bugs ?)

---

## Ce qui a été dit

- Qlearning :
  - représentation de l'état dans la qtable \rarr; enlever les tuples
  - étudier les méthodes de compression (cf forums codingame ...)
- DQN :
  - Générique : 
    - (Optionnel) utiliser Pytorch plutôt que Tensorflow (plus de connaissances)
    - Tracer des courbes :
      - de loss &rarr; divergence normale au début mais doit se stabiliser / diminuer
      - de gradient (plus dur)
    - Se concentrer sur une seule bibliothèque :
      - Ne pas utiliser numpy (Pytorch / Tensorflow)
      - Utiliser les random d'une seule librairie (évite les conflits quand on set une seed)
  - MPR :
    - Mettre une limite au nombre d'itérations par épisode (100 par ex)
    - Reward : exp(-distance)
    - Augmenter le pas de temps entre deux updates
  - UTTT :
    - Reward petite grille bonne idée (mais peut être trompeur car peut encourager à compléter une petite grille inutile)
- UCT :
  - Courbes de comparaisons :
    - Faire jouer contre heuristique
    - Faire jouer contre les bots du git

---

## À faire

- MPR :
  - Refactoring du code pour match l'API de gym, se débarrasser des vecteurs pygame
  - Se renseigner sur les méthodes de compression pour codingame
- Continuer de travailler sur les bots en place
- Dans le futur (quand les bots marchent bien) :
  - Passer aux actions continues &rarr; policy gradient
  - Tester DQN sur Codingame