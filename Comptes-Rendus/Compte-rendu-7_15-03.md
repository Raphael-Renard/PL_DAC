# Compte-rendu 7 : 15/03

## Ce qui a été fait

- Ultimate-Tic-Tac-Toe :
    - enregistrement des parties jouées dans un fichier csv
    - début d'implémentation de DQN avec un CNN qui prend la petite grille dans laquelle on doit jouer
- Mad Pod Racing
  - Qlearning : création de courbes (comparaison des bots &gamma;=1 et &gamma;=.99 avec le bot heuristique)
  - DQN : début d'implémentation
---

## Ce qui a été dit

- Qlearning : 
  - Instabilité des courbes normale
  - Tester sur codingame : pickle &rarr; int base 64 à copier/coller dans codingame
- UCT :
  - Créer un bot aléatoire pour comparer (ou d'autres bots ex. github)
  - Jouer contre lui-même en faisant varier les hyperparamètres pour trouver les optimaux
  - Tester en local et pas codingame &rarr; + de simulations
- DQN :
  - Généraliser le modèle (et par la suite l'adapter au problème voulu si besoin)
  - S'inspirer de l'interface de gym
  - Pour tester/débugger : essayer des jeux gym (ex. lunar lander, car pole)
  - Objectif :
    - MPR &rarr; battre le bot qlearning tabular
    - UTTT &rarr; résultats incertains mais ça se tente

---

## À faire

- Poursuive l'implémentation de QDN (en mettant une version générique en commun)
- Tester le bot Qlearning sur codingame
- Tester le bot UCT en local contre un bot random 
