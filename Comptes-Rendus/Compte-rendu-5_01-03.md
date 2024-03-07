# Compte-rendu 5 : 01/03

## Ce qui a été fait

- Monte-Carlo :
    - Optimisation de l'algorithme
    - Changement de la représentation du board et des coups possibles
    - Changement de l'évaluation des coups gagnants

- Mad Pod Racing :
  - Révision de la physique du jeu pour être beaucoup plus près de la version codingame
  - Séparation des intefaces graphique et textuelle
  - Intégration de la possibilité de faire jouer plusieurs bots en même temps pour les comparer

- Q-learning :
  - Révision de la discrétisation pour s'adapter à la nouvelle physique de MPR
  - L'apprentissage converge mais les résultats ne sont pas toujours pertinents


---

## Ce qui a été dit

- Qlearning
  - Reward : comparer une fonction basée sur la distance et une prenant en compte le temps
  - Discrétisation :
    - Décrire les distances/angles de manière non uniforme
    - Angles : garder un cône de 90 degrés (45 de chaque côté)
- Ultimate Tic Tac Toe :
  - cprofile : descendre dans la liste pour avoir les fonctions qui font appel aux méthodes numpy
  - génération de nombres aléatoires : générer un grand nombre et le parcourir en shiftant pour chaque génération

---

## À faire

- Tester différents scénarios avec le q-learning pour le faire marcher
- Continuer d'optimiser UCT
- Commencer à réfléchir à l'utilisation de réseaux de neurones

