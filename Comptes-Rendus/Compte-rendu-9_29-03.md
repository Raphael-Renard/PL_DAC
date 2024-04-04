# Compte-rendu 9 : 29/03

## Ce qui a été fait
- DQN :
    - changement de Tensorflow vers Pytorch
    - courbes de loss

- UTTT : 
    - Courbe MCTS contre heuristique
    - DQN converge mais donne de très mauvais résultats contre bot aléatoire

- Qlearning :
    - Généralisation de l'environnement MPR pour qu'il soit utilisable par tous les algorithmes (pour l'instant seulement utilisé par Qlearning)
    - Tentatives de débuggage (accélération toujours erratique)
    - Utilisation de gzip pour compresser la table &rarr; pas de grand chabgement dans la taille 
---

## Ce qui a été dit

- Qlearning tabulaire :
  - Essayer de mettre le thrust relatif dans codingame
  - Lier le thrust et la vitesse dans l'état
  
- DQN :
  - Tronquer les épisodes en train (fait dans le nouvel environnement)
  - Tracer les courbes de loss

- UTTT :
  - La loss de train est bizarre
    - Ressemble à une loss de train normale pas de RL
    - &rarr; Overfitting complet ou problème d'algo
    - Vérifier si pour l'inférence obtenue toutes les q-values sont à 0
    - La loss devrait être instable au départ, puis se stabiliser vers une valeur
  - Tracer l'entropie des sorties :
    - Si converge vers 0 ou aléatoire complet : pas bon
  - Essayer de mettre &epsilon; à 50% et décroître
    - Devrait apporter plus d'instabilité à la courbe de loss

---

## À faire

- Passer à Policy Gradient
  - Lire les articles
  - Commencer à implémenter si clair





