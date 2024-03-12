# Compte-rendu 6 : 08/03

## Ce qui a été fait

- Qlearning : 
  - Amélioration de la discrétisation des états -> le bot adopte des trajectoires correctes
  - Amélioration des actions possibles -> le bot a plus de précision dans ses mouvements
  - Augmentation du poids de la distance dans la récompense -> le bot va plus vite
  - Etat actuel : le bot qlearning a des performances comparables à celles des bots heuristiques
---

## Ce qui a été dit

- Qlearning : ne pas changer les performances en scalant le reward
- Tic tac toe : le code rajouté peut contenir des bugs (faire attention)

---

## À faire

- Qlearning
  - Essayer gamma=1
  - Créer des courbes pour comparer les performances heuristique/qlearning
- Simulations UCB : garder une trace des parties (déroulement + résultat) &rarr; création d'un dataset qui pourra servir à de l'apprentissage supervisé
- Commencer à implémenter les Réseaux DQN (cf slides mattermost)
  - Sur MPR (adaptation du Qlearning) et UTTT (plus de réflexion)