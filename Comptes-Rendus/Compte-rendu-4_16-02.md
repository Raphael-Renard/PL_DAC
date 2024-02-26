# Compte-rendu 4 : 16/02

## Ce qui a été fait

**Commun :**
- Poursuite de la lecture du livre

**Eliott :**
- A compléter

**Luc :**
- Implémentation d'une version basique de Mad Pod Racing
- Début d'application du Q learning sur Mad Pod Racing

**Raphaël :**
- Début d'implémentation du Q learning
- Correction de bugs survenant quand on utilise MCTS sur codingame

---

## Ce qui a été dit

- Module `cprofile` utile pour analyser les temps d'exécution de fonctions et voir les éléments pertinents à optimiser
- Pour la matrice du Qlearning, on peut utiliser un dictionnaire pour indexer les états
- Il faut faire attention à garder un code compact

---

## À faire

- Accélérer UCT (recoder `get_possible_moves`)
    - Garder les coups possible en mémoire et les supprimer au fur et à mesure (ne pas recalculer les coups possibles à chaque étape)
    - Représenter le board par un array numpy (pas de listes python), voire une représentation par bits
    - Pour les évaluations de coups gagnants utiliser des sommes partielles
- Séparer l'interface graphique de la boucle de jeu de Mad Pod Racing et poursuivre le Qlearning dessus
- Revoir l'organisation pour une meilleure efficacité de travail

