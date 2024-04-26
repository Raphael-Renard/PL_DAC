# Compte-rendu 10 : 05/04

## Ce qui a été fait
- UTTT : 
    - courbe entropie augmente, converge
    - les q-values ne sont pas à 0

- Policy Gradient :
    - implémentation sur Cart Pole
    - début d'implémentation sur UTTT (problème de coups illégaux)

- Qlearning tabulaire :
  - Mise en place du thrust relatif dans l'état
  - Test sur codingame : résultats satisfaisants


---

## Ce qui a été dit

- DQN : 
  - UTTT :
    - Qvalues très faibles (bizarre si arrive à gagner quelques parties)
    - Découper le jeu en 2 parties (s'arrêter quand on gagne une mini-grille voir si ça marche)
    - Regarder la différence quand on met comme reward :
      - 0 partout
      - Reward quand on complète une mini-grille
  - MPR :
    - Temps d'éxec trop long
    - loss super régulière
    - C = 50 : trop peu
    - loss calculée en fin d'épisode pas stable (calculer en moyenne par épisode ou décorrélée des épisodes)
- Policy Gradient :
  - Coups illégaux : reward = -100
  - Coup légal : reward positif

---

## À faire

