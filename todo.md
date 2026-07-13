1. vérifier rest_days dans team_snapshot de features.py
2. vérifier 1500 dans team_snapshot de features.py
3. Rajouter les features calculés par le TD machine learnia pour la comparaison
4. Bcp trop de AI slop dans les commentaires du notebooks
5. Préparer un tableau de résultats pour le README.md
6. Vérifier que l'équipe gagnante n'est pas la home_team comme dans le notebook (normalement corrigé par l'ajout d'asymétrie)
7. Est-ce vraiment le cas ? "Règle unique et non négociable : toute feature d'un match n'utilise que
l'information disponible AVANT le coup d'envoi. Chaque bloc ci-dessous
s'appuie donc soit sur un shift(1), soit sur un état mis à jour après coup."
8. Ajouter une simulation pour la CDM actuelle 
9. Ajouter une note sur Tabicl : il est plus performant qu'un xgboost tuné mais pour le peu de gain + le temps d'inférence trop long on usera xgb