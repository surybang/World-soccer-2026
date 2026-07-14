"""Matchs restants de la Coupe du Monde 2026.

Le tournoi a démarré le 11 juin 2026. On COUPE l'entraînement à cette date :
le modèle n'a jamais vu un seul match de ce tournoi, ni dans son train, ni dans
les snapshots qui alimentent ses features. Les pronostics ci-dessous sont donc
réellement hors échantillon.
"""

import pandas as pd

WC_START = pd.Timestamp("2026-06-11")

# Les quatre demi-finalistes, dans l'ordre du tableau.
DEMI_FINALES = [
    {"a": "France",  "b": "Spain",     "date": "2026-07-14",
     "city": "Dallas",   "country": "United States", "label": "Demi-finale 1"},
    {"a": "England", "b": "Argentina", "date": "2026-07-15",
     "city": "Atlanta",  "country": "United States", "label": "Demi-finale 2"},
]

FINALE = {"date": "2026-07-19", "city": "New York",
          "country": "United States", "label": "Finale"}

# Dallas, Atlanta et New York sont toutes au niveau de la mer : le levier
# altitude ne joue AUCUN rôle dans ce tournoi. Seuls le voyage et l'écart
# climatique pèsent un peu, et l'Elo fait le reste.