"""Rejoue la Coupe du Monde 2026 avec le modèle d'avant le tournoi.

Règle du jeu : TOUTES les prédictions utilisent l'état des équipes au 10 juin 2026,
la veille du coup d'envoi. Aucune mise à jour au fil de la compétition.

Limite: le modèle est binaire (victoire ou défaite), les nuls ayant été écartés de l'entraînement.
"""
import numpy as np
import pandas as pd

TOURNAMENT = "FIFA World Cup"


def wc_matches(raw, start):
    """Matchs de la Coupe du Monde DÉJÀ JOUÉS, depuis le coup d'envoi.

    Le dataset peut contenir des rencontres programmées mais pas encore
    disputées (scores à NaN). On ne rétro-teste que ce qui a un résultat.
    """
    m = raw[(raw["tournament"] == TOURNAMENT) & (raw["date"] >= start)]
    m = m.dropna(subset=["home_score", "away_score"])
    return m.sort_values("date").reset_index(drop=True)


def run(raw, start, proba_fn, teams_connues):
    """Prédit chaque match du tournoi et compare au résultat réel.

    proba_fn(a, b, city, country) -> P(a bat b), typiquement predict.match_proba.

    Renvoie un DataFrame avec, par match : la probabilité annoncée, l'équipe
    donnée favorite, le résultat réel, et si le modèle avait raison.
    """
    matchs = wc_matches(raw, start)
    if matchs.empty:
        return pd.DataFrame()

    lignes = []
    for m in matchs.itertuples():
        a, b = m.home_team, m.away_team
        if a not in teams_connues or b not in teams_connues:
            continue  # équipe absente du périmètre (barragiste jamais vue, etc.)

        p = proba_fn(a, b, m.city, m.country)

        if m.home_score > m.away_score:
            reel = a
        elif m.away_score > m.home_score:
            reel = b
        else:
            reel = None  # nul : hors du champ d'un modèle binaire

        favori = a if p >= .5 else b
        confiance = max(p, 1 - p)

        lignes.append({
            "date": m.date,
            "phase": phase_de(m.date, start),
            "equipe_a": a,
            "equipe_b": b,
            "score": f"{int(m.home_score)} - {int(m.away_score)}",
            "p_a": p,
            "favori": favori,
            "confiance": confiance,
            "vainqueur": reel,
            "nul": reel is None,
            "correct": (reel == favori) if reel is not None else None,
        })

    return pd.DataFrame(lignes)


def phase_de(date, start):
    """Découpage approximatif du tournoi 2026 (48 équipes, 104 matchs)."""
    j = (pd.Timestamp(date) - pd.Timestamp(start)).days
    if j <= 16:
        return "Phase de groupes"
    if j <= 22:
        return "Seizièmes"
    if j <= 26:
        return "Huitièmes"
    if j <= 30:
        return "Quarts"
    if j <= 34:
        return "Demi-finales"
    return "Finale"


def summarise(bt):
    """Résumé global : ce que le modèle a réussi, et ce qu'il ne pouvait pas faire."""
    if bt.empty:
        return {}

    decisifs = bt[~bt["nul"]]
    n_nuls = int(bt["nul"].sum())

    from sklearn.metrics import brier_score_loss, log_loss

    y = (decisifs["vainqueur"] == decisifs["equipe_a"]).astype(int).to_numpy()
    p = decisifs["p_a"].to_numpy()

    return {
        "n_matchs": int(len(bt)),
        "n_decisifs": int(len(decisifs)),
        "n_nuls": n_nuls,
        "accuracy": float(decisifs["correct"].mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(set(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(set(y)) > 1 else np.nan,
        "confiance_moyenne": float(decisifs["confiance"].mean()),
    }


def par_phase(bt):
    """Taux de réussite par tour. Les tours avancés sont plus durs à prédire."""
    d = bt[~bt["nul"]]
    ordre = ["Phase de groupes", "Seizièmes", "Huitièmes", "Quarts",
             "Demi-finales", "Finale"]
    g = d.groupby("phase").agg(
        matchs=("correct", "size"),
        reussite=("correct", "mean"),
        confiance=("confiance", "mean"))
    return g.reindex([p for p in ordre if p in g.index])


def calibration(bt, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    """Le modèle tient-il ses promesses ?

    Quand il annonce 70% de confiance, gagne-t-il 70% du temps ? C'est la
    question qui compte : une accuracy élevée obtenue avec des probabilités
    mal calibrées produirait des simulations de tournoi trompeuses.
    """
    d = bt[~bt["nul"]].copy()
    d["tranche"] = pd.cut(d["confiance"], bins=list(bins), right=False)
    g = d.groupby("tranche", observed=True).agg(
        matchs=("correct", "size"),
        annonce=("confiance", "mean"),
        observe=("correct", "mean"))
    return g.dropna()


def surprises(bt, n=8):
    """Les échecs les plus cuisants : forte confiance, mauvais pronostic."""
    d = bt[(~bt["nul"]) & (~bt["correct"].astype(bool))]
    return d.nlargest(n, "confiance")[
        ["date", "phase", "equipe_a", "equipe_b", "score",
         "favori", "confiance", "vainqueur"]]
