"""Sélection de features, faite proprement.

Le piège à éviter : mesurer l'importance sur le jeu de TEST, retirer les
features qui semblent inutiles, puis réévaluer sur ce même test. On aurait
choisi les features en regardant la réponse, et le gain annoncé serait en
partie du surapprentissage sur le test.

Tout ce qui suit se passe donc EXCLUSIVEMENT sur le train, en validation
croisée temporelle. Le test n'est touché qu'une fois, à la toute fin, pour
comparer le modèle élagué au modèle complet.

Deux méthodes complémentaires :

1. FEATURES FANTÔMES (esprit Boruta). Pour chaque feature réelle, on ajoute une
   copie permutée d'elle-même : son "ombre". Une feature qui ne bat pas sa
   propre ombre n'apporte pas plus qu'un bruit de même distribution. C'est le
   test statistique qui manque à un simple classement d'importances : il donne
   un SEUIL au lieu d'un ordre.

2. RFECV. Élimination récursive, en mesurant à chaque étape par validation
   croisée. Répond à une autre question : quel est le plus petit sous-ensemble
   qui ne dégrade pas la performance ?
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from world_soccer_2026.benchmark import preprocessor
from world_soccer_2026.features import FEATURES


def _xgb(seed=42, **kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, learning_rate=.05, max_depth=4,
                  subsample=.8, colsample_bytree=.8, eval_metric="logloss",
                  tree_method="hist", random_state=seed, n_jobs=-1)
    params.update(kw)
    return XGBClassifier(**params)


def features_fantomes(X, y, n_runs=15, seed=42, quantile=1.0):
    """Chaque feature bat-elle sa propre ombre ?

    Pour chaque itération : on double le jeu avec des copies permutées, on
    entraîne, et on note quelles features réelles dépassent la MEILLEURE des
    ombres (quantile=1.0, le critère strict de Boruta). Une feature qui gagne
    rarement ce duel n'apporte rien de plus qu'un bruit.

    Renvoie le taux de victoire de chaque feature, entre 0 et 1.
    """
    rng = np.random.default_rng(seed)
    victoires = {f: 0 for f in FEATURES}

    for _ in range(n_runs):
        Xs = X[FEATURES].copy()
        for f in FEATURES:                       # l'ombre de chaque feature
            Xs[f"ombre_{f}"] = rng.permutation(X[f].to_numpy())

        m = _xgb(seed=int(rng.integers(1e6))).fit(Xs, y)
        imp = pd.Series(m.feature_importances_, index=Xs.columns)

        ombres = imp[[c for c in Xs.columns if c.startswith("ombre_")]]
        seuil = ombres.quantile(quantile)        # la meilleure ombre

        for f in FEATURES:
            if imp[f] > seuil:
                victoires[f] += 1

    return (pd.Series(victoires, name="taux_victoire")
            .div(n_runs)
            .sort_values(ascending=False)
            .to_frame())


def rfecv(X, y, n_splits=4, min_features=3):
    """Élimination récursive avec validation croisée TEMPORELLE.

    Retire la feature la moins importante, mesure, recommence. Renvoie
    l'historique complet, pour qu'on voie la courbe et pas seulement l'optimum.
    """
    from sklearn.metrics import log_loss

    cv = TimeSeriesSplit(n_splits=n_splits)
    restantes = list(FEATURES)
    historique = []

    while len(restantes) >= min_features:
        pertes = []
        importances = np.zeros(len(restantes))

        for tr, va in cv.split(X):
            pipe = Pipeline([("prep", preprocessor()), ("clf", _xgb())])
            pipe.fit(X.iloc[tr][restantes], y[tr])
            p = pipe.predict_proba(X.iloc[va][restantes])[:, 1]
            pertes.append(log_loss(y[va], p, labels=[0, 1]))
            importances += pipe.named_steps["clf"].feature_importances_

        historique.append({"n_features": len(restantes),
                           "log_loss_cv": float(np.mean(pertes)),
                           "features": list(restantes)})

        pire = restantes[int(np.argmin(importances))]
        restantes.remove(pire)

    return pd.DataFrame(historique)


def meilleur_sous_ensemble(hist, tolerance=0.001):
    """Le plus PETIT jeu de features qui reste dans la tolérance de l'optimum.

    On ne prend pas bêtement le minimum de log loss : un modèle à 22 features
    qui bat un modèle à 8 features de 0.0002 ne vaut pas la complexité en plus.
    On cherche le plus petit jeu qui ne dégrade pas significativement.
    """
    best = hist["log_loss_cv"].min()
    ok = hist[hist["log_loss_cv"] <= best + tolerance]
    ligne = ok.loc[ok["n_features"].idxmin()]
    return list(ligne["features"]), float(ligne["log_loss_cv"])
