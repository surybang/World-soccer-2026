"""Explicabilité : qu'est-ce qui fait réellement la décision du modèle ?

Trois angles, du plus global au plus local.

1. IMPORTANCE PAR PERMUTATION, sur le jeu de TEST. On mélange une colonne et on
   regarde de combien la log loss se dégrade. À préférer à `feature_importances_`
   de XGBoost, qui est biaisée en faveur des variables à forte cardinalité et se
   calcule sur le train.

2. IMPORTANCE PAR GROUPE. Permuter `diff_elo` seul sous-estime son rôle, parce
   que `diff_pts_10` peut partiellement le remplacer : le modèle compense. En
   permutant tout le bloc "force" d'un coup, on mesure ce que le bloc apporte
   réellement. C'est la mesure qui compte quand les features sont corrélées, ce
   qui est massivement le cas ici.

3. SHAP, pour un match donné : quelles features ont poussé la décision, et dans
   quel sens.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Regroupement thématique des features.
GROUPES = {
    "force (Elo)": ["diff_elo", "diff_sos_10"],
    "forme récente": ["diff_pts_5", "diff_pts_10", "diff_pts_20",
                      "diff_gs_10", "diff_gc_10", "diff_gd_10", "diff_win_10"],
    "géographie": ["diff_alt_shock", "diff_travel_km", "diff_lat_shift",
                   "diff_climate_shift", "venue_altitude"],
    "contexte": ["home_advantage", "importance", "is_friendly"],
    "historique direct": ["h2h_n", "h2h_winrate"],
    "expérience": ["diff_n_matches", "min_n_matches", "diff_rest_days"],
}


def _perte(model, X, y, cols=None):
    p = model.predict_proba(X if cols is None else X[cols])[:, 1]
    return log_loss(y, p, labels=[0, 1])


def permutation(model, X, y, n_repeats=10, seed=42):
    """Dégradation de la log loss quand chaque feature est mélangée."""
    rng = np.random.default_rng(seed)
    base = _perte(model, X, y)

    lignes = []
    for f in X.columns:
        pertes = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            pertes.append(_perte(model, Xp, y))
        pertes = np.array(pertes)
        lignes.append({"feature": f,
                       "degradation": pertes.mean() - base,
                       "ecart_type": pertes.std()})

    return (pd.DataFrame(lignes)
            .sort_values("degradation", ascending=False)
            .set_index("feature"))


def permutation_par_groupe(model, X, y, n_repeats=10, seed=42):
    """Idem, mais en permutant des BLOCS de features corrélées.

    C'est la mesure honnête ici. Le modèle peut compenser la perte de `diff_elo`
    avec `diff_pts_10` ; permuter les deux ensemble empêche cette compensation
    et révèle ce que le bloc apporte vraiment.
    """
    rng = np.random.default_rng(seed)
    base = _perte(model, X, y)

    lignes = []
    for nom, cols in GROUPES.items():
        cols = [c for c in cols if c in X.columns]
        pertes = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            for c in cols:                 # même permutation pour tout le bloc,
                Xp[c] = Xp[c].to_numpy()[idx]   # ce qui casse le lien à la cible
            pertes.append(_perte(model, Xp, y))   # sans casser les corrélations
        pertes = np.array(pertes)                  # internes au bloc
        lignes.append({"groupe": nom,
                       "n_features": len(cols),
                       "degradation": pertes.mean() - base,
                       "ecart_type": pertes.std()})

    return (pd.DataFrame(lignes)
            .sort_values("degradation", ascending=False)
            .set_index("groupe"))


def ablation_groupe(model_fn, X_tr, y_tr, X_te, y_te):
    """Réentraîne le modèle SANS chaque groupe, et mesure ce qu'on perd.

    Complémentaire de la permutation : la permutation mesure ce que le modèle
    ENTRAÎNÉ utilise ; l'ablation mesure ce qu'un modèle RÉENTRAÎNÉ sans ces
    features saurait faire. Si la géographie est redondante avec l'Elo, la
    permutation la dira importante alors que l'ablation la dira inutile.
    """
    toutes = list(X_tr.columns)
    complet = model_fn().fit(X_tr, y_tr)
    base = _perte(complet, X_te, y_te)

    lignes = [{"retiré": "rien (référence)", "log_loss": base, "cout": 0.0}]
    for nom, cols in GROUPES.items():
        garde = [c for c in toutes if c not in cols]
        if not garde or len(garde) == len(toutes):
            continue
        m = model_fn().fit(X_tr[garde], y_tr)
        ll = _perte(m, X_te[garde], y_te)
        lignes.append({"retiré": nom, "log_loss": ll, "cout": ll - base})

    return (pd.DataFrame(lignes)
            .sort_values("cout", ascending=False)
            .set_index("retiré"))


def shap_un_match(model, ligne, top=6):
    """Contributions SHAP d'un match, via le TreeSHAP natif d'XGBoost.

    Les colonnes sont déduites de `ligne`, et non importées depuis features.py :
    le modèle déployé n'utilise qu'un sous-ensemble (FEATURES_SELECTED), et
    importer FEATURES en dur ferait diverger l'explication du modèle.
    """
    import xgboost as xgb

    feats = list(ligne.columns)

    clf = model.named_steps["clf"]
    Xp = model.named_steps["prep"].transform(ligne)
    dm = xgb.DMatrix(Xp, feature_names=feats)

    # la dernière colonne est le biais (valeur de base), on l'écarte
    contribs = clf.get_booster().predict(dm, pred_contribs=True)[0][:-1]

    out = (pd.DataFrame({"feature": feats,
                         "contribution": contribs,
                         "valeur": ligne.iloc[0].to_numpy()})
           .assign(poids=lambda d: d["contribution"].abs())
           .sort_values("poids", ascending=False)
           .head(top)
           .drop(columns="poids")
           .set_index("feature"))
    out["pousse_vers"] = np.where(out["contribution"] > 0, "équipe A", "équipe B")
    return out
