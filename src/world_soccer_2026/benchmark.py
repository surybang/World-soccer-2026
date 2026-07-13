"""Benchmark de modèles.

Deux partis pris méthodologiques, tous deux différents du réflexe par défaut :

1. SPLIT CHRONOLOGIQUE, pas aléatoire. On prédit le futur avec le passé, donc
   le test doit être postérieur au train. Un train_test_split aléatoire laisse
   le modèle voir des matchs de 2025 pour en prédire de 2015 : les métriques
   sont optimistes et ne reflètent pas l'usage réel.

2. LOG LOSS ET BRIER en métrique principale, pas l'accuracy. La simulation de
   tournoi consomme des predict_proba : ce qui compte est la CALIBRATION.
   Un modèle à 66% bien calibré est plus utile qu'un modèle à 67% qui annonce
   90% de confiance à tort.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from world_soccer_2026.features import FEATURES, build_pair_frame

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Découpage
# ---------------------------------------------------------------------------

def temporal_split(df, test_year=2022):
    """Train strictement antérieur à test_year, test à partir de test_year."""
    train = df[df["date"].dt.year < test_year]
    test = df[df["date"].dt.year >= test_year]
    return train, test


def make_xy(df, mirror=False):
    frame = build_pair_frame(df, mirror=mirror)
    return frame[FEATURES], frame["target"].to_numpy()


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def preprocessor(scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_models(seed=42):
    from xgboost import XGBClassifier

    return {
        "baseline_favori_domicile": Pipeline([
            ("prep", preprocessor()),
            ("clf", DummyClassifier(strategy="constant", constant=1)),
        ]),
        "regression_logistique": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=20,
                n_jobs=-1, random_state=seed)),
        ]),
        "hist_gbm": Pipeline([
            ("prep", preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed)),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss", tree_method="hist",
                random_state=seed, n_jobs=-1)),
        ]),
    }


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def score(name, y_true, proba):
    return {
        "modele": name,
        "accuracy": accuracy_score(y_true, (proba >= 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
    }


def build_tabicl(seed=42, **kwargs):
    """TabICL, modèle de fondation tabulaire. Renvoie None s'il n'est pas installé.

    Il est traité à part des autres dans run_benchmark, pour une raison
    structurelle et non par caprice : son inférence est en O(n²) sur la taille
    du CONTEXTE (le train). Lui donner le train augmenté en miroir quadruplerait
    le coût pour un gain nul, puisque l'antisymétrie est déjà garantie au
    moment de la prédiction par le moyennage de tournament.build_pair_matrix.
    Il reçoit donc le train DIRECT, là où les autres reçoivent le miroir.
    """
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        return None
    return Pipeline([
        ("prep", preprocessor()),          # TabICL ne gère pas les NaN
        ("clf", TabICLClassifier(random_state=seed, **kwargs)),
    ])


def run_benchmark(df, test_year=2022, mirror_train=True, seed=42,
                  include_tabicl=True, tabicl_max_rows=40_000):
    """Entraîne tous les modèles et renvoie (tableau de scores, données).

    Tous les modèles sont évalués sur EXACTEMENT le même jeu de test, ce qui
    rend les lignes du tableau comparables. Seul le jeu d'entraînement diffère
    pour TabICL (train direct plutôt que miroir), pour la raison expliquée
    dans build_tabicl.
    """
    train_df, test_df = temporal_split(df, test_year)

    # Le miroir n'est appliqué QU'AU TRAIN. Le test garde ses lignes
    # originales, sinon chaque match compterait deux fois et les métriques
    # perdraient leur sens.
    X_tr, y_tr = make_xy(train_df, mirror=mirror_train)
    X_te, y_te = make_xy(test_df, mirror=False)

    print(f"Train : {len(X_tr):>6} lignes "
          f"({train_df['date'].dt.year.min()}-{train_df['date'].dt.year.max()}"
          f"{', miroir inclus' if mirror_train else ''})")
    print(f"Test  : {len(X_te):>6} lignes "
          f"({test_df['date'].dt.year.min()}-{test_df['date'].dt.year.max()})")
    print(f"Baseline (toujours l'équipe à domicile) : {y_te.mean():.1%}\n")

    rows = []
    for name, model in build_models(seed).items():
        model.fit(X_tr, y_tr)
        rows.append(score(name, y_te, model.predict_proba(X_te)[:, 1]))

    if include_tabicl:
        tabicl = build_tabicl(seed, kv_cache=True)
        if tabicl is None:
            print("tabicl non installé (uv add tabicl), ligne absente du tableau\n")
        else:
            X_ct, y_ct = make_xy(train_df, mirror=False)     # contexte direct
            if len(X_ct) > tabicl_max_rows:
                # TabICLv2 est pré-entraîné sur 300 à 48k lignes : au-delà, on
                # garde la période la plus RÉCENTE, la plus pertinente ici.
                X_ct, y_ct = X_ct.tail(tabicl_max_rows), y_ct[-tabicl_max_rows:]
            print(f"tabicl : contexte de {len(X_ct)} lignes (train direct, sans miroir)")

            t0 = time.time()
            tabicl.fit(X_ct, y_ct)
            tabicl.predict_proba(X_te.head(100))
            per_row = (time.time() - t0) / 100
            print(f"  ~{per_row * len(X_te) / 60:.1f} min estimées pour "
                  f"les {len(X_te)} lignes de test\n")

            rows.append(score("tabicl", y_te, tabicl.predict_proba(X_te)[:, 1]))

    return (pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True),
            (X_tr, y_tr, X_te, y_te))


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def tune_xgboost(X_tr, y_tr, n_trials=60, seed=42):
    """Recherche d'hyperparamètres, validation croisée TEMPORELLE.

    TimeSeriesSplit et non KFold : un KFold validerait sur du passé avec un
    modèle entraîné sur du futur. Le tuning optimiserait alors une métrique
    optimiste qui ne se transfère pas au vrai jeu de test.
    """
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        }
        losses = []
        for tr_idx, va_idx in cv.split(X_tr):
            pipe = Pipeline([
                ("prep", preprocessor()),
                ("clf", XGBClassifier(**params, eval_metric="logloss",
                                      tree_method="hist", random_state=seed,
                                      n_jobs=-1)),
            ])
            pipe.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
            p = pipe.predict_proba(X_tr.iloc[va_idx])[:, 1]
            losses.append(log_loss(y_tr[va_idx], p, labels=[0, 1]))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def fit_best_xgboost(study, X_tr, y_tr, seed=42):
    from xgboost import XGBClassifier
    return Pipeline([
        ("prep", preprocessor()),
        ("clf", XGBClassifier(**study.best_params, eval_metric="logloss",
                              tree_method="hist", random_state=seed, n_jobs=-1)),
    ]).fit(X_tr, y_tr)
