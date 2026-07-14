"""Construit les artefacts que l'app Streamlit consommera.

> À lancer UNE FOIS en local sinon réutiliser ce qui est dans le dossier artifacts.

Pourquoi ne pas simplement appeler prepare() au démarrage de l'app :
  - prepare() met une dizaine de secondes et relit tout results.csv ;
  - Streamlit Community Cloud plafonne à 1 Go de RAM et redémarre souvent ;
  - le modèle n'a en réalité besoin que de l'ÉTAT COURANT de chaque équipe
    (Elo, forme sur 10 matchs, base géographique) et de l'historique des
    confrontations directes. Tout ça tient dans quelques dizaines de Ko.

L'app ne lit donc jamais results.csv. Elle charge trois petits fichiers.
"""
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from world_soccer_2026 import backtest
from world_soccer_2026.benchmark import make_xy, preprocessor
from world_soccer_2026.features import FEATURES_SELECTED, prepare
from world_soccer_2026.fixtures import WC_START
from world_soccer_2026.selection import select_teams
from world_soccer_2026.utils import load_results

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def head_to_head_table(df):
    """Bilan des confrontations directes, pour toutes les paires qui se sont vues.

    build_pair_matrix mettait h2h_n=0 et h2h_winrate=0.5 en dur, donc ignorait
    l'historique direct. Ici on le calcule pour de bon : c'est une feature que
    le modèle a apprise, autant la lui fournir correctement.
    """
    a = df["home_team"].to_numpy()
    b = df["away_team"].to_numpy()
    home_win = (df["home_score"] > df["away_score"]).to_numpy()

    tally: dict[tuple[str, str], list[int]] = {}
    for i in range(len(df)):
        k = (a[i], b[i])
        r = tally.setdefault(k, [0, 0])
        r[0] += 1
        r[1] += int(home_win[i])
        k2 = (b[i], a[i])
        r2 = tally.setdefault(k2, [0, 0])
        r2[0] += 1
        r2[1] += int(not home_win[i])

    return {f"{x}|{y}": {"n": n, "winrate": w / n}
            for (x, y), (n, w) in tally.items()}


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    print("1/5  chargement et filtrage")
    brut = load_results(ROOT / "data" / "results.csv")   # gardé pour le rétro-test

    raw = brut[(brut["year"] >= 1994) & (brut["date"] < WC_START)]
    print(f"     coupure au {WC_START.date()} : {len(raw)} matchs, "
          f"dernier le {raw['date'].max().date()}")

    raw, _ = select_teams(raw, min_matches=50, require_fifa=True)

    # COUPURE AU COUP D'ENVOI DU TOURNOI.
    # Ce n'est pas qu'une question de train : les SNAPSHOTS (Elo, forme) sont
    # eux aussi calculés sur ces données. Si l'Elo de la France intégrait ses
    # matchs de Coupe du Monde, il y aurait fuite même avec un train propre.
    # On coupe donc tout à la même date. Le modèle n'a jamais vu ce tournoi.
    raw = raw[(raw["year"] >= 1994) & (raw["date"] < WC_START)]
    print(f"     coupure au {WC_START.date()} : {len(raw)} matchs, "
          f"dernier le {raw['date'].max().date()}")

    raw, _ = select_teams(raw, min_matches=50, require_fifa=True)

    print("2/5  features (Elo, forme, h2h, géo)")
    df, ratings, base, snapshots = prepare(raw)

    print("3/5  entraînement XGBoost")
    # Pas de tuning : Optuna dégrade la log loss sur les deux jeux de features
    # (0.4454 -> 0.4463 sur 12 features, 0.4473 -> 0.4475 sur 22). Les valeurs
    # par défaut sont déjà dans la zone plate de l'espace de recherche.
    # Le modèle final est entraîné sur TOUT l'historique disponible, c'est à dire
    # tout ce qui précède le coup d'envoi du tournoi.
    X_all, y_all = make_xy(df, mirror=True)

    model = Pipeline([
        ("prep", preprocessor()),
        ("clf", XGBClassifier(n_estimators=400, learning_rate=.05, max_depth=4,
                              subsample=.8, colsample_bytree=.8,
                              eval_metric="logloss", tree_method="hist",
                              random_state=42, n_jobs=-1)),
    ]).fit(X_all[FEATURES_SELECTED], y_all)

    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
    print(f"     {len(FEATURES_SELECTED)} features, {len(X_all)} lignes (miroir inclus)")

    print("4/5  snapshots des équipes")
    equipes = list(snapshots.index)
    snap_df = snapshots

    out = snap_df.join(base.reindex(snap_df.index))
    out.to_parquet(ARTIFACTS / "teams.parquet")

    h2h = head_to_head_table(df)
    (ARTIFACTS / "h2h.json").write_text(json.dumps(h2h))

    # meta.json DOIT être écrit ICI, avant l'étape 5 : le rétro-test appelle
    # predict.load(), qui lit les QUATRE artefacts. L'écrire en fin de main()
    # faisait planter le premier build sur un dossier artifacts/ vierge.
    meta = {
        "n_matchs": int(len(df)),
        "n_equipes": len(equipes),
        "derniere_date": str(df["date"].max().date()),
        "coupure": str(WC_START.date()),
        "features": FEATURES_SELECTED,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

    print("5/5  rétro-test sur la Coupe du Monde en cours")
    from world_soccer_2026 import predict
    predict.load.cache_clear()          # relire les artefacts qu'on vient d'écrire
    connues = set(snap_df.index)

    bt = backtest.run(
            brut, WC_START,
            lambda a, b, city, country: predict.match_proba(a, b, city, country),
            connues)

    if bt.empty:
        print("     aucun match de CDM 2026 dans results.csv")
    else:
        bt.to_parquet(ARTIFACTS / "backtest.parquet")
        r = backtest.summarise(bt)
        print(f"{r['n_matchs']} matchs, {r['n_nuls']} nuls écartés")
        print(f"réussite {r['accuracy']:.1%} sur les {r['n_decisifs']} décisifs "
              f"| log loss {r['log_loss']:.4f}")

    total = sum(f.stat().st_size for f in ARTIFACTS.iterdir())
    print(f"\nartifacts/ : {total / 1024:.0f} Ko, {len(equipes)} équipes")
    for f in sorted(ARTIFACTS.iterdir()):
        print(f"  {f.name:<20} {f.stat().st_size / 1024:>7.1f} Ko")


if __name__ == '__main__':
    main()
