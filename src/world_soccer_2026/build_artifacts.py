"""Construit les artefacts que l'app Streamlit consommera.

À lancer UNE FOIS en local, puis committer le dossier artifacts/ dans le repo.

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
import numpy as np
import pandas as pd

from world_soccer_2026.features import TEAM_STATS, prepare, team_snapshot
from world_soccer_2026.benchmark import fit_best_xgboost, make_xy, temporal_split, tune_xgboost
import world_soccer_2026.backtest
from world_soccer_2026.fixtures import WC_START
from world_soccer_2026.selection import select_teams
from world_soccer_2026.utils import load_results

ARTIFACTS = Path("artifacts")
N_TRIALS = 8


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
    raw = load_results("data/results.csv")

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
    df, ratings, base = prepare(raw)

    print(f"3/5  tuning XGBoost ({N_TRIALS} essais Optuna)")
    train_df, _ = temporal_split(df, test_year=2022)
    # Le tuning se fait sur le train du split temporel (mesure honnête), mais
    # le modèle final est réentraîné sur TOUT l'historique disponible, c'est à
    # dire tout ce qui précède le coup d'envoi du tournoi.
    X_tr, y_tr = make_xy(train_df, mirror=True)
    study = tune_xgboost(X_tr, y_tr, n_trials=N_TRIALS)

    X_all, y_all = make_xy(df, mirror=True)
    model = fit_best_xgboost(study, X_all, y_all)
    joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
    print(f"     log loss CV : {study.best_value:.4f}")

    print("4/5  snapshots des équipes")
    equipes = sorted(set(df["home_team"]) | set(df["away_team"]))
    snaps = {t: team_snapshot(df, ratings, t) for t in equipes}
    snap_df = pd.DataFrame(snaps).T[TEAM_STATS]
    snap_df.index.name = "team"

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
        "best_params": study.best_params,
        "cv_log_loss": float(study.best_value),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

    print("5/5  rétro-test sur la Coupe du Monde en cours")
    import predict
    predict.load.cache_clear()          # relire les artefacts qu'on vient d'écrire
    connues = set(snap_df.index)

    bt = world_soccer_2026.backtest.run(
        load_results("data/results.csv"), WC_START,
        lambda a, b, city, country: predict.match_proba(a, b, city, country),
        connues)

    if bt.empty:
        print("     aucun match de CDM 2026 dans results.csv")
    else:
        bt.to_parquet(ARTIFACTS / "backtest.parquet")
        r = world_soccer_2026.backtest.summarise(bt)
        print(f"     {r['n_matchs']} matchs, {r['n_nuls']} nuls écartés")
        print(f"     réussite {r['accuracy']:.1%} sur les {r['n_decisifs']} décisifs "
              f"| log loss {r['log_loss']:.4f}")

    total = sum(f.stat().st_size for f in ARTIFACTS.iterdir())
    print(f"\nartifacts/ : {total / 1024:.0f} Ko, {len(equipes)} équipes")
    for f in sorted(ARTIFACTS.iterdir()):
        print(f"  {f.name:<20} {f.stat().st_size / 1024:>7.1f} Ko")


if __name__ == '__main__':
    main()