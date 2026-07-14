"""Feature engineering pour la prédiction de matchs internationaux.

Règle unique et non négociable : toute feature d'un match n'utilise que
l'information disponible AVANT le coup d'envoi. Chaque bloc ci-dessous
s'appuie donc soit sur un shift(1), soit sur un état mis à jour après coup.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Enjeu du match
# ---------------------------------------------------------------------------


def tournament_weight(name) -> float:
    """Poids Elo du match selon l'enjeu (barème World Football Elo)."""
    n = str(name).lower()
    if "friendly" in n:
        return 20.0
    if "qualification" in n or "qualifier" in n:
        return 30.0
    if "fifa world cup" in n:
        return 60.0
    if any(k in n for k in ("uefa euro", "copa américa", "copa america",
                            "african cup", "asian cup", "gold cup",
                            "nations league", "confederations cup")):
        return 50.0
    return 40.0


IMPORTANCE = {20.0: 0, 30.0: 1, 40.0: 2, 50.0: 3, 60.0: 4}


# ---------------------------------------------------------------------------
# 2. Elo
# ---------------------------------------------------------------------------


def add_elo(df, home_bonus=100.0, init=1500.0):
    """Ajoute l'Elo pré-match des deux équipes et leur nombre de matchs joués.

    Pourquoi c'est la feature centrale : une moyenne de points brute ne dit
    rien sans savoir CONTRE QUI ces points ont été pris. L'Elo est par
    construction ajusté à la force de l'adversaire, et se met à jour d'autant
    plus que le résultat était improbable.

    Renvoie (df enrichi, dict des notes finales).
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    played: dict[str, int] = {}

    n = len(df)
    eh = np.empty(n)
    ea = np.empty(n)
    nh = np.empty(n, dtype=int)
    na = np.empty(n, dtype=int)

    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy(dtype=float)
    as_ = df["away_score"].to_numpy(dtype=float)
    neutral = df["neutral"].to_numpy(dtype=bool)
    weight = df["tournament"].map(tournament_weight).to_numpy(dtype=float)

    for i in range(n):
        h, a = home[i], away[i]
        rh = ratings.get(h, init)
        ra = ratings.get(a, init)
        eh[i], ea[i] = rh, ra
        nh[i], na[i] = played.get(h, 0), played.get(a, 0)

        # l'avantage du terrain entre dans l'espérance, pas dans la note
        adv = 0.0 if neutral[i] else home_bonus
        expected_h = 1.0 / (1.0 + 10 ** (-(rh + adv - ra) / 400.0))

        gd = abs(hs[i] - as_[i])
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = (11.0 + gd) / 8.0

        if hs[i] > as_[i]:
            actual_h = 1.0
        elif hs[i] < as_[i]:
            actual_h = 0.0
        else:
            actual_h = 0.5

        delta = weight[i] * g * (actual_h - expected_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        played[h] = played.get(h, 0) + 1
        played[a] = played.get(a, 0) + 1

    df["home_elo"] = eh
    df["away_elo"] = ea
    df["home_n_matches"] = nh
    df["away_n_matches"] = na
    return df, ratings


# ---------------------------------------------------------------------------
# 3. Forme, force du calendrier, repos
# ---------------------------------------------------------------------------

WINDOWS = (5, 10, 20)


def add_team_form(df):
    """Forme multi-fenêtres, strength of schedule et jours de repos.

    Passe par un format long (une ligne par équipe et par match) : une même
    équipe apparaît tantôt en home_team, tantôt en away_team, et il faut
    réunir ses deux faces pour reconstituer son historique.
    """
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    def side(prefix, other):
        return pd.DataFrame({
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df[f"{prefix}_team"],
            "gs": df[f"{prefix}_score"],
            "gc": df[f"{other}_score"],
            "opp_elo": df[f"{other}_elo"],
            "side": prefix,
        })

    long = pd.concat([side("home", "away"), side("away", "home")],
                     ignore_index=True)

    long["pts"] = np.where(long["gs"] > long["gc"], 3,
                           np.where(long["gs"] == long["gc"], 1, 0))
    long["win"] = (long["gs"] > long["gc"]).astype(int)
    long["gd"] = long["gs"] - long["gc"]

    long = long.sort_values(["team", "date", "match_id"])
    g = long.groupby("team", sort=False)

    long["rest_days"] = (long["date"] - g["date"].shift(1)).dt.days.clip(upper=365)

    for w in WINDOWS:
        long[f"pts_{w}"] = g["pts"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=3).mean())

    for col, name in [("gs", "gs_10"), ("gc", "gc_10"), ("gd", "gd_10"),
                      ("win", "win_10"), ("opp_elo", "sos_10")]:
        long[name] = g[col].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean())

    feats = [f"pts_{w}" for w in WINDOWS] + [
        "gs_10", "gc_10", "gd_10", "win_10", "sos_10", "rest_days"]

    for prefix in ("home", "away"):
        sub = long.loc[long["side"] == prefix, ["match_id"] + feats]
        sub = sub.rename(columns={c: f"{prefix}_{c}" for c in feats})
        df = df.merge(sub, on="match_id", how="left")

    return df


def add_head_to_head(df):
    """Historique direct entre les deux équipes, strictement antérieur."""
    df = df.sort_values("date").reset_index(drop=True)
    pair = pd.Series([tuple(sorted(p)) for p in
                      zip(df["home_team"], df["away_team"])], index=df.index)
    first = pair.map(lambda t: t[0])

    won_first = np.where(
        df["home_team"] == first,
        (df["home_score"] > df["away_score"]).astype(float),
        (df["away_score"] > df["home_score"]).astype(float))

    tmp = df.assign(_pair=pair, _won_first=won_first)
    g = tmp.groupby("_pair", sort=False)
    df["h2h_n"] = g.cumcount()
    prior = g["_won_first"].transform(lambda s: s.shift(1).expanding().mean())

    # ramener du référentiel "équipe alphabétiquement première" vers "home"
    df["h2h_home_winrate"] = np.where(
        df["home_team"] == first, prior, 1.0 - prior)
    df["h2h_home_winrate"] = df["h2h_home_winrate"].fillna(0.5)
    return df


# ---------------------------------------------------------------------------
# 4. Cadrage symétrique A contre B
# ---------------------------------------------------------------------------

TEAM_STATS = ["elo", "n_matches", "pts_5", "pts_10", "pts_20",
              "gs_10", "gc_10", "gd_10", "win_10", "sos_10", "rest_days"]

# Features géographiques (cf. geo.py) : inadaptation de l'équipe au lieu du match
GEO_STATS = ["alt_shock", "travel_km", "lat_shift", "climate_shift"]
ALL_STATS = TEAM_STATS + GEO_STATS

FEATURES = (
    [f"diff_{s}" for s in ALL_STATS]
    + ["home_advantage", "importance", "is_friendly",
       "h2h_n", "h2h_winrate", "min_n_matches", "venue_altitude"]
)

# Sous-ensemble sans géographie, pour l'ablation
FEATURES_NO_GEO = [f for f in FEATURES
                   if not any(g in f for g in GEO_STATS)
                   and f != "venue_altitude"]

# Les features du TD d'origine (Machine Learnia), reconstruites à l'identique
# dans notre cadrage A contre B. Sert de point de départ mesurable : toute la
# progression du projet se lit comme l'écart entre ce jeu et les suivants.
FEATURES_TD = ["diff_pts_10", "diff_gs_10", "diff_gc_10",
               "home_advantage", "is_friendly"]

# Jeu retenu après sélection (RFECV + test contre features fantômes).
# Douze features au lieu de vingt-deux, à performance égale.
# Voir la partie "Sélection de features" du notebook : ce jeu est déterminé
# SUR LE TRAIN uniquement, jamais en regardant le test.
FEATURES_SELECTED = [
    "diff_elo", "diff_sos_10", "diff_pts_20", "diff_gd_10", "diff_rest_days",
    "diff_alt_shock", "diff_travel_km", "diff_climate_shift", "venue_altitude",
    "h2h_winrate", "home_advantage", "min_n_matches",
]

JEUX_DE_FEATURES = {
    "TD (5 features)": FEATURES_TD,
    "sans géographie": FEATURES_NO_GEO,
    "complet": FEATURES,
    "sélectionné": FEATURES_SELECTED,
}


def build_pair_frame(df, mirror=False):
    """Passe du cadrage 'domicile contre extérieur' au cadrage 'A contre B'.

    home_advantage vaut +1 si A reçoit, -1 si A se déplace, 0 si terrain
    neutre. Toutes les autres features sont des différences A moins B : elles
    changent donc de signe quand on inverse les équipes.

    Intérêt : le problème devient antisymétrique, ce qui est exactement ce
    qu'on veut pour simuler un tournoi en terrain neutre. Avec mirror=True on
    duplique chaque match dans les deux sens (cible inversée), ce qui double
    le train et force le modèle à apprendre cette antisymétrie.
    """
    imp = df["tournament"].map(tournament_weight).map(IMPORTANCE)

    from world_soccer_2026.geo import city_altitude

    def frame(a, b, sign):
        out = pd.DataFrame(index=df.index)
        for s in ALL_STATS:
            out[f"diff_{s}"] = df[f"{a}_{s}"] - df[f"{b}_{s}"]
        out["venue_altitude"] = df["city"].map(city_altitude)
        out["home_advantage"] = np.where(df["neutral"], 0, sign)
        out["importance"] = imp
        out["is_friendly"] = (df["tournament"] == "Friendly").astype(int)
        out["h2h_n"] = df["h2h_n"]
        out["h2h_winrate"] = (df["h2h_home_winrate"] if sign > 0
                              else 1.0 - df["h2h_home_winrate"])
        out["min_n_matches"] = np.minimum(df["home_n_matches"],
                                          df["away_n_matches"])
        out["date"] = df["date"].values
        out["team_a"] = df[f"{a}_team"].values
        out["team_b"] = df[f"{b}_team"].values
        return out

    direct = frame("home", "away", 1)
    direct["target"] = (df["home_score"] > df["away_score"]).astype(int)
    if not mirror:
        return direct

    swapped = frame("away", "home", -1)
    swapped["target"] = (df["away_score"] > df["home_score"]).astype(int)
    return pd.concat([direct, swapped], ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Pipeline complet
# ---------------------------------------------------------------------------


def prepare(raw, drop_draws=True):
    """Elo -> forme -> head to head -> géo.

    Renvoie (df, ratings, base, snapshots).

    `snapshots` est l'état de chaque équipe à la fin de la période, capturé
    AVANT la suppression des nuls. C'est essentiel : add_team_form calcule
    pts_10 sur tous les matchs, nuls compris, alors que le df renvoyé n'a plus
    que les matchs décisifs. Reconstruire l'état a posteriori depuis ce df
    surestimerait la forme d'environ +0.12 point par match (les nuls, qui ne
    rapportent qu'un point, disparaissent de la moyenne). Le modèle verrait
    alors une distribution à l'entraînement et une autre en prédiction.
    """
    df = raw.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])
    df, ratings = add_elo(df)
    df = add_team_form(df)
    df = add_head_to_head(df)

    from world_soccer_2026.geo import add_geo, load_coords, team_home_base
    coords = load_coords()
    base = team_home_base(df, coords)
    df = add_geo(df, coords, base)
    df = df.dropna(subset=["home_pts_10", "away_pts_10"]).reset_index(drop=True)

    snapshots = build_snapshots(df, ratings)

    if drop_draws:
        df = df[df["home_score"] != df["away_score"]].reset_index(drop=True)
    return df, ratings, base, snapshots


# ---------------------------------------------------------------------------
# 6. État courant d'une équipe (pour prédire un match futur)
# ---------------------------------------------------------------------------

def build_snapshots(df, ratings):
    """État courant de chaque équipe, lu dans les colonnes déjà calculées.

    On ne recalcule rien : add_team_form a produit home_pts_10, away_sos_10 et
    consorts avec le bon shift(1) et sur TOUS les matchs. Il suffit de prendre,
    pour chaque équipe, la ligne de son dernier match et d'y lire ses features
    du bon côté (home ou away).

    Cette approche corrige trois défauts de l'ancienne version, qui
    reconstruisait l'état à la main :
      - pts_20 valait pts_10 (la fenêtre de reconstruction était de 10) ;
      - rest_days était une constante de 30 jours, contre une médiane réelle
        de l'ordre de 10 ;
      - les nuls étaient absents du calcul, gonflant pts_10.
    """
    stats = [f"pts_{w}" for w in WINDOWS] + [
        "gs_10", "gc_10", "gd_10", "win_10", "sos_10", "rest_days"]

    lignes = {}
    for team in sorted(set(df["home_team"]) | set(df["away_team"])):
        m = df[(df["home_team"] == team) | (df["away_team"] == team)]
        if m.empty:
            continue
        derniere = m.iloc[-1]
        cote = "home" if derniere["home_team"] == team else "away"

        etat = {s: float(derniere[f"{cote}_{s}"]) for s in stats}
        etat["elo"] = float(derniere[f"{cote}_elo"])
        etat["n_matches"] = float(derniere[f"{cote}_n_matches"])

        # L'Elo et le compteur de matchs stockés dans la ligne sont ceux
        # d'AVANT ce match. On applique la dernière mise à jour, qui est dans
        # `ratings`, pour obtenir l'état réellement courant.
        etat["elo"] = float(ratings[team])
        etat["n_matches"] += 1

        lignes[team] = etat

    out = pd.DataFrame(lignes).T
    out.index.name = "team"
    return out[TEAM_STATS]


def team_snapshot(snapshots, team):
    """État d'une équipe, sous forme de dict. Lève si l'équipe est inconnue."""
    if team not in snapshots.index:
        raise KeyError(
            f"'{team}' absente des snapshots. Elle a probablement été écartée "
            f"par select_teams (trop peu de matchs, ou hors périmètre FIFA).")
    return dict(snapshots.loc[team])
