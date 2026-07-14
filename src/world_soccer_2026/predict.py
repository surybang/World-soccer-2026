"""Inférence pure : aucun accès à results.csv, aucun recalcul de features.

Partagé par l'app Streamlit et le notebook. Tout vient des artefacts produits
par build_artifacts.py.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from world_soccer_2026.features import ALL_STATS, FEATURES, GEO_STATS, TEAM_STATS
from world_soccer_2026.geo import ALTITUDE_FLOOR, CITY_ALTITUDE, city_altitude, haversine, load_coords

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


@lru_cache(maxsize=1)
def load():
    model = joblib.load(ARTIFACTS / "model.joblib")
    teams = pd.read_parquet(ARTIFACTS / "teams.parquet")
    h2h = json.loads((ARTIFACTS / "h2h.json").read_text())
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    coords = load_coords()
    return model, teams, h2h, meta, coords


@lru_cache(maxsize=1)
def load_backtest():
    """Rétro-test du tournoi en cours. Calculé au build, jamais ici."""
    f = ARTIFACTS / "backtest.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def geo_for(team, teams, venue_alt, venue_lat, venue_lon):
    """Inadaptation d'une équipe à un stade : dénivelé, voyage, choc climatique."""
    t = teams.loc[team]
    monte = venue_alt >= ALTITUDE_FLOOR
    return {
        "alt_shock": max(0.0, venue_alt - t["home_alt"]) if monte else 0.0,
        "travel_km": float(haversine(t["home_lat"], t["home_lon"],
                                     venue_lat, venue_lon)),
        "lat_shift": abs(venue_lat - t["home_lat"]),
        "climate_shift": abs(abs(venue_lat) - abs(t["home_lat"])),
    }


def _row(a, b, teams, h2h, venue_alt, venue_lat, venue_lon,
         importance, home_advantage, neutraliser_geo=False):
    stats_a = dict(teams.loc[a][TEAM_STATS])
    stats_b = dict(teams.loc[b][TEAM_STATS])

    if neutraliser_geo:
        # En phase finale de Coupe du Monde, les équipes sont sur place depuis
        # des semaines : acclimatées, et le "voyage" se réduit à un vol interne.
        # Or travel_km et alt_shock ont été APPRIS sur des qualifs et des
        # amicaux, où l'équipe arrive deux jours avant. La sémantique de la
        # feature change entre l'entraînement et la prédiction, ce qui est pire
        # qu'une feature absente : le modèle applique un handicap qui n'existe
        # plus. On met donc les écarts géographiques à zéro.
        geo = {s: 0.0 for s in GEO_STATS}
        stats_a.update(geo)
        stats_b.update(geo)
        venue_alt = 0.0
    else:
        stats_a.update(geo_for(a, teams, venue_alt, venue_lat, venue_lon))
        stats_b.update(geo_for(b, teams, venue_alt, venue_lat, venue_lon))

    duel = h2h.get(f"{a}|{b}", {"n": 0, "winrate": 0.5})

    r = {f"diff_{s}": stats_a[s] - stats_b[s] for s in ALL_STATS}
    r.update(home_advantage=home_advantage, importance=importance,
             is_friendly=int(importance == 0),
             h2h_n=duel["n"], h2h_winrate=duel["winrate"],
             min_n_matches=min(stats_a["n_matches"], stats_b["n_matches"]),
             venue_altitude=venue_alt)
    return r


def match_proba(team_a, team_b, venue_city="Dallas", venue_country="United States",
                importance=4, home_advantage=0, neutraliser_geo=True):
    """P(team_a bat team_b), garantie antisymétrique.

    neutraliser_geo : met les écarts géographiques à zéro. À utiliser en
    contexte de tournoi, où les équipes sont sur place depuis des semaines.
    """
    model, teams, h2h, meta, coords = load()
    feats = meta["features"]

    alt = city_altitude(venue_city)
    lat, lon = coords.get(venue_country, (40.4, -3.7))

    ab = _row(team_a, team_b, teams, h2h, alt, lat, lon,
              importance, home_advantage, neutraliser_geo)
    ba = _row(team_b, team_a, teams, h2h, alt, lat, lon,
              importance, -home_advantage, neutraliser_geo)

    batch = pd.DataFrame([ab, ba])[feats]
    p = model.predict_proba(batch)[:, 1]
    return float(0.5 * (p[0] + (1.0 - p[1])))


VENUES = {
    "Dallas (neutre)": ("Dallas", "United States"),
    "New York (finale)": ("New York", "United States"),
    "Atlanta": ("Atlanta", "United States"),
    "Mexico City (2 240 m)": ("Mexico City", "Mexico"),
    "La Paz (3 640 m)": ("La Paz", "Bolivia"),
    "Madrid": ("Madrid", "Spain"),
}


def _match_row(a, b, venue_city, venue_country, importance=4, home_advantage=0,
               neutraliser_geo=True):
    """La ligne de features d'un match, telle que le modèle la voit."""
    _, teams, h2h, meta, coords = load()          # meta, et non _
    alt = city_altitude(venue_city)
    lat, lon = coords.get(venue_country, (40.4, -3.7))
    r = _row(a, b, teams, h2h, alt, lat, lon, importance, home_advantage,
             neutraliser_geo)
    return pd.DataFrame([r])[meta["features"]]


def explain_match(a, b, venue_city="Dallas", venue_country="United States",
                  importance=4, home_advantage=0, neutraliser_geo=True, top=7):
    """Contributions SHAP pour ce match précis, via le TreeSHAP natif d'XGBoost."""
    from world_soccer_2026 import explain as expl

    model, *_ = load()
    ligne = _match_row(a, b, venue_city, venue_country, importance,
                       home_advantage, neutraliser_geo)
    return expl.shap_un_match(model, ligne, top=top)


def compare(a, b, venue_city="Dallas", venue_country="United States"):
    """Écarts bruts entre les deux équipes, pour lire le pronostic à l'oeil."""
    _, teams, _, _, coords = load()
    alt = city_altitude(venue_city)
    lat, lon = coords.get(venue_country, (40.4, -3.7))

    lignes = []
    for s_ in ["elo", "pts_20", "pts_10", "gd_10", "sos_10", "rest_days"]:
        lignes.append({"critère": s_, a: teams.loc[a, s_], b: teams.loc[b, s_]})

    ga = geo_for(a, teams, alt, lat, lon)
    gb = geo_for(b, teams, alt, lat, lon)
    for s_ in ["alt_shock", "travel_km"]:
        lignes.append({"critère": s_, a: ga[s_], b: gb[s_]})

    return pd.DataFrame(lignes).set_index("critère").round(0)


def bracket_probabilities(demis, finale, neutraliser_geo=True):
    """Probabilités de titre EXACTES pour un dernier carré.

    Avec quatre équipes, pas besoin de Monte-Carlo : on énumère les quatre
    scénarios et on somme. Le Monte-Carlo n'est nécessaire que sur un tableau
    de 32 équipes, où les chemins se comptent en milliards.

        P(A championne) = P(A gagne sa demie)
                          x somme, sur les adversaires possibles en finale, de
                            P(cet adversaire arrive) x P(A le bat)
    """
    (a1, b1) = demis[0]["a"], demis[0]["b"]
    (a2, b2) = demis[1]["a"], demis[1]["b"]

    p1 = match_proba(a1, b1, demis[0]["city"], demis[0]["country"], neutraliser_geo=neutraliser_geo)
    p2 = match_proba(a2, b2, demis[1]["city"], demis[1]["country"], neutraliser_geo=neutraliser_geo)
    demi = {a1: p1, b1: 1 - p1, a2: p2, b2: 1 - p2}

    titres = {}
    for x, adversaires in [(a1, (a2, b2)), (b1, (a2, b2)),
                           (a2, (a1, b1)), (b2, (a1, b1))]:
        total = sum(demi[y] * match_proba(x, y, finale["city"], finale["country"])
                    for y in adversaires)
        titres[x] = demi[x] * total

    return (pd.Series(demi).sort_values(ascending=False),
            pd.Series(titres).sort_values(ascending=False))


def available_teams():
    _, teams, _, _, _ = load()
    return sorted(teams.index)


def known_venues():
    """Villes dont on connaît l'altitude, triées par altitude décroissante."""
    return sorted(CITY_ALTITUDE.items(), key=lambda kv: -kv[1])
