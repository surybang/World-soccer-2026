"""Inférence pure : aucun accès à results.csv, aucun recalcul de features.

Partagé par l'app Streamlit et le notebook. Tout vient des artefacts produits
par build_artifacts.py.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from world_soccer_2026.features import ALL_STATS, FEATURES, GEO_STATS, TEAM_STATS
from world_soccer_2026.geo import ALTITUDE_FLOOR, CITY_ALTITUDE, city_altitude, haversine, load_coords

ARTIFACTS = Path("artifacts")


@lru_cache(maxsize=1)
def load():
    """Charge modèle, snapshots, head-to-head et métadonnées. Mis en cache."""
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
         importance, home_advantage):
    stats_a = dict(teams.loc[a][TEAM_STATS])
    stats_b = dict(teams.loc[b][TEAM_STATS])
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


def match_proba(team_a, team_b, venue_city="Madrid", venue_country="Spain",
                importance=4, home_advantage=0, venue_altitude=None):
    """P(team_a bat team_b), garantie antisymétrique.

    On calcule les DEUX sens et on moyenne : p = (p(A,B) + 1 - p(B,A)) / 2.
    L'antisymétrie devient une propriété arithmétique et non apprise, donc
    exacte quel que soit le modèle. Sans ça, inverser l'ordre des deux équipes
    change le pronostic de plusieurs points.

    venue_altitude permet de forcer une altitude arbitraire (curseur de l'app)
    plutôt que d'utiliser celle de la ville.
    """
    model, teams, h2h, _, coords = load()

    alt = city_altitude(venue_city) if venue_altitude is None else float(venue_altitude)
    lat, lon = coords.get(venue_country, (40.4, -3.7))

    ab = _row(team_a, team_b, teams, h2h, alt, lat, lon, importance, home_advantage)
    ba = _row(team_b, team_a, teams, h2h, alt, lat, lon, importance, -home_advantage)

    batch = pd.DataFrame([ab, ba])[FEATURES]
    p = model.predict_proba(batch)[:, 1]
    return float(0.5 * (p[0] + (1.0 - p[1])))


def explain(team_a, team_b, venue_city="Madrid", venue_country="Spain",
            venue_altitude=None):
    """Écarts bruts entre les deux équipes, pour afficher le POURQUOI."""
    _, teams, h2h, _, coords = load()
    alt = city_altitude(venue_city) if venue_altitude is None else float(venue_altitude)
    lat, lon = coords.get(venue_country, (40.4, -3.7))

    lignes = []
    for s in ["elo", "pts_10", "gs_10", "gc_10", "sos_10"]:
        lignes.append({"critère": s,
                       team_a: teams.loc[team_a, s],
                       team_b: teams.loc[team_b, s]})
    ga = geo_for(team_a, teams, alt, lat, lon)
    gb = geo_for(team_b, teams, alt, lat, lon)
    for s in ["alt_shock", "travel_km"]:
        lignes.append({"critère": s, team_a: ga[s], team_b: gb[s]})

    return pd.DataFrame(lignes).set_index("critère").round(0)


def bracket_probabilities(demis, finale):
    """Probabilités de titre EXACTES pour un dernier carré.

    Avec quatre équipes, pas besoin de Monte-Carlo : on énumère les quatre
    scénarios possibles et on somme. Le Monte-Carlo n'était nécessaire que
    parce qu'un tableau de 32 équipes a 2^31 chemins.

        P(A championne) = P(A gagne sa demie)
                          x somme sur les adversaires possibles en finale de
                            P(cet adversaire arrive) x P(A le bat en finale)
    """
    (a1, b1), (a2, b2) = (demis[0]["a"], demis[0]["b"]), (demis[1]["a"], demis[1]["b"])

    p1 = match_proba(a1, b1, demis[0]["city"], demis[0]["country"])
    p2 = match_proba(a2, b2, demis[1]["city"], demis[1]["country"])
    demi = {a1: p1, b1: 1 - p1, a2: p2, b2: 1 - p2}

    titres = {}
    for x in (a1, b1):
        total = 0.0
        for y in (a2, b2):
            pf = match_proba(x, y, finale["city"], finale["country"])
            total += demi[y] * pf
        titres[x] = demi[x] * total
    for y in (a2, b2):
        total = 0.0
        for x in (a1, b1):
            pf = match_proba(y, x, finale["city"], finale["country"])
            total += demi[x] * pf
        titres[y] = demi[y] * total

    return (pd.Series(demi).sort_values(ascending=False),
            pd.Series(titres).sort_values(ascending=False))


def available_teams():
    _, teams, _, _, _ = load()
    return sorted(teams.index)


def known_venues():
    """Villes dont on connaît l'altitude, triées par altitude décroissante."""
    return sorted(CITY_ALTITUDE.items(), key=lambda kv: -kv[1])
