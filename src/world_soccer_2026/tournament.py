"""Simulation de tournoi à partir d'un modèle entraîné.

Deux corrections par rapport à une simulation naïve :

1. ANTISYMÉTRIE GARANTIE. Un modèle à base d'arbres n'assure pas
   p(A bat B) + p(B bat A) == 1 : l'écart observé peut dépasser 0.2 sur un
   même match. On calcule donc les deux sens et on moyenne, ce qui rend la
   contrainte exacte. Sans ça, l'ordre dans lequel on passe les équipes
   change le pronostic, ce qui est absurde en terrain neutre.

2. TOUTES LES PAIRES EN UN SEUL predict_proba. Un bracket de 32 équipes ne
   compte que 32*31/2 = 496 affiches possibles. On les calcule une fois, en
   batch, et le Monte-Carlo tourne ensuite sur un simple lookup numpy.
   Indispensable pour TabICL, dont l'inférence est coûteuse.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from world_soccer_2026.features import ALL_STATS, team_snapshot
from world_soccer_2026.geo import ALTITUDE_FLOOR, city_altitude, haversine, load_coords


def venue_geo(team, base, venue_city, venue_country, coords):
    """Inadaptation d'une équipe à un stade donné (cf. geo.py)."""
    v_alt = city_altitude(venue_city)
    v_lat, v_lon = coords[venue_country]
    t = base.loc[team]

    return {
        "alt_shock": max(0.0, v_alt - t["home_alt"]) if v_alt >= ALTITUDE_FLOOR else 0.0,
        "travel_km": float(haversine(t["home_lat"], t["home_lon"], v_lat, v_lon)),
        "lat_shift": abs(v_lat - t["home_lat"]),
        "climate_shift": abs(abs(v_lat) - abs(t["home_lat"])),
    }


def build_pair_matrix(model, teams, snapshots, base,
                      venue=("Dallas", "USA"), importance=4, coords=None):
    """Renvoie P[(a, b)] = probabilité que a batte b, sur un stade donné.

    Le `venue` compte : à Madrid (667 m) le choc d'altitude est nul pour tout le
    monde, mais la distance parcourue ne l'est pas. Une Coupe du Monde à Mexico
    donnerait un tableau très différent.

    P est antisymétrique par construction : P[(a, b)] + P[(b, a)] == 1.
    """
    if coords is None:
        coords = load_coords()
    venue_city, venue_country = venue

    teams = list(dict.fromkeys(teams))
    snaps = {t: team_snapshot(snapshots, t) for t in teams}
    geo = {t: venue_geo(t, base, venue_city, venue_country, coords) for t in teams}
    for t in teams:
        snaps[t].update(geo[t])

    pairs = list(combinations(teams, 2))
    v_alt = city_altitude(venue_city)
    feats = list(model.named_steps["clf"].feature_names_in_)

    def row(a, b):
        r = {f"diff_{s}": snaps[a][s] - snaps[b][s] for s in ALL_STATS}
        r.update(home_advantage=0, importance=importance, is_friendly=0,
                 h2h_n=0, h2h_winrate=0.5, venue_altitude=v_alt,
                 min_n_matches=min(snaps[a]["n_matches"], snaps[b]["n_matches"]))
        return r

    # un seul appel au modèle pour les 2 x 496 lignes
    ab = pd.DataFrame([row(a, b) for a, b in pairs])[feats]
    ba = pd.DataFrame([row(b, a) for a, b in pairs])[feats]
    batch = pd.concat([ab, ba], ignore_index=True)

    proba = model.predict_proba(batch)[:, 1]
    n = len(pairs)
    p_ab, p_ba = proba[:n], proba[n:]
    p = 0.5 * (p_ab + (1.0 - p_ba))          # antisymétrisation

    P = {}
    for (a, b), v in zip(pairs, p):
        P[(a, b)] = float(v)
        P[(b, a)] = float(1.0 - v)
    return P


def flatten(bracket):
    """Accepte [(a, b), (c, d), ...] ou [a, b, c, d, ...] et renvoie la liste plate."""
    if bracket and isinstance(bracket[0], (tuple, list)):
        return [t for match in bracket for t in match]
    return list(bracket)


def simulate_deterministic(bracket, P):
    """Le favori de chaque match passe. Renvoie (résultats par tour, champion).

    Chaque tour est une liste de tuples (team_a, team_b, winner, probability),
    directement consommable par utils.print_round.
    """
    current = flatten(bracket)
    rounds = []
    while len(current) > 1:
        results, winners = [], []
        for a, b in zip(current[::2], current[1::2]):
            p = P[(a, b)]
            w = a if p >= 0.5 else b
            results.append((a, b, w, p if w == a else 1 - p))
            winners.append(w)
        rounds.append(results)
        current = winners
    return rounds, current[0]


def simulate_monte_carlo(bracket, P, n_sim=10_000, seed=42):
    """Tire le vainqueur de chaque match selon sa probabilité, n_sim fois.

    C'est la seule sortie honnête quand chaque match est à 60/40 : le favori
    déterministe donne un champion qui a l'air certain, alors qu'il faut
    enchaîner 5 victoires. Renvoie (probabilités de titre, taux de finale).
    """
    rng = np.random.default_rng(seed)
    flat = flatten(bracket)

    champions, finalists = [], []
    for _ in range(n_sim):
        current = flat
        while len(current) > 1:
            draws = rng.random(len(current) // 2)
            nxt = [a if u < P[(a, b)] else b
                   for (a, b), u in zip(zip(current[::2], current[1::2]), draws)]
            if len(nxt) == 2:
                finalists.extend(nxt)
            current = nxt
        champions.append(current[0])

    titles = pd.Series(champions).value_counts(normalize=True)
    finals = pd.Series(finalists).value_counts() / n_sim
    return titles, finals
