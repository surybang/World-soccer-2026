"""Features géographiques : altitude, distance parcourue, choc climatique.

Idée directrice : ce n'est PAS l'altitude ou le froid du stade qui compte, c'est
**l'inadaptation relative de l'équipe à ce stade**. Jouer à 3 600 m ne pénalise
pas la Bolivie, et jouer à -20 °C ne pénalise pas la Norvège. Toutes les features
d'ici sont donc des ÉCARTS entre l'équipe et le lieu, jamais des caractéristiques
du lieu seul.

Second choix de conception : la « base » de chaque équipe (son altitude et sa
latitude d'origine) est DÉRIVÉE DES DONNÉES, à partir des stades où elle joue ses
matchs à domicile. Aucune table à maintenir, et ça gère sans effort les cas
tordus (Angleterre, Écosse et Pays de Galles ne sont pas des pays ISO).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Altitude des stades
# ---------------------------------------------------------------------------
# L'effet altitude est extrêmement concentré : une poignée de villes portent
# quasiment tout le signal. En dessous de ~1 000 m l'effet physiologique est
# négligeable, donc un défaut à 0 est une approximation acceptable.
# Altitudes en mètres.

CITY_ALTITUDE = {
    # Andes :
    "La Paz": 3640, "El Alto": 4150, "Potosí": 4090, "Oruro": 3706,
    "Cochabamba": 2558, "Sucre": 2810, "Santa Cruz de la Sierra": 416,
    "Quito": 2850, "Ambato": 2577, "Riobamba": 2754, "Cuenca": 2560,
    "Bogotá": 2640, "Bogota": 2640, "Tunja": 2820, "Manizales": 2160,
    "Pasto": 2527, "Medellín": 1495, "Medellin": 1495,
    "Cusco": 3399, "Juliaca": 3825, "Arequipa": 2335, "Huancayo": 3271,
    "Lima": 154, "La Rioja": 498, "Toluca": 2660,
    # Mexique et Amérique centrale
    "Mexico City": 2240, "Ciudad de México": 2240, "Guadalajara": 1566,
    "Puebla": 2135, "Pachuca": 2400, "San José": 1170, "San Jose": 1170,
    "Guatemala City": 1500, "Tegucigalpa": 990, "Quetzaltenango": 2333,
    # Afrique de l'Est et australe
    "Addis Ababa": 2355, "Asmara": 2325, "Nairobi": 1795, "Kampala": 1190,
    "Kigali": 1567, "Bujumbura": 774, "Johannesburg": 1753, "Pretoria": 1339,
    "Bloemfontein": 1395, "Harare": 1490, "Lusaka": 1279, "Gaborone": 1014,
    "Maseru": 1600, "Mbabane": 1243, "Antananarivo": 1276, "Windhoek": 1655,
    "Sana'a": 2250, "Sanaa": 2250,
    # Asie centrale et Himalaya
    "Thimphu": 2334, "Kathmandu": 1400, "Bishkek": 800, "Almaty": 848,
    "Kabul": 1790, "Lhasa": 3656, "Kunming": 1892,
    # Europe (effet marginal mais présent)
    "Madrid": 667, "Andorra la Vella": 1023, "Sofia": 595, "Munich": 519,
    "Ankara": 938, "Erzurum": 1890, "Yerevan": 989, "Tehran": 1200,
    "Denver": 1609, "Colorado Springs": 1839, "Salt Lake City": 1288,
}

# Seuil physiologique : en dessous, on considère qu'il n'y a pas de contrainte.
ALTITUDE_FLOOR = 1000.0


def city_altitude(city) -> float:
    return float(CITY_ALTITUDE.get(str(city).strip(), 0.0))


# ---------------------------------------------------------------------------
# Coordonnées
# ---------------------------------------------------------------------------
# Le dataset international_results emploie des noms de pays qui ne suivent pas
# toujours la nomenclature ISO. On mappe les écarts connus.

ALIASES = {
    "United States": "United States", "USA": "United States",
    "South Korea": "Korea, Republic of", "North Korea": "Korea, Democratic People's Republic of",
    "Ivory Coast": "Côte d'Ivoire", "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo, The Democratic Republic of the",
    "Congo": "Congo", "Republic of Ireland": "Ireland",
    "Czech Republic": "Czechia", "Czechia": "Czechia",
    "Turkey": "Türkiye", "Türkiye": "Türkiye",
    "Russia": "Russian Federation", "Iran": "Iran, Islamic Republic of",
    "Syria": "Syrian Arab Republic", "Vietnam": "Viet Nam",
    "Laos": "Lao People's Democratic Republic", "Brunei": "Brunei Darussalam",
    "Taiwan": "Taiwan, Province of China", "Chinese Taipei": "Taiwan, Province of China",
    "Hong Kong": "Hong Kong", "Macau": "Macao",
    "Bolivia": "Bolivia, Plurinational State of",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Tanzania": "Tanzania, United Republic of", "Moldova": "Moldova, Republic of",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "North Macedonia": "North Macedonia", "Macedonia": "North Macedonia",
    "Palestine": "Palestine, State of", "Swaziland": "Eswatini",
    "Eswatini": "Eswatini", "Cabo Verde": "Cabo Verde",
    # Nations britanniques : pas des pays ISO, coordonnées approchées
    "England": None, "Scotland": None, "Wales": None, "Northern Ireland": None,
}

MANUAL_COORDS = {
    "England": (52.5, -1.5), "Scotland": (56.8, -4.2),
    "Wales": (52.3, -3.7), "Northern Ireland": (54.6, -6.7),
    "Kosovo": (42.6, 21.0), "Curaçao": (12.2, -69.0),
    "Tahiti": (-17.7, -149.4), "New Caledonia": (-21.3, 165.5),
    "Réunion": (-21.1, 55.5), "Zanzibar": (-6.1, 39.3),
}


def load_coords(path="data/country_centroids.csv"):
    """Renvoie un dict {nom de pays -> (lat, lon)} adapté au vocabulaire du dataset."""
    table = pd.read_csv(path)
    base = {r.country: (r.latitude, r.longitude) for r in table.itertuples()}

    coords = dict(base)
    for local, iso in ALIASES.items():
        if iso and iso in base:
            coords[local] = base[iso]
    coords.update(MANUAL_COORDS)
    return coords


def coverage(names, coords):
    """Noms non résolus. À exécuter une fois pour compléter MANUAL_COORDS."""
    return sorted({n for n in names if n not in coords})


def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points, en vectorisé."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Base d'origine de chaque équipe, dérivée des données
# ---------------------------------------------------------------------------

def team_home_base(df, coords):
    """Altitude et latitude d'origine de chaque équipe.

    On les lit dans les données : on prend les stades où l'équipe joue ses
    matchs À DOMICILE (donc hors terrain neutre) et on en prend la médiane.
    La Bolivie ressort à ~3 600 m sans qu'on ait rien à coder.

    Repli sur le centroïde du pays si l'équipe n'a jamais reçu.
    """
    home = df[~df["neutral"].astype(bool)]
    alt = home.groupby("home_team").apply(
        lambda g: np.median([city_altitude(c) for c in g["city"]]),
        include_groups=False)

    lat = home.groupby("home_team")["country"].agg(
        lambda s: np.median([coords[c][0] for c in s if c in coords])
        if any(c in coords for c in s) else np.nan)
    lon = home.groupby("home_team")["country"].agg(
        lambda s: np.median([coords[c][1] for c in s if c in coords])
        if any(c in coords for c in s) else np.nan)

    base = pd.DataFrame({"home_alt": alt, "home_lat": lat, "home_lon": lon})

    # équipes qui n'ont jamais reçu : centroïde de leur pays
    all_teams = set(df["home_team"]) | set(df["away_team"])
    for t in all_teams - set(base.index):
        if t in coords:
            base.loc[t] = [0.0, coords[t][0], coords[t][1]]

    base["home_alt"] = base["home_alt"].fillna(0.0)
    return base


# ---------------------------------------------------------------------------
# Features géographiques par match
# ---------------------------------------------------------------------------

GEO_STATS = ["alt_shock", "travel_km", "lat_shift", "climate_shift"]


def add_geo(df, coords=None, base=None):
    """Ajoute, pour les deux équipes, leur inadaptation au lieu du match.

    Pour chaque équipe :
      alt_shock     : mètres de dénivelé subis (0 si le stade est plus bas ou
                      sous le seuil physiologique). Monter coûte, descendre non.
      travel_km     : distance entre sa base et le stade.
      lat_shift     : |latitude du stade - sa latitude|, proxy du dépaysement.
      climate_shift : écart de latitude ABSOLUE, proxy du choc thermique
                      (un Norvégien au Qatar, un Brésilien en Norvège).
    """
    if coords is None:
        coords = load_coords()
    if base is None:
        base = team_home_base(df, coords)

    venue_alt = df["city"].map(city_altitude).to_numpy(dtype=float)
    v_lat = df["country"].map(lambda c: coords.get(c, (np.nan, np.nan))[0]).to_numpy(dtype=float)
    v_lon = df["country"].map(lambda c: coords.get(c, (np.nan, np.nan))[1]).to_numpy(dtype=float)

    for prefix in ("home", "away"):
        team = df[f"{prefix}_team"]
        t_alt = team.map(base["home_alt"]).to_numpy(dtype=float)
        t_lat = team.map(base["home_lat"]).to_numpy(dtype=float)
        t_lon = team.map(base["home_lon"]).to_numpy(dtype=float)

        # seul le fait de MONTER coûte, et seulement au-dessus du seuil
        effective = np.maximum(venue_alt, ALTITUDE_FLOOR)
        df[f"{prefix}_alt_shock"] = np.maximum(
            0.0, np.where(venue_alt < ALTITUDE_FLOOR, 0.0, effective - t_alt))

        df[f"{prefix}_travel_km"] = haversine(t_lat, t_lon, v_lat, v_lon)
        df[f"{prefix}_lat_shift"] = np.abs(v_lat - t_lat)
        df[f"{prefix}_climate_shift"] = np.abs(np.abs(v_lat) - np.abs(t_lat))

    return df
