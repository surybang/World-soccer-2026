"""Sélection des équipes à conserver avant tout calcul.

Le dataset international_results de martj42 est plus large que les sélections
FIFA : il contient des équipes régionales et non reconnues (ConIFA, Island
Games, Jeux insulaires), comme l'Andalousie, l'Occitanie, Ynys Môn, Tamil Eelam
ou Jersey.

Pourquoi ce n'est pas qu'un détail cosmétique : **un Elo n'a de sens que si le
graphe des confrontations est connexe.** L'Andalousie joue l'Occitanie, qui joue
la Padanie : elles forment un bassin quasi isolé du reste du monde. Leurs notes
Elo dérivent dans leur coin et ne sont PAS sur la même échelle que celles des
sélections FIFA. Le modèle recevrait alors un nombre d'apparence comparable, mais
qui ne l'est pas. C'est plus dangereux qu'une feature naïve, parce que ça inspire
confiance.

Trois critères, du plus sémantique au plus mécanique. On les combine.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

# Compétitions dont la participation atteste d'une affiliation FIFA.
FIFA_TOURNAMENTS = (
    "FIFA World Cup",
    "FIFA World Cup qualification",
)


def fifa_affiliated(df):
    """Équipes ayant disputé au moins un match de Coupe du Monde ou de qualif.

    Critère le plus propre : quasiment toutes les fédérations membres de la FIFA
    entrent en qualification, et aucune équipe ConIFA ne le fait. Aucune liste à
    maintenir à la main.
    """
    fifa = df[df["tournament"].isin(FIFA_TOURNAMENTS)]
    return set(fifa["home_team"]) | set(fifa["away_team"])


def match_counts(df):
    return pd.concat([df["home_team"], df["away_team"]]).value_counts()


def largest_component(df):
    """Plus grande composante connexe du graphe des confrontations."""
    g = nx.Graph()
    g.add_edges_from(zip(df["home_team"], df["away_team"]))
    if g.number_of_nodes() == 0:
        return set()
    return max(nx.connected_components(g), key=len)


def diagnose(df):
    """Inventaire avant filtrage : composantes, équipes hors FIFA, petits volumes."""
    g = nx.Graph()
    g.add_edges_from(zip(df["home_team"], df["away_team"]))
    comps = sorted(nx.connected_components(g), key=len, reverse=True)

    fifa = fifa_affiliated(df)
    counts = match_counts(df)
    teams = set(counts.index)

    print(f"{len(teams)} équipes, {len(df)} matchs")
    print(f"{len(comps)} composante(s) connexe(s), "
          f"la principale en compte {len(comps[0])}")
    if len(comps) > 1:
        for c in comps[1:]:
            print(f"  isolée : {sorted(c)}")

    hors_fifa = teams - fifa
    print(f"\n{len(hors_fifa)} équipes sans aucun match FIFA "
          f"(Coupe du Monde ou qualification)")
    apercu = counts[list(hors_fifa)].sort_values(ascending=False)
    print(apercu.head(15).to_string())
    return comps, hors_fifa


def select_teams(df, min_matches=50, require_fifa=True, verbose=True):
    """Restreint le dataset aux équipes exploitables.

    Le filtrage est ITÉRATIF : retirer une équipe fait baisser le compteur de
    matchs de ses adversaires, qui peuvent à leur tour passer sous le seuil.
    On boucle jusqu'au point fixe.

    Renvoie (df filtré, équipes conservées).
    """
    keep = set(match_counts(df).index)
    if require_fifa:
        keep &= fifa_affiliated(df)

    out = df
    for _ in range(20):
        out = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]
        counts = match_counts(out)
        survivants = set(counts[counts >= min_matches].index)
        survivants &= largest_component(out)   # connexité du graphe Elo
        if survivants == keep:
            break
        keep = survivants

    out = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

    if verbose:
        retirees = len(set(match_counts(df).index)) - len(keep)
        print(f"{retirees} équipes retirées, {len(keep)} conservées")
        print(f"{len(df) - len(out)} matchs écartés, {len(out)} conservés "
              f"({len(out) / len(df):.1%})")

    return out.reset_index(drop=True), keep
