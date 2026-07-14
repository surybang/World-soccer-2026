"""Bac à sable : deux équipes, un stade, une probabilité.

L'intérêt de cette démo n'est pas de dire qui gagne (n'importe quel modèle le
fait), c'est de rendre TANGIBLE le fait que le LIEU est une entrée du modèle.
Le curseur d'altitude est là pour ça : on voit la Bolivie basculer.
"""

import numpy as np
import pandas as pd
import streamlit as st

import predict
from world_soccer_2026.fixtures import DEMI_FINALES, FINALE, WC_START

st.set_page_config(page_title="Pronostic de match international",
                   page_icon="⚽", layout="centered")


@st.cache_resource
def charger():
    predict.load()          # met en cache modèle + snapshots
    return predict.available_teams(), predict.load()[3]


equipes, meta = charger()

st.title("Coupe du Monde 2026")
st.caption(
    f"XGBoost entraîné sur {meta['n_matchs']:,} matchs internationaux, "
    f"**coupé au {meta['coupure']}**, la veille du coup d'envoi. Le modèle "
    "n'a jamais vu un seul match de ce tournoi, ni dans son entraînement, ni "
    "dans les données qui alimentent ses features. Les pronostics ci-dessous "
    "sont donc réellement hors échantillon."
    .replace(",", " ")
)

# ============================================================ MATCHS À VENIR
st.header("Le dernier carré")

demi, titres = predict.bracket_probabilities(DEMI_FINALES, FINALE)

for m in DEMI_FINALES:
    p = predict.match_proba(m["a"], m["b"], m["city"], m["country"])
    favori = m["a"] if p >= .5 else m["b"]
    st.markdown(f"**{m['label']}** — {m['date']}, {m['city']}")
    c1, c2, c3 = st.columns([3, 2, 3])
    c1.metric(m["a"], f"{p:.0%}")
    c3.metric(m["b"], f"{1 - p:.0%}")
    c2.markdown(
        f"<div style='text-align:center;padding-top:18px;color:#888'>"
        f"{'serré' if abs(p - .5) < .06 else favori + ' favori'}</div>",
        unsafe_allow_html=True)
    st.progress(float(p))

st.subheader("Qui soulève la coupe ?")
st.caption(
    "Probabilités EXACTES, pas simulées : avec quatre équipes il n'y a que "
    "quatre chemins possibles, on les énumère. Le Monte-Carlo n'est nécessaire "
    "que sur un tableau complet, où il y a des milliards de chemins.")

tab = pd.DataFrame({"atteint la finale": demi, "gagne le tournoi": titres})
st.bar_chart(tab["gagne le tournoi"])
st.dataframe(tab.style.format("{:.1%}"), use_container_width=True)

st.info(
    "Remarque la marche entre les deux colonnes : atteindre la finale à 60% "
    "puis la gagner à 55% ne fait que 33% de titre. **Enchaîner deux matchs "
    "serrés est bien plus dur que gagner un match serré.** C'est ce que les "
    "pronostics déterministes cachent systématiquement.")

st.divider()

# ================================================== LE MODÈLE AVAIT-IL RAISON ?
bt = predict.load_backtest()

if not bt.empty:
    import backtest as bt_mod

    st.header("Le modèle avait-il raison ?")
    st.caption(
        f"Chaque match déjà joué du tournoi, prédit avec l'état des équipes au "
        f"{meta['coupure']}. Aucune mise à jour en cours de route : les quarts "
        "sont prédits sans savoir ce qui s'est passé en huitièmes.")

    r = bt_mod.summarise(bt)

    c1, c2, c3 = st.columns(3)
    c1.metric("Bons pronostics", f"{r['accuracy']:.0%}",
              help=f"sur les {r['n_decisifs']} matchs décisifs")
    c2.metric("Confiance moyenne", f"{r['confiance_moyenne']:.0%}",
              help="ce que le modèle s'accordait en moyenne")
    c3.metric("Nuls", r["n_nuls"],
              help="Le modèle est binaire : il ne peut structurellement pas "
                   "prévoir un nul. Ces matchs sont exclus du score, pas cachés.")

    ecart = r["accuracy"] - r["confiance_moyenne"]
    if abs(ecart) < .05:
        st.success(
            f"Le modèle tient sa promesse : il s'accordait "
            f"{r['confiance_moyenne']:.0%} de confiance, il a eu raison "
            f"{r['accuracy']:.0%} du temps. C'est ça, un modèle calibré.")
    elif ecart < 0:
        st.warning(
            f"Le modèle était trop sûr de lui : {r['confiance_moyenne']:.0%} "
            f"de confiance annoncée pour {r['accuracy']:.0%} de réussite.")
    else:
        st.info(
            f"Le modèle était trop prudent : {r['accuracy']:.0%} de réussite "
            f"pour {r['confiance_moyenne']:.0%} de confiance annoncée.")

    onglet1, onglet2, onglet3 = st.tabs(
        ["Par tour", "Les plus grosses erreurs", "Tous les matchs"])

    with onglet1:
        st.dataframe(
            bt_mod.par_phase(bt).style.format(
                {"reussite": "{:.0%}", "confiance": "{:.0%}"}),
            use_container_width=True)
        st.caption(
            "Les tours avancés sont plus durs à prédire, et ce n'est pas un "
            "défaut du modèle : plus on avance, plus les équipes se ressemblent. "
            "Un huitième oppose souvent un favori à un outsider, une demi-finale "
            "oppose deux favoris.")

    with onglet2:
        st.caption("Forte confiance, mauvais pronostic. Les vraies surprises.")
        st.dataframe(
            bt_mod.surprises(bt).style.format({"confiance": "{:.0%}"}),
            use_container_width=True, hide_index=True)

    with onglet3:
        vue = bt.copy()
        vue["pronostic"] = vue.apply(
            lambda x: f"{x['favori']} ({x['confiance']:.0%})", axis=1)
        vue["issue"] = np.where(
            vue["nul"], "nul (hors champ)",
            np.where(vue["correct"].fillna(False), "correct", "raté"))
        st.dataframe(
            vue[["date", "phase", "equipe_a", "equipe_b", "score",
                 "pronostic", "issue"]],
            use_container_width=True, hide_index=True)

    with st.expander("Le modèle tient-il ses promesses ? (calibration)"):
        st.dataframe(
            bt_mod.calibration(bt).style.format(
                {"annonce": "{:.0%}", "observe": "{:.0%}"}),
            use_container_width=True)
        st.caption(
            "Chaque ligne : quand le modèle annonçait cette confiance, à quelle "
            "fréquence avait-il raison ? Les deux colonnes doivent se ressembler. "
            "C'est plus important que le taux de réussite brut : une simulation "
            "de tournoi ne vaut que ce que valent ses probabilités.")

    st.divider()

# ================================================================ BAC À SABLE
st.header("Bac à sable")
st.caption("Choisis deux équipes et un stade.")

# ---------------------------------------------------------------- les équipes
col1, col2 = st.columns(2)
with col1:
    a = st.selectbox("Équipe A", equipes,
                     index=equipes.index("France") if "France" in equipes else 0)
with col2:
    reste = [e for e in equipes if e != a]
    b = st.selectbox("Équipe B", reste,
                     index=reste.index("Brazil") if "Brazil" in reste else 1)

# ------------------------------------------------------------------ le stade
st.subheader("Où se joue le match ?")

villes = predict.known_venues()
labels = [f"{v} ({alt:,.0f} m)".replace(",", " ") for v, alt in villes]

c1, c2 = st.columns([2, 1])
with c1:
    choix = st.selectbox("Stade", ["Terrain neutre au niveau de la mer"] + labels)
with c2:
    terrain = st.radio("Avantage du terrain",
                       ["Neutre", "A reçoit", "B reçoit"], index=0)

if choix.startswith("Terrain neutre"):
    ville, pays, altitude = "Madrid", "Spain", 0.0
else:
    i = labels.index(choix)
    ville, altitude = villes[i][0], float(villes[i][1])
    pays = "Spain"          # la latitude du stade compte peu ici

altitude = st.slider(
    "Altitude du stade (m)", 0, 4000, int(altitude), step=50,
    help="Le modèle a appris que MONTER coûte, et seulement au-dessus de "
         "1 000 m. Une équipe andine n'est pas pénalisée chez elle.")

enjeu = st.select_slider(
    "Enjeu", options=["Amical", "Qualification", "Continental",
                      "Grand tournoi", "Coupe du Monde"],
    value="Coupe du Monde")
importance = ["Amical", "Qualification", "Continental",
              "Grand tournoi", "Coupe du Monde"].index(enjeu)

home_adv = {"Neutre": 0, "A reçoit": 1, "B reçoit": -1}[terrain]

# ---------------------------------------------------------------- prédiction
p = predict.match_proba(a, b, ville, pays, importance=importance,
                        home_advantage=home_adv, venue_altitude=altitude)

st.divider()
gagnant, proba = (a, p) if p >= .5 else (b, 1 - p)
st.metric(f"{gagnant} l'emporte", f"{proba:.0%}")
st.progress(float(p))
st.caption(f"{a} {p:.1%}  ·  {b} {1 - p:.1%}")

if abs(p - .5) < .06:
    st.info("Match très serré : le modèle ne tranche pas vraiment.")

# ------------------------------------------------------------------ pourquoi
with st.expander("Pourquoi ce pronostic ?"):
    st.dataframe(predict.explain(a, b, ville, pays, venue_altitude=altitude),
                 use_container_width=True)
    st.markdown(
        "**elo** : force ajustée à l'adversaire. C'est le critère dominant.  \n"
        "**sos_10** : Elo moyen des 10 derniers adversaires. Un `pts_10` élevé "
        "contre un `sos_10` faible, c'est une forme en trompe-l'oeil.  \n"
        "**alt_shock** : mètres de dénivelé subis par l'équipe.  \n"
        "**travel_km** : distance parcourue jusqu'au stade.")

# -------------------------------------------------- l'effet du lieu, en courbe
with st.expander("L'effet de l'altitude, en continu"):
    paliers = list(range(0, 4001, 250))
    courbe = [predict.match_proba(a, b, ville, pays, importance=importance,
                                  home_advantage=home_adv, venue_altitude=alt)
              for alt in paliers]
    st.line_chart(pd.DataFrame({f"P({a} gagne)": courbe}, index=paliers))
    st.caption(
        "Si les deux équipes viennent du niveau de la mer, la courbe est plate : "
        "l'altitude les pénalise autant l'une que l'autre. Essaie avec la Bolivie "
        "ou l'Équateur, qui sont chez elles en altitude.")

st.divider()
st.caption(
    "Antisymétrie garantie : inverser A et B donne exactement la probabilité "
    "complémentaire. Le modèle est interrogé dans les deux sens puis moyenné.")