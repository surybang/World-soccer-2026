"""Coupe du Monde 2026 : pronostics, rétro-test et bac à sable."""

import numpy as np
import pandas as pd
import streamlit as st

from world_soccer_2026 import predict
from world_soccer_2026.fixtures import DEMI_FINALES, FINALE

st.set_page_config(page_title="Coupe du Monde 2026", page_icon="⚽",
                   layout="centered")


@st.cache_resource
def charger():
    predict.load()
    return predict.available_teams(), predict.load()[3]


equipes, meta = charger()

st.title("Coupe du Monde 2026")
st.caption(
    f"XGBoost entraîné sur {meta['n_matchs']} matchs internationaux, coupé au "
    f"{meta['coupure']}, la veille du coup d'envoi. Le modèle n'a vu aucun match "
    "de ce tournoi, ni à l'entraînement, ni dans les données qui alimentent ses "
    "features.")

# 1. MATCHS À VENIR
st.header("Le dernier carré")

demi, titres = predict.bracket_probabilities(DEMI_FINALES, FINALE)

for m in DEMI_FINALES:
    p = predict.match_proba(m["a"], m["b"], m["city"], m["country"])
    st.markdown(f"**{m['label']}** · {m['date']} · {m['city']}")
    c1, c2 = st.columns(2)
    c1.metric(m["a"], f"{p:.0%}")
    c2.metric(m["b"], f"{1 - p:.0%}")
    st.progress(float(p))

st.subheader("Qui soulève la coupe ?")

tab = pd.DataFrame({"atteint la finale": demi, "gagne le tournoi": titres})
st.bar_chart(tab["gagne le tournoi"])
st.dataframe(tab.style.format("{:.1%}"), width='stretch')

st.caption(
    "L'écart entre les deux colonnes est le point à retenir : atteindre la "
    "finale à 60% puis la gagner à 55% ne fait que 33% de titre. Enchaîner deux "
    "matchs serrés est nettement plus dur que gagner un match serré.")

st.divider()

# 2. LE MODÈLE AVAIT-IL RAISON ?
bt = predict.load_backtest()

if not bt.empty:
    from world_soccer_2026 import backtest as bt_mod

    st.header("Le modèle avait-il raison ?")
    st.caption(
        f"Chaque match déjà joué, prédit avec l'état des équipes au "
        f"{meta['coupure']}. Aucune mise à jour en cours de route : les quarts "
        "sont prédits sans connaître les huitièmes.")

    r = bt_mod.summarise(bt)

    c1, c2, c3 = st.columns(3)
    c1.metric("Bons pronostics", f"{r['accuracy']:.0%}",
              help=f"sur les {r['n_decisifs']} matchs décisifs")
    c2.metric("Confiance moyenne", f"{r['confiance_moyenne']:.0%}")
    c3.metric("Nuls", r["n_nuls"],
              help="Le modèle est binaire. Ces matchs sont exclus du score, "
                   "pas dissimulés.")

    ecart = r["accuracy"] - r["confiance_moyenne"]
    if abs(ecart) < .05:
        st.success(
            f"Le modèle s'accordait {r['confiance_moyenne']:.0%} de confiance "
            f"et a eu raison {r['accuracy']:.0%} du temps. Il est calibré.")
    elif ecart < 0:
        st.warning(
            f"Trop sûr de lui : {r['confiance_moyenne']:.0%} de confiance "
            f"annoncée pour {r['accuracy']:.0%} de réussite.")
    else:
        st.info(
            f"Trop prudent : {r['accuracy']:.0%} de réussite pour "
            f"{r['confiance_moyenne']:.0%} de confiance annoncée.")

    o1, o2, o3 = st.tabs(["Par tour", "Les plus grosses erreurs", "Tous les matchs"])

    with o1:
        st.dataframe(
            bt_mod.par_phase(bt).style.format(
                {"reussite": "{:.0%}", "confiance": "{:.0%}"}),
            width='stretch')
        st.caption(
            "Les tours avancés sont plus durs à prédire, et ce n'est pas un "
            "défaut du modèle : un huitième oppose souvent un favori à un "
            "outsider, une demi-finale oppose deux favoris.")

    with o2:
        st.caption("Forte confiance, mauvais pronostic.")
        st.dataframe(bt_mod.surprises(bt).style.format({"confiance": "{:.0%}"}),
                     width='stretch', hide_index=True)

    with o3:
        vue = bt.copy()
        vue["pronostic"] = vue.apply(
            lambda x: f"{x['favori']} ({x['confiance']:.0%})", axis=1)
        vue["issue"] = np.where(
            vue["nul"], "nul (hors champ)",
            np.where(vue["correct"].fillna(False), "correct", "raté"))
        st.dataframe(
            vue[["date", "phase", "equipe_a", "equipe_b", "score",
                 "pronostic", "issue"]],
            width='stretch', hide_index=True)

    with st.expander("Calibration : le modèle tient-il ses promesses ?"):
        st.dataframe(
            bt_mod.calibration(bt).style.format(
                {"annonce": "{:.0%}", "observe": "{:.0%}"}),
            width='stretch')
        st.caption(
            "Quand le modèle annonçait cette confiance, à quelle fréquence "
            "avait-il raison ? Les effectifs sont faibles (une dizaine de matchs "
            "par tranche), donc ce tableau illustre plus qu'il ne démontre.")

    st.divider()

# 3. BAC À SABLE
st.header("Bac à sable")
st.caption("Deux équipes, un stade.")

c1, c2 = st.columns(2)
with c1:
    a = st.selectbox("Équipe A", equipes,
                     index=equipes.index("France") if "France" in equipes else 0)
with c2:
    reste = [e for e in equipes if e != a]
    b = st.selectbox("Équipe B", reste,
                     index=reste.index("Brazil") if "Brazil" in reste else 1)

c1, c2 = st.columns(2)
with c1:
    ville, pays = predict.VENUES[st.selectbox("Stade", list(predict.VENUES))]
with c2:
    terrain = st.radio("Terrain", ["Neutre", "A reçoit", "B reçoit"],
                       index=0, horizontal=True)

neutraliser = st.checkbox(
    "Contexte tournoi (équipes acclimatées, déjà sur place)", value=True,
    help="Les features géographiques (voyage, altitude, climat) ont été apprises "
         "sur des qualifs et des amicaux, où l'équipe arrive deux jours avant. "
         "En Coupe du Monde, les équipes sont sur place depuis des semaines : "
         "ces handicaps n'existent plus. Décoche pour voir l'effet brut.")

home_adv = {"Neutre": 0, "A reçoit": 1, "B reçoit": -1}[terrain]

p = predict.match_proba(a, b, ville, pays, importance=4,
                        home_advantage=home_adv,
                        neutraliser_geo=neutraliser)

gagnant, proba = (a, p) if p >= .5 else (b, 1 - p)
st.metric(f"{gagnant} l'emporte", f"{proba:.0%}")
st.progress(float(p))
st.caption(f"{a} {p:.1%} · {b} {1 - p:.1%}")

if abs(p - .5) < .06:
    st.info("Le modèle ne tranche pas vraiment.")

with st.expander("Pourquoi ce pronostic ?"):
    shap_df = predict.explain_match(a, b, ville, pays, home_advantage=home_adv)

    if shap_df is not None:
        st.caption(
            "Contributions SHAP, en log-odds. Positif pousse vers "
            f"{a}, négatif vers {b}. La somme des contributions donne l'écart "
            "entre la probabilité prédite et la probabilité moyenne du modèle.")
        st.dataframe(
            shap_df.style.format({"contribution": "{:+.3f}", "valeur": "{:.1f}"}),
            width='stretch')
    else:
        st.caption("Installe `shap` pour voir le détail des contributions.")

    st.markdown("**Écarts bruts entre les deux équipes**")
    st.dataframe(predict.compare(a, b, ville, pays), width='stretch')

st.divider()
st.caption(
    "Antisymétrie garantie : inverser A et B donne exactement la probabilité "
    "complémentaire. Le modèle est interrogé dans les deux sens, puis moyenné.")
