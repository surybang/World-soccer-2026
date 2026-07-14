# Coupe du Monde 2026

Prédiction de matchs internationaux, prolongement d'un TD de Machine Learnia
(cahier de vacances 2026). Le TD s'arrête à une RandomForest sur cinq features.
Ce projet reprend le problème depuis la donnée brute et mesure ce qu'apporte
chaque étape.

**Le modèle est coupé au 10 juin 2026**, la veille du coup d'envoi. Il n'a vu aucun
match du tournoi, ni à l'entraînement, ni dans les features qui alimentent ses
prédictions. Les pronostics et le rétro-test sont donc hors échantillon.

---

## Le résultat en une phrase

**Le gain vient de la représentation, pas du machine learning.** Le jeu de features
du TD atteint 0,561 de log loss. La formule d'Elo, publiée en 1960 pour les échecs et
appliquée sans le moindre apprentissage, atteint 0,456. Le meilleur modèle, après
vingt-deux features, augmentation miroir, sélection RFECV et soixante essais d'Optuna,
atteint 0,445.

Passer de la forme brute à un Elo rapporte **0,105**. Tout ce qu'on empile ensuite
rapporte **0,011**, soit **dix fois moins**.

---

## Résultats

Jeu de test : tous les matchs à partir de 2022, split chronologique (jamais aléatoire).

| Jeu de features | n | Modèle | Accuracy | ROC AUC | Log loss | Brier |
|---|---:|---|---:|---:|---:|---:|
| TD d'origine | 5 | Logistique | 0.7109 | 0.7556 | 0.5614 | 0.1907 |
| TD d'origine | 5 | XGBoost | 0.7116 | 0.7551 | 0.5609 | 0.1903 |
| *Elo brut (zéro apprentissage)* | *1* | *Formule de 1960* | *0.7848* | *0.8551* | *0.4559* | *0.1488* |
| Sans géographie | 17 | Logistique | 0.7872 | 0.8582 | 0.4499 | 0.1469 |
| Sans géographie | 17 | XGBoost | 0.7897 | 0.8573 | 0.4509 | 0.1470 |
| Complet | 22 | Logistique | 0.7888 | 0.8589 | 0.4482 | 0.1465 |
| Complet | 22 | XGBoost | 0.7869 | 0.8590 | 0.4473 | 0.1462 |
| Complet | 22 | XGBoost tuné | 0.7909 | 0.8589 | 0.4475 | 0.1462 |
| Complet | 22 | TabICL | 0.7912 | 0.8608 | 0.4462 | 0.1454 |
| Sélectionné (RFECV) | 12 | Logistique | 0.7866 | 0.8591 | 0.4478 | 0.1463 |
| Sélectionné (RFECV) | 12 | XGBoost tuné | 0.7903 | 0.8597 | 0.4463 | 0.1459 |
| **Sélectionné (RFECV)** | **12** | **XGBoost** | **0.7894** | **0.8606** | **0.4454** | **0.1455** |

> **Modèle déployé** : XGBoost sur les 12 variables retenues par RFECV, sans tuning.
> Meilleure log loss du benchmark, moitié moins de features, et pas d'Optuna dans le
> build.

### Lire ce tableau

**Le modèle ne compte pas.** À jeu de features constant, la régression logistique et
XGBoost sont à égalité, à quelques millièmes près. Ce n'est pas un accident : l'Elo
*est* un modèle de [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model),
soit une sigmoïde d'une différence de forces. Une logistique sur `diff_elo` reproduit
donc exactement la formule d'Elo. On a donné au modèle une feature qui contient déjà
la bonne forme fonctionnelle : il ne reste plus rien de non-linéaire à apprendre.

**Le tuning ne sert pas seulement à rien, il nuit légèrement.** Sur les deux jeux de
features, soixante essais d'Optuna dégradent la log loss (0,4454 vers 0,4463 sur douze
features ; 0,4473 vers 0,4475 sur vingt-deux). L'écart est du bruit, mais le signe est
constant : la recherche sur-optimise la validation croisée dans un espace où il n'y a
rien à trouver. Les hyperparamètres par défaut étaient déjà dans la zone plate.

**Douze features suffisent.** Le jeu sélectionné égale le jeu complet, et le dépasse
même sur les trois métriques probabilistes. Moins de features, moins de bruit, moins de
surapprentissage. La sélection ayant été faite sur le train uniquement, ce gain est
honnête.

**L'AUC de l'Elo brut est celle de `diff_elo` tout court.** La sigmoïde est strictement
monotone : elle ne change pas l'ordre des matchs, or l'AUC ne dépend que de l'ordre.
Tout le pipeline améliore le pouvoir discriminant de 0,0055 (0,8551 vers 0,8606). En
revanche, la formule d'Elo transforme un score en probabilité *calibrée*, et c'est là
que le modèle apporte un peu plus : il calibre mieux que les constantes fixes
d'eloratings.net.

### Pas besoin de modèle, alors ?

La question se pose sérieusement, et elle a trois réponses.

**Le gain existe, il est simplement petit.** 0,011 de log loss, ce n'est pas zéro. Une
partie vient de `diff_sos_10`, qui corrige l'Elo par la force du calendrier récent, une
autre de l'avantage du terrain, mieux estimé que la constante de 100 points de la
formule d'origine.

**La démarche vaut plus que le résultat.** Sans le pipeline complet, on ne pourrait pas
*savoir* que l'Elo suffit. C'est en construisant vingt-deux features, en les ablatant
par groupe et en comparant six modèles qu'on établit ce que la formule seule ne peut
pas prouver. Le modèle est ici l'instrument de mesure, pas la conclusion.

**Et la conclusion, elle, est robuste.** Elle arrive par quatre chemins indépendants :
l'ablation par groupe (l'Elo pèse quinze fois plus que tout le reste), l'égalité
logistique/XGBoost, l'échec du tuning, et l'Elo brut lui-même. Quatre mesures qui disent
la même chose, c'est ce qui rend un résultat solide.

---

## Décisions méthodologiques

### Périmètre : filtrer avant tout calcul

Le dataset contient des sélections non affiliées à la FIFA (ConIFA, Jeux insulaires :
Occitanie, Andalousie, Ynys Môn, Tamil Eelam). Ce n'est pas qu'une question de
propreté : **un Elo n'a de sens que si le graphe des confrontations est connexe.**
Ces équipes ne jouent qu'entre elles et forment une composante isolée. Leurs notes
dérivent sur une échelle sans commune mesure avec celle des sélections FIFA, tout en
ayant l'apparence de nombres comparables. Le risque n'est pas l'imprécision, c'est la
confiance injustifiée.

Filtrage **itératif** (retirer une équipe fait baisser le compteur de ses adversaires)
sur trois critères : affiliation FIFA (avoir disputé une qualification mondiale, ce qui
ne demande aucune liste à maintenir), volume minimal de 50 matchs, appartenance à la
composante connexe principale. Résultat : 117 équipes écartées sur 323, mais seulement
6,3% des matchs.

### Antisymétrie : une propriété à restaurer

Le TD entraîne sur « domicile contre extérieur » puis prédit en terrain neutre, où
cette asymétrie n'existe plus. Conséquence : `predict(A, B)` et `predict(B, A)` peuvent
se contredire, jusqu'à **22 points d'écart** sur un même match.

L'origine du problème est structurelle. Les modèles classiques de comparaisons par
paires (Thurstone 1927, Bradley-Terry 1952) posent `P(i bat j) = f(force_i - force_j)` :
l'antisymétrie y découle de la forme du modèle. Un RandomForest, lui, n'en sait rien.

Correction en deux temps :

| | violation moyenne | max |
|---|---:|---:|
| Sans miroir | 0.0591 | 0.220 |
| Entraîné en miroir | 0.0345 | 0.145 |
| **Miroir + moyennage** | **0.0000** | **0.000** |

L'augmentation miroir (chaque match dupliqué dans les deux sens, cible inversée)
*apprend* la symétrie sans la garantir. Le moyennage explicite à la prédiction
(`p = (p(A,B) + 1 - p(B,A)) / 2`) la rend **exacte, par arithmétique et non par
apprentissage**.

Vérifié : sur les matchs neutres du test, le modèle prédit 53,2% pour l'équipe A alors
que le taux réel est 56,1%, et il s'accorde à 81% avec « le meilleur Elo gagne ». Il
prédit bien la meilleure équipe, pas la position dans la ligne.

### Un décalage de distribution, corrigé

`prepare()` supprime les nuls (~25% des matchs) **après** avoir calculé la forme.
L'état courant des équipes doit donc être capturé **avant** cette suppression.

Le reconstruire depuis le DataFrame filtré donnait un `pts_10` surestimé de **+0,12
point par match**, pour 69% des équipes : les « dix derniers matchs » devenaient les
dix derniers matchs *décisifs*, et les nuls (qui ne rapportent qu'un point)
disparaissaient de la moyenne. Le modèle était entraîné sur une distribution et
interrogé sur une autre.

### Évaluation

**Split chronologique**, jamais aléatoire. Un `train_test_split(random_state=42)`
laisserait le modèle voir des matchs de 2025 pour en prédire de 2015. Ce n'est pas de
la fuite au sens strict, mais les métriques deviennent optimistes par rapport à l'usage
réel.

**Log loss et Brier** en métrique principale, pas l'accuracy. La simulation de tournoi
consomme des `predict_proba` : ce qui compte est la **calibration**. Un modèle à 66%
bien calibré est plus utile qu'un modèle à 67% qui annonce 90% de confiance à tort.

**Interpréter la log loss.** Elle ne se lit jamais dans l'absolu. La référence n'est pas
le hasard pur (0,693) mais la baseline informée : un modèle qui répondrait toujours
« 62% » (le taux de victoire à domicile) obtiendrait 0,663. Le modèle à 0,445 réduit
donc la log loss de **33%** par rapport à cette baseline.

### Sélection de features, faite honnêtement

Mesurer l'importance sur le test, retirer les features qui semblent inutiles, puis
réévaluer sur ce même test reviendrait à **choisir ses features en regardant la
réponse**. Tout se passe donc exclusivement sur le train, en validation croisée
temporelle. Le test n'est touché qu'une fois, à la fin.

**Features fantômes** (esprit Boruta) : pour chaque feature réelle, on ajoute une copie
permutée d'elle-même, son ombre. Une feature qui ne bat pas sa propre ombre n'apporte
pas plus qu'un bruit de même distribution. C'est le test statistique qui manque à un
simple classement d'importances : il donne un **seuil**, pas seulement un ordre.

**RFECV** avec `TimeSeriesSplit` : quel est le plus petit sous-ensemble qui ne dégrade
pas la performance ? On ne prend pas le minimum brut de log loss, mais le plus petit jeu
dans la tolérance.

---

## Ce qui décide réellement

Mesuré par permutation sur le test, par permutation par groupe et par ablation.

**`diff_elo` écrase tout** : 0,255 de dégradation à la permutation, contre 0,014 pour la
suivante. Un facteur 18.

**L'ablation par groupe** est encore plus nette. Retirer les deux features de force
coûte 0,0533 de log loss ; retirer n'importe quel autre groupe coûte moins de 0,004 :

| Groupe retiré | Coût (log loss) |
|---|---:|
| force (Elo) | 0.0533 |
| géographie | 0.0036 |
| forme récente | 0.0030 |
| contexte | 0.0023 |
| historique direct | 0.0019 |
| expérience | 0.0008 |

**Deux mesures complémentaires.** La permutation mesure ce que le modèle *entraîné*
utilise ; l'ablation mesure ce qu'un modèle *réentraîné* sans ces features saurait
faire. Une feature redondante avec l'Elo apparaîtra importante à la permutation et
inutile à l'ablation. Ici, les deux convergent.

**La géographie ne rapporte rien.** Cinq features (altitude, voyage, climat) pour 0,0036
de log loss, soit le même ordre que l'écart-type des mesures. Elle est conservée dans le
jeu sélectionné, mais elle ne pèse pas.

---

## Rétro-test sur le tournoi

Chaque match déjà joué, prédit avec l'état des équipes au 10 juin 2026. **Aucune mise à
jour en cours de route** : les quarts sont prédits sans connaître les huitièmes/seizièmes.

| | |
|---|---|
| Matchs prédits | à remplir |
| dont décisifs | 72 |
| dont nuls (hors champ du modèle) | à remplir |
| Bons pronostics | ~86% |
| Confiance moyenne | ~76% |
| Log loss | à remplir |

L'écart entre **réussite** et **confiance moyenne** est ce qu'il faut regarder : s'ils
coïncident, le modèle est calibré, ce qui est un résultat plus solide qu'une accuracy
élevée. Ici la réussite dépasse la confiance de dix points, ce qui suggère un modèle
légèrement **trop prudent**. Attention toutefois : sur 72 matchs, cet écart n'est pas
significatif.

### Pourquoi le modèle « se dégrade » en phase finale

| Phase | Matchs | Réussite | Confiance | Écart d'Elo moyen |
|---|---:|---:|---:|---:|
| Phase de groupes | 52 | 87% | 77% | 194 |
| Seizièmes | 13 | 100% | 78% | 202 |
| Huitièmes | 7 | 57% | 70% | 166 |

L'explication intuitive (« les équipes se valent davantage ») est juste, mais elle
n'opère pas sur la moyenne : l'écart d'Elo moyen ne chute quasiment pas. **Le tournoi ne
fait pas baisser l'écart de niveau moyen, il en réduit la dispersion.**

En phase de groupes, un quart des affiches dépassent 270 points d'écart : le modèle les
gagne les yeux fermés. En huitièmes, ces matchs faciles n'existent plus, les équipes
faibles ayant été éliminées. Il ne reste que des questions difficiles.

Le modèle n'a pas régressé : **son jeu d'examen a changé.** Et il l'annonce lui-même, sa
confiance moyenne descendant de 77% à 70%. C'est le signe d'un modèle calibré.

*Réserve : 13 puis 7 matchs. Ces trois lignes sont compatibles avec un modèle dont la
performance ne varierait pas du tout entre les phases.*

---

## Note sur TabICL

[TabICL](https://github.com/soda-inria/tabicl) est un **modèle de fondation tabulaire**
développé par l'équipe SODA d'Inria (Qu, Holzmüller, Varoquaux, Le Morvan). Son
fonctionnement n'a rien à voir avec celui d'un GBM : il ne fait pas de descente de
gradient sur nos données. Pré-entraîné sur des millions de tables synthétiques, il
« lit » notre jeu de données en une seule passe avant d'un transformer et prédit dans la
foulée. C'est de l'*in-context learning* : `y_pred = model(X_train, y_train, X_test)`,
sans mise à jour de paramètres.

**Ses résultats ici sont excellents.** Il obtient la meilleure ROC AUC (0,8608) et le
meilleur Brier (0,1454) du benchmark, **sans aucun tuning, sans sélection de features et
sans entraînement**, là où le modèle retenu a bénéficié d'un RFECV et d'hyperparamètres
choisis. C'est cohérent avec ce qu'annoncent ses auteurs : sur TabArena et TALENT,
TabICLv2 sans réglage dépasse l'état de l'art et bat des XGBoost, CatBoost ou LightGBM
lourdement tunés sur environ 80% des jeux.

**Pourquoi on ne le retient pas.** Son inférence est en `O(n² + nm²)` sur la taille du
contexte, et le contexte, c'est le jeu d'entraînement lui-même : nos ~17 000 lignes
passent dans le transformer à chaque `predict`. Sur GPU c'est une affaire de secondes ;
sur CPU c'est lent, et Streamlit Community Cloud ne fournit pas de GPU. Le coût est
structurel, pas accidentel : il découle du mécanisme d'attention qui fait sa force.

Deux conséquences pratiques dans ce projet : on lui passe le train **sans augmentation
miroir** (doubler le contexte quadruplerait le coût pour un gain nul, l'antisymétrie
étant rétablie au moment de la prédiction), et `run_benchmark` chronomètre 100 lignes
avant de lancer l'inférence complète.

**Le choix de XGBoost est un choix de déploiement, pas un jugement de qualité.** Sur un
poste équipé d'un GPU, TabICL serait le candidat naturel, et l'on gagnerait en prime
toute l'étape de tuning.

Références : [TabICL, ICML 2025](https://arxiv.org/abs/2502.05564) ·
[TabICLv2, 2026](https://arxiv.org/abs/2602.11139)

---

## Structure

```txt
World-soccer-2026/
├── data/
│   ├── results.csv              martj42/international_results
│   └── country_centroids.csv
│
├── src/world_soccer_2026/
│   ├── utils.py                 fourni avec le TD
│   ├── selection.py             périmètre des équipes (FIFA, graphe connexe)
│   ├── geo.py                   altitude, distance, choc climatique
│   ├── features.py              Elo, forme, head-to-head, cadrage A contre B
│   ├── benchmark.py             split temporel, pipelines, métriques, Optuna
│   ├── explain.py               permutation, importance par groupe, SHAP
│   ├── selection_features.py    features fantômes, RFECV
│   ├── tournament.py            probabilités par paire, bracket, Monte-Carlo
│   ├── backtest.py              rejoue le tournoi et note le modèle
│   ├── fixtures.py              matchs restants, date de coupure (11 juin 2026)
│   │
│   ├── build_artifacts.py       À LANCER UNE FOIS -> produit artifacts/
│   ├── predict.py               inférence pure, ne lit que artifacts/
│   └── app.py                   interface Streamlit
│
├── artifacts/
│   ├── model.joblib             XGBoost, 12 features, coupé au 10 juin 2026
│   ├── teams.parquet            état courant de chaque équipe
│   ├── h2h.json                 bilan des confrontations directes
│   ├── backtest.parquet         rétro-test, calculé au build
│   └── meta.json                date de coupure, liste des features du modèle
│
├── notebooks/projet_02_bis.ipynb
├── pyproject.toml
└── README.md
```

**L'app ne lit ni le notebook ni `results.csv`.** `build_artifacts.py` fait tout le
travail lourd en local et produit `artifacts/` (~230 Ko) ; `predict.py` fait de
l'inférence pure en ~10 ms ; `app.py` affiche. C'est ce qui rend le déploiement possible
sur Community Cloud, qui plafonne à 1 Go de RAM et redémarre souvent.

## Lancer

```shell
uv sync
uv run python -m world_soccer_2026.build_artifacts
uv run streamlit run src/world_soccer_2026/app.py
```

---

## Limites

### Ce que le modèle ne peut structurellement pas faire

**Il ne prédit pas les nuls.** Écartés de l'entraînement, soit environ un quart des
matchs. Les probabilités affichées sont conditionnelles à l'existence d'un vainqueur :
utilisable en phase finale, où le nul se résout aux tirs au but, pas en phase de
groupes, où le classement dépend des points.

**Il ne voit que le score.** Ni composition d'équipe, ni blessure, ni xG. Une France
privée de trois titulaires conserve son Elo. C'est le plus gros angle mort, et aucune
feature dérivée des colonnes existantes ne peut le combler.

**Il ignore la trajectoire.** L'Elo est markovien : il connaît la note courante d'une
équipe, pas sa dynamique. Une génération qui monte et une équipe qui décline sont
interchangeables à Elo égal.

### Décalages entre entraînement et prédiction

**`travel_km` change de sens en phase finale.** La feature mesure la distance entre la
base d'une équipe et le stade. Elle a été apprise sur des qualifs et des amicaux, où
l'équipe arrive deux jours avant. En Coupe du Monde, les sélections sont sur place
depuis des semaines : la France en finale à New York n'a pas parcouru 6 000 km, elle a
pris un vol interne. Le modèle appliquerait un handicap qui n'existe plus. **Une feature
dont le sens change entre l'entraînement et la prédiction est pire qu'une feature
absente.** D'où le drapeau `neutraliser_geo`, actif par défaut en contexte de tournoi.

**`rest_days` est un héritage, pas une prévision.** Le snapshot donne l'écart entre les
deux derniers matchs *avant la coupure*, et non le repos réel avant le match à venir.
Dans le rétro-test, la valeur est identique pour tous les matchs d'une même équipe,
alors que les tours s'enchaînent tous les trois à cinq jours.

**L'effet altitude est extrapolé.** Il est appris presque entièrement sur des qualifs
sud-américaines, où la Bolivie reçoit à La Paz sans que l'adversaire ait le temps de
s'acclimater. En Coupe du Monde, les sélections disposent de stages de préparation.
L'extrapolation n'est pas vérifiée. Sans conséquence en 2026 (stades au niveau de la
mer), mais elle invaliderait un tournoi joué à Mexico.

### Choix de conception discutables

**La table d'altitude est curatée à la main.** Elle couvre les villes qui portent le
signal (Andes, Mexique, plateau éthiopien) et met zéro partout ailleurs.

**Le climat est approximé par la latitude.** Reykjavik et Moscou n'ont pas le même climat
à latitude comparable. La vraie donnée serait la température le jour du match.

**Les constantes de l'Elo ne sont pas optimisées.** Le K de 20 à 60, le diviseur 400 et
le bonus de 100 points à domicile viennent de la spécification eloratings.net. Personne
ne les a réglées sur ces données. Vu que l'Elo *est* le modèle, c'est probablement là que
se trouve le gain le plus accessible.

**Le périmètre dépend d'un seuil arbitraire.** `min_matches=50` écarte 117 sélections.

### Ce que les mesures ne démontrent pas

**Le rétro-test porte sur trop peu de matchs.** 72 rencontres, dont 7 en huitièmes. Les
tranches de calibration comptent une dizaine de matchs chacune : elles illustrent, elles
ne démontrent pas.

**L'avantage sur l'Elo brut n'est pas établi.** 0,4454 contre 0,4559, soit 0,0105 de log
loss, du même ordre que la dispersion entre les modèles du haut du tableau (0,4454 à
0,4482). Un bootstrap serait nécessaire, et il ne conclurait probablement pas.

**La supériorité de TabICL non plus.** Il gagne dans 72,9% des rééchantillonnages
bootstrap, là où il en faudrait environ 95% pour conclure (p < 0,05).

**La sélection de features a été faite une seule fois**, avec une seed fixée. Un autre
tirage donnerait peut-être un autre sous-ensemble.

### Portée

**Un seul tournoi, un seul sport, une seule époque.** Rien ne garantit que ces
conclusions tiennent sur des championnats de clubs, où les effectifs changent moins et
où la fréquence des matchs est bien plus élevée.

---

## Pour aller plus loin

Les trois premières pistes améliorent le modèle. Les suivantes s'attaquent à la vraie
limite : ce projet a montré que **le modèle n'est pas le goulot d'étranglement**.

**Calibrer les constantes de l'Elo.** Un grid search sur K, le diviseur et le bonus à
domicile rapporterait probablement plus que tout ce qu'on a empilé au-dessus. Si l'Elo
est le modèle, autant régler l'Elo plutôt que XGBoost.

**Poisson bivarié / Dixon-Coles.** Modéliser le nombre de buts de chaque équipe plutôt
que le vainqueur. C'est la référence sur le football, et les nuls redeviennent
modélisables.

**Modèle hiérarchique.** Chaque match est ici traité comme une observation indépendante,
alors que la même équipe apparaît des centaines de fois.

**Classification à trois classes**, pour récupérer les ~25% de matchs nuls écartés.

**Des données d'une autre nature.** C'est ici qu'est la limite structurelle. Le dataset
ne contient qu'une chose : qui a joué contre qui, et combien de buts. Aucune information
sur *comment* le match s'est joué. Or l'Elo résume déjà toute cette information dans un
seul nombre, ce qui explique qu'aucune feature dérivée des mêmes colonnes n'apporte quoi
que ce soit : elles sont toutes des reformulations du même signal.

- **Les xG (expected goals).** Une équipe qui gagne 1-0 avec 0,4 xG contre 2,1 a été
  chanceuse ; le score ment sur sa performance réelle. C'est le principal correctif au
  bruit du score, et ce que les modèles professionnels utilisent en priorité
  (Understat, FBref, StatsBomb).
- **La composition d'équipe.** L'Elo suit une sélection, pas un effectif. Probablement
  le plus gros angle mort.
- **Les blessures et suspensions**, connues avant le match et intégrées par le marché
  bien avant nous.
- **Les données de tracking** (positions, pressing, distances), qui décrivent le jeu
  lui-même.

Il ne s'agit donc pas d'avoir *plus de lignes* : trente ans d'historique suffisent
largement. Il s'agit d'avoir des **colonnes que le score ne peut pas produire**.

**Suivi d'expériences.** Ce projet a comparé une dizaine de configurations dont les
résultats ont été recopiés à la main. MLflow aurait tracé automatiquement paramètres,
métriques et artefacts, et versionné les modèles **avec leur signature d'entrée**, ce qui
aurait détecté en amont les divergences entre le jeu de features du modèle et celui de
l'inférence.