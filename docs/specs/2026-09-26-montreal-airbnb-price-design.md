# Design — Prédiction du prix des logements Airbnb à Montréal

Date : 2026-09-26
Statut : validé (en attente de relecture finale)

## 1. Objectif et périmètre

**Question :** combien peut coûter une nuit dans un logement Airbnb à Montréal, selon ses caractéristiques et son emplacement ?

**Public visé :** recruteurs Data Scientist / ML. Le projet doit montrer le cycle complet : collecte, nettoyage, exploration, modélisation, évaluation, déploiement.

### Données

| Source | Contenu | Licence |
|--------|---------|---------|
| [Inside Airbnb — Montréal](https://insideairbnb.com/get-the-data/) | `listings.csv.gz`, instantané le plus récent (un seul) | CC BY 4.0 |
| Données ouvertes STM / Montréal | Coordonnées des stations de métro | Licence ouverte |

La distance au centre-ville est calculée par rapport à Place Ville-Marie (45.5017, -73.5673). Aucun fichier supplémentaire.

### Inclus

- Types de logement : logement entier et chambre privée.
- Prix par nuit entre 20 $ et 1 000 $ (seuils justifiés dans le notebook d'exploration).
- Cible : `log(prix)`. Les prédictions sont reconverties en dollars pour l'affichage et les métriques.

### Exclu (listé dans « Pistes d'amélioration » du README)

- NLP sur les descriptions, analyse des photos.
- Séries temporelles, saisonnalité.
- Deep learning.

### Critères de réussite

1. Repo GitHub propre, README clair avec graphiques.
2. Modèle final nettement meilleur que la référence naïve (médiane du quartier) sur le jeu de test, comparaison présentée honnêtement.
3. App Streamlit déployée sur Streamlit Community Cloud, lien dans le README.

## 2. Structure du repo et pipeline

```
montreal-airbnb-price/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                ← ignoré par git
│   └── processed/          ← ignoré par git
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modelisation.ipynb
├── src/
│   ├── __init__.py
│   ├── download.py
│   ├── clean.py
│   ├── features.py
│   └── train.py
├── models/                 ← modèle final versionné (.joblib)
├── app/
│   └── streamlit_app.py
└── tests/
```

### Flux de données

1. **`src/download.py`** : télécharge `listings.csv.gz` et les stations de métro dans `data/raw/`.
2. **`src/clean.py`** : lit `data/raw/`, convertit le prix texte (`"$1,200.00"` → `1200.0`), filtre les types de logement et les prix hors bornes, garde environ 15 colonnes utiles, traite les valeurs manquantes, écrit `data/processed/listings_clean.parquet`.
3. **`src/features.py`** : ajoute la distance (haversine, en km) au métro le plus proche et au centre-ville, le nombre d'équipements et des indicateurs binaires (Wi-Fi, climatisation, stationnement, cuisine).
4. **`notebooks/01_exploration.ipynb`** : distribution des prix, prix par quartier, carte folium des annonces.
5. **`notebooks/02_modelisation.ipynb`** + **`src/train.py`** : comparaison des modèles dans le notebook, entraînement et sauvegarde du modèle final dans `models/` par `train.py`.
6. **`app/streamlit_app.py`** : charge le modèle sauvegardé et prédit.

Reproduction complète en 3 commandes :

```bash
python -m src.download
python -m src.train
streamlit run app/streamlit_app.py
```

`train.py` appelle lui-même `clean` et `features` : pas d'étape manuelle intermédiaire.

### Stack

Python 3.11+, pandas, numpy, pyarrow, matplotlib, seaborn, folium, scikit-learn, joblib, streamlit, pytest.

## 3. Modélisation et évaluation

### Découpage

- 80 % entraînement / 20 % test (`random_state` fixé). Le test reste intact jusqu'à l'évaluation finale.
- Comparaison des modèles par validation croisée à 5 plis sur l'entraînement.

### Modèles

| # | Modèle | Rôle |
|---|--------|------|
| 0 | Médiane du prix par quartier | Référence naïve |
| 1 | Ridge | Linéaire, interprétable |
| 2 | RandomForestRegressor | Non linéaire |
| 3 | HistGradientBoostingRegressor | Candidat principal |

Tous les modèles passent par un `Pipeline` scikit-learn : `ColumnTransformer` (one-hot pour les catégorielles, standardisation pour les numériques dans Ridge), puis le modèle. L'app réutilise ce pipeline tel quel.

Réglage léger des hyperparamètres (`RandomizedSearchCV`, environ 20 itérations) sur le meilleur modèle uniquement.

### Métriques

- **MAE en dollars** (principale).
- **R²**.
- Calculées sur les prix reconvertis en dollars (`exp` de la prédiction).

### Pièges évités

- Fuite de données : exclusion des colonnes dérivées du prix (revenus estimés, etc.).
- Aucun choix de modélisation fait en regardant le jeu de test.

### Interprétation

- Importance par permutation sur le jeu de test.
- Analyse des erreurs par quartier et par type de logement.

Aucun score n'est fixé à l'avance ; le README présente les résultats obtenus.

## 4. App, tests, README

### App Streamlit

- Barre latérale : quartier, type de logement, chambres, lits, salles de bain, capacité, équipements (cases à cocher), station de métro proche (sert à calculer les distances).
- Zone principale : prix estimé par nuit, fourchette indicative (± MAE du modèle), carte de localisation.
- Si `models/` est vide : message explicite invitant à lancer `python -m src.train`.
- Déploiement : Streamlit Community Cloud, connecté au repo GitHub.

### Tests (pytest)

- `clean.py` : conversion `"$1,200.00"` → `1200.0` ; lignes sans prix retirées ; filtrage des bornes de prix.
- `features.py` : haversine correcte sur deux points connus ; une annonce située sur une station a une distance métro ≈ 0.
- Test de bout en bout sur un petit jeu fictif (10 lignes) : nettoyage → features → entraînement → prédiction, sans téléchargement.

### README

1. Titre, accroche, lien vers l'app, capture d'écran.
2. Contexte et question.
3. Données : sources, licence, date de l'instantané.
4. 2–3 graphiques clés.
5. Résultats : tableau de comparaison des modèles et variables importantes.
6. Limites et pistes d'amélioration.
7. Reproduire le projet (3 commandes).

### Git

- `.gitignore` : `data/raw/`, `data/processed/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/`.
- Commits petits et fréquents, faits par le propriétaire du repo.

## 5. Ajustements après prototypage (2026-09-26)

Un prototype complet a été exécuté sur l'instantané du 2026-06-15 (10 656 annonces, 9 165 gardées). Il a conduit à ces ajustements, qui priment sur les sections précédentes :

- **Instantané figé** : l'URL de l'instantané 2026-06-15 est fixée dans `src/config.py` pour que les résultats soient reproductibles.
- **Stations de métro** : extraites du GTFS de la STM (`stops.txt`, 68 stations) et versionnées dans `data/external/metro_stations.csv`, car l'app déployée en a besoin.
- **Pas de `data/processed/`** : `load_dataset()` recalcule nettoyage + variables en ~2 s.
- **Modules ajoutés** : `src/config.py` (chemins, constantes) et `src/model.py` (modèles, référence naïve, validation croisée).
- **App à la racine** : `streamlit_app.py` à la racine du repo (imports de `src` directs, nom attendu par défaut par Streamlit Community Cloud). Le dossier `app/` est supprimé.
- **Graphiques** : sauvegardés dans `figures/` (versionné) pour le README.
- **Variables du modèle** : on exclut les notes, le nombre d'avis et le statut superhost, que l'utilisateur de l'app ne peut pas connaître pour un nouveau logement. On garde `minimum_nights`, qui s'avère la variable la plus importante : environ la moitié des annonces imposent 31 nuits minimum, avec un prix médian de 90 $ contre 245 $.
- **Salles de bain** : lues depuis `bathrooms_text` (0,1 % de valeurs manquantes) plutôt que `bathrooms` (21 %).
- **Valeurs manquantes** : imputées par la médiane dans le `Pipeline` (et non dans `clean.py`) pour éviter toute fuite entre entraînement et test.
- **Cible** : `TransformedTargetRegressor(func=np.log, inverse_func=np.exp)`, donc `predict()` renvoie directement des dollars.
- **Fourchette de l'app** : prix × (1 ± erreur relative médiane sur le test) au lieu de ± MAE, qui donnait des bornes absurdes pour les petits prix.
- **Dépendances** : `requirements.txt` (exécution de l'app, versions figées) et `requirements-dev.txt` (notebooks, tests).
