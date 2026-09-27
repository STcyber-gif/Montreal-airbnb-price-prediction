# Plan d'implémentation — Prédiction du prix Airbnb à Montréal

> Plan à suivre tâche par tâche, dans l'ordre. Chaque étape se coche (`- [ ]` → `- [x]`).
> Les commits sont faits **par toi** : chaque tâche se termine par les commandes git à lancer.

**Objectif :** prédire le prix par nuit d'un logement Airbnb à Montréal, du téléchargement des données jusqu'à une app Streamlit en ligne.

**Architecture :** des modules Python dans `src/` (téléchargement → nettoyage → variables → modèles → entraînement), testés avec pytest ; deux notebooks qui racontent l'analyse ; une app Streamlit à la racine qui charge le modèle sauvegardé.

**Stack :** Python 3.12, pandas, numpy, scikit-learn, joblib, Streamlit, matplotlib, seaborn, folium, Jupyter, pytest.

**Référence :** [docs/specs/2026-09-26-montreal-airbnb-price-design.md](../specs/2026-09-26-montreal-airbnb-price-design.md). La section 5 de la spec liste les ajustements faits après prototypage.

Tout le code de ce plan a été exécuté sur les vraies données (instantané du 2026-06-15) : 15 tests passent, et l'entraînement donne une MAE d'environ 52 $ sur le jeu de test.

---

## Structure finale du repo

```
montreal-airbnb-price/
├── README.md
├── requirements.txt          ← dépendances de l'app (utilisé par Streamlit Cloud)
├── requirements-dev.txt      ← + notebooks et tests
├── pytest.ini
├── .gitignore
├── streamlit_app.py          ← l'app de démo
├── data/
│   ├── raw/                  ← listings.csv.gz (ignoré par git)
│   └── external/
│       └── metro_stations.csv  ← 68 stations (versionné)
├── docs/
│   ├── specs/
│   └── plans/
├── figures/                  ← graphiques pour le README (versionnés)
├── models/
│   └── model.joblib          ← modèle final (versionné, ~19 Mo)
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modelisation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             ← chemins et constantes
│   ├── download.py           ← téléchargement des données
│   ├── clean.py              ← nettoyage
│   ├── features.py           ← création des variables
│   ├── model.py              ← modèles comparés + validation croisée
│   └── train.py              ← entraînement et sauvegarde du modèle final
└── tests/
    ├── conftest.py           ← données fictives partagées
    ├── test_download.py
    ├── test_clean.py
    ├── test_features.py
    ├── test_model.py
    └── test_app.py
```

**Convention :** toutes les commandes se lancent dans un terminal ouvert **à la racine du projet** (`montreal-airbnb-price/`), dans **Anaconda Prompt**, avec l'environnement `airbnb-mtl` activé (voir Tâche 0).

---

### Tâche 0 : Installer Python et préparer l'environnement

Python est déjà disponible grâce à Anaconda : on crée un environnement dédié au projet.

**Fichiers :**
- Créer : `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `pyproject.toml`, `src/__init__.py`, `src/config.py`
- Modifier : `.gitignore`

- [ ] **Étape 1 : Ouvrir « Anaconda Prompt »**

Python est fourni par Anaconda (`C:\ProgramData\anaconda3`), qui n'est pas dans le PATH de Windows : `python` et `conda` ne marchent donc que dans **Anaconda Prompt** (menu Démarrer). Place-toi dans le projet :

```bash
cd "C:\Users\sidim\Documents\Personal Project\montreal-airbnb-price"
```

- [ ] **Étape 2 : Créer l'environnement conda du projet**

Un environnement isole les bibliothèques du projet : on ne touche pas à `base`, qui contient les paquets d'Anaconda. On prend Python 3.12, la version choisie aussi pour le déploiement.

```bash
conda create -n airbnb-mtl python=3.12 -y
```

Active-le (à refaire à chaque nouveau terminal) :

```bash
conda activate airbnb-mtl
```

Une fois activé, `(airbnb-mtl)` s'affiche au début de la ligne. Vérifie :

```bash
python --version
```

Attendu : `Python 3.12.x`

- [ ] **Étape 3 : Créer `requirements.txt`**

Ce fichier contient seulement ce dont l'app a besoin : Streamlit Cloud l'installera. Les versions sont figées, parce qu'un modèle sauvegardé avec une version de scikit-learn doit être rechargé avec la même version.

```text
pandas==3.0.6
numpy==2.5.3
scikit-learn==1.9.1
joblib==1.6.0
streamlit==1.64.0
```

- [ ] **Étape 4 : Créer `requirements-dev.txt`**

```text
-r requirements.txt
matplotlib==3.11.2
seaborn==0.13.2
folium==0.20.0
jupyter==1.1.1
pytest==9.1.1
```

- [ ] **Étape 5 : Installer les dépendances**

```bash
python -m pip install -r requirements-dev.txt
```

Attendu : se termine par `Successfully installed ...` (quelques minutes).

- [ ] **Étape 6 : Créer `pytest.ini`**

`pythonpath = .` permet aux tests de faire `from src... import ...`.

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Étape 7 : Créer `src/__init__.py`**

Fichier vide. Il indique à Python que `src/` est un package importable.

- [ ] **Étape 8 : Créer `src/config.py`**

Tous les chemins et constantes au même endroit : on ne les réécrit jamais ailleurs.

```python
"""Chemins et constantes partagés par tout le projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
MODELS_DIR = ROOT / "models"

# Instantané Inside Airbnb utilisé pour le projet (licence CC BY 4.0).
# Pour changer d'instantané : copier la nouvelle URL depuis https://insideairbnb.com/get-the-data/
LISTINGS_URL = "https://data.insideairbnb.com/canada/qc/montreal/2026-06-15/data/listings.csv.gz"
# Données GTFS de la STM (horaires et arrêts), dont on extrait les stations de métro.
GTFS_URL = "https://www.stm.info/sites/default/files/gtfs/gtfs_stm.zip"

LISTINGS_RAW = RAW_DIR / "listings.csv.gz"
METRO_STATIONS = EXTERNAL_DIR / "metro_stations.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

# Place Ville-Marie, utilisée comme « centre-ville ».
DOWNTOWN_LAT, DOWNTOWN_LON = 45.5017, -73.5673

PRICE_MIN, PRICE_MAX = 20, 1000
ROOM_TYPES = ["Entire home/apt", "Private room"]

RANDOM_STATE = 42
```

- [ ] **Étape 9 : Vérifier**

```bash
python -c "from src.config import ROOT; import sklearn, streamlit; print(ROOT)"
```

Attendu : le chemin complet de `montreal-airbnb-price` s'affiche, sans erreur.

- [ ] **Étape 10 : Rendre `src` importable depuis n'importe où (notebooks compris)**

Sans ça, `from src... import ...` ne marche que si Python est lancé depuis la racine du projet. Un notebook tourne souvent depuis un autre dossier, et on obtient alors `ModuleNotFoundError: No module named 'src'`. On déclare donc le projet comme un package et on l'installe en mode « éditable » (`-e`) : Python va lire directement le dossier `src/`, et tes modifications sont prises en compte sans réinstaller.

Crée `pyproject.toml` à la racine :

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "montreal-airbnb-price"
version = "0.1.0"
requires-python = ">=3.12"

[tool.setuptools]
packages = ["src"]
```

Installe-le dans l'environnement :

```bash
python -m pip install -e .
```

Ajoute cette ligne à `.gitignore` (dossier généré par l'installation) :

```text
*.egg-info/
```

Vérifie depuis un autre dossier :

```bash
cd ..
python -c "from src.config import ROOT; print(ROOT)"
cd montreal-airbnb-price
```

Attendu : le chemin de `montreal-airbnb-price` s'affiche.

- [ ] **Étape 11 : Commit (à faire toi-même)**

```bash
git add requirements.txt requirements-dev.txt pytest.ini pyproject.toml .gitignore src/__init__.py src/config.py
git commit -m "chore: prépare l'environnement Python et la configuration"
```

---

### Tâche 1 : Télécharger les données

**Fichiers :**
- Créer : `tests/test_download.py`, `src/download.py`
- Généré : `data/raw/listings.csv.gz` (ignoré par git), `data/external/metro_stations.csv` (versionné)

**Contexte :** le fichier `stops.txt` de la STM contient, pour chaque station, une ligne « parent » (`location_type` = 1, `stop_id` = `STATION_M...`) et des lignes « quai » et « accès » qui pointent vers elle via `parent_station`. On garde un quai par station, parce que son nom est mieux écrit (« Station Angrignon » plutôt que « STATION ANGRIGNON »).

- [ ] **Étape 1 : Écrire le test qui échoue** — `tests/test_download.py`

```python
import pandas as pd

from src.download import extract_metro_stations


def test_extract_metro_stations_keeps_one_row_per_station():
    # Extrait simplifié du fichier stops.txt de la STM (tout est lu en texte).
    stops = pd.DataFrame(
        {
            "stop_id": ["STATION_M118", "43", "43-01", "STATION_M700", "70", "51234"],
            "stop_name": [
                "STATION ANGRIGNON",
                "Station Angrignon",
                "Station Angrignon",
                "STATION LONGUEUIL",
                "Station Longueuil-Université de Sherbrooke -Zone B",
                "Arrêt de bus Sherbrooke / Saint-Denis",
            ],
            "stop_lat": ["45.446397", "45.446466", "45.446319", "45.525", "45.524", "45.518"],
            "stop_lon": ["-73.603293", "-73.603118", "-73.603835", "-73.522", "-73.521", "-73.568"],
            "location_type": ["1", "0", "2", "1", "0", "0"],
            "parent_station": [None, "STATION_M118", "STATION_M118", None, "STATION_M700", None],
        }
    )
    stations = extract_metro_stations(stops)
    assert stations["name"].tolist() == ["Angrignon", "Longueuil-Université de Sherbrooke"]
    assert stations.loc[0, "lat"] == 45.446466
    assert list(stations.columns) == ["name", "lat", "lon"]
```

- [ ] **Étape 2 : Vérifier qu'il échoue**

```bash
python -m pytest tests/test_download.py -v
```

Attendu : ÉCHEC avec `ModuleNotFoundError: No module named 'src.download'`

- [ ] **Étape 3 : Écrire `src/download.py`**

```python
"""Télécharge les données brutes : annonces Airbnb et stations de métro."""
import io
import urllib.request
import zipfile

import pandas as pd

from src.config import GTFS_URL, LISTINGS_RAW, LISTINGS_URL, METRO_STATIONS


def download_listings() -> None:
    LISTINGS_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {LISTINGS_URL} ...")
    urllib.request.urlretrieve(LISTINGS_URL, LISTINGS_RAW)
    print(f"-> {LISTINGS_RAW}")


def extract_metro_stations(stops: pd.DataFrame) -> pd.DataFrame:
    """Garde une ligne par station de métro à partir du fichier stops.txt du GTFS."""
    parent = stops["parent_station"].fillna("")
    is_metro_platform = (stops["location_type"] == "0") & parent.str.startswith("STATION_M")
    stations = stops[is_metro_platform].drop_duplicates("parent_station")
    names = (
        stations["stop_name"]
        .str.removeprefix("Station ")
        .str.replace(r"\s*-Zone B$", "", regex=True)
    )
    return pd.DataFrame(
        {
            "name": names.to_numpy(),
            "lat": stations["stop_lat"].astype(float).to_numpy(),
            "lon": stations["stop_lon"].astype(float).to_numpy(),
        }
    ).sort_values("name", ignore_index=True)


def download_metro_stations() -> None:
    METRO_STATIONS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {GTFS_URL} (~45 Mo) ...")
    with urllib.request.urlopen(GTFS_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("stops.txt") as f:
        stops = pd.read_csv(f, dtype=str)
    stations = extract_metro_stations(stops)
    stations.to_csv(METRO_STATIONS, index=False)
    print(f"-> {METRO_STATIONS} ({len(stations)} stations)")


if __name__ == "__main__":
    download_listings()
    download_metro_stations()
```

- [ ] **Étape 4 : Vérifier que le test passe**

```bash
python -m pytest tests/test_download.py -v
```

Attendu : `1 passed`

- [ ] **Étape 5 : Télécharger les vraies données**

```bash
python -m src.download
```

Attendu : deux lignes `-> ...`, la dernière se terminant par `(68 stations)`. Le fichier `data/raw/listings.csv.gz` fait environ 6 Mo.

- [ ] **Étape 6 : Vérifier ce que git voit**

```bash
git status
```

Attendu : `data/external/` apparaît, mais **pas** `data/raw/` (ignoré grâce au `.gitignore`).

- [ ] **Étape 7 : Commit (à faire toi-même)**

```bash
git add tests/test_download.py src/download.py data/external/metro_stations.csv
git commit -m "feat: télécharge les annonces Airbnb et les stations de métro STM"
```

---

### Tâche 2 : Nettoyer les annonces

**Fichiers :**
- Créer : `tests/conftest.py`, `tests/test_clean.py`, `src/clean.py`

**Contexte :** dans le fichier brut, le prix est du texte (`"$1,200.00"`) et 11 % des annonces n'ont pas de prix. Le nombre de salles de bain est plus fiable dans `bathrooms_text` (0,1 % de valeurs manquantes) que dans `bathrooms` (21 %). On ne garde que les colonnes utiles : les colonnes calculées à partir du prix (`estimated_revenue_l365d`, `price_quote_*`) permettraient au modèle de « tricher ».

- [ ] **Étape 1 : Créer les données fictives partagées** — `tests/conftest.py`

pytest charge automatiquement ce fichier : chaque test peut recevoir `raw_listings` ou `stations` en paramètre.

```python
"""Petites données fictives partagées par les tests (aucun téléchargement)."""
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Berri-UQAM", "Mont-Royal", "Atwater"],
            "lat": [45.515299, 45.524518, 45.489796],
            "lon": [-73.561273, -73.581870, -73.586078],
        }
    )


@pytest.fixture
def raw_listings() -> pd.DataFrame:
    """40 annonces au format d'Inside Airbnb, avec quelques lignes à filtrer."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "id": range(n),
            "price": [f"${p:,.2f}" for p in rng.uniform(50, 400, n)],
            "room_type": ["Entire home/apt", "Private room"] * (n // 2),
            "neighbourhood_cleansed": ["Ville-Marie", "Le Plateau-Mont-Royal", "Verdun", "Rosemont"] * (n // 4),
            "latitude": rng.uniform(45.45, 45.55, n),
            "longitude": rng.uniform(-73.65, -73.55, n),
            "accommodates": rng.integers(1, 7, n),
            "bedrooms": rng.integers(1, 4, n).astype(float),
            "beds": rng.integers(1, 5, n).astype(float),
            "bathrooms_text": ["1 bath", "1.5 shared baths", "Half-bath", "2 baths"] * (n // 4),
            "amenities": [json.dumps(["Wifi", "Kitchen"]), json.dumps(["Air conditioning"])] * (n // 2),
            "minimum_nights": rng.integers(1, 32, n),
            "estimated_revenue_l365d": rng.uniform(0, 50000, n),  # colonne « fuite » à ignorer
        }
    )
    df.loc[0, "price"] = None  # prix manquant
    df.loc[1, "price"] = "$5,000.00"  # prix aberrant
    df.loc[2, "room_type"] = "Shared room"  # type exclu
    return df
```

- [ ] **Étape 2 : Écrire les tests qui échouent** — `tests/test_clean.py`

```python
import numpy as np
import pandas as pd

from src.clean import clean_listings, parse_bathrooms, parse_price


def test_parse_price_handles_dollar_sign_and_commas():
    prices = parse_price(pd.Series(["$1,200.00", "$85.50", None]))
    assert prices.iloc[0] == 1200.0
    assert prices.iloc[1] == 85.5
    assert np.isnan(prices.iloc[2])


def test_parse_bathrooms():
    baths = parse_bathrooms(pd.Series(["1 bath", "1.5 shared baths", "Shared half-bath", None]))
    assert baths.iloc[0] == 1.0
    assert baths.iloc[1] == 1.5
    assert baths.iloc[2] == 0.5
    assert np.isnan(baths.iloc[3])


def test_clean_listings_filters_rows(raw_listings):
    clean = clean_listings(raw_listings)
    assert len(clean) == 37  # 40 - prix manquant - prix aberrant - chambre partagée
    assert clean["price"].between(20, 1000).all()
    assert set(clean["room_type"]) == {"Entire home/apt", "Private room"}


def test_clean_listings_drops_leaky_columns(raw_listings):
    clean = clean_listings(raw_listings)
    assert "estimated_revenue_l365d" not in clean.columns
    assert "bathrooms_text" not in clean.columns
    assert {"bathrooms", "bath_shared"} <= set(clean.columns)
```

- [ ] **Étape 3 : Vérifier qu'ils échouent**

```bash
python -m pytest tests/test_clean.py -v
```

Attendu : ÉCHEC avec `ModuleNotFoundError: No module named 'src.clean'`

- [ ] **Étape 4 : Écrire `src/clean.py`**

```python
"""Nettoyage des annonces brutes d'Inside Airbnb."""
import pandas as pd

from src.config import PRICE_MAX, PRICE_MIN, ROOM_TYPES

# Colonnes gardées. On exclut volontairement les colonnes calculées à partir
# du prix (estimated_revenue_l365d, price_quote_*) pour éviter les fuites de données.
COLUMNS = [
    "id",
    "price",
    "room_type",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "amenities",
    "minimum_nights",
]


def parse_price(prices: pd.Series) -> pd.Series:
    """'$1,200.00' -> 1200.0 ; valeur manquante ou illisible -> NaN."""
    return pd.to_numeric(prices.str.replace(r"[$,]", "", regex=True), errors="coerce")


def parse_bathrooms(texts: pd.Series) -> pd.Series:
    """'1.5 shared baths' -> 1.5 ; 'Half-bath' -> 0.5 ; manquant -> NaN."""
    numbers = pd.to_numeric(texts.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    is_half = texts.str.contains("half", case=False, na=False)
    return numbers.mask(is_half, 0.5)


def clean_listings(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[COLUMNS].copy()
    df["price"] = parse_price(df["price"])
    df = df[df["room_type"].isin(ROOM_TYPES)]
    # between() renvoie False pour les NaN : les annonces sans prix sont retirées aussi.
    df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
    df["bathrooms"] = parse_bathrooms(df["bathrooms_text"])
    df["bath_shared"] = df["bathrooms_text"].str.contains("shared", case=False, na=False).astype(int)
    return df.drop(columns=["bathrooms_text"]).reset_index(drop=True)
```

- [ ] **Étape 5 : Vérifier que les tests passent**

```bash
python -m pytest tests/test_clean.py -v
```

Attendu : `4 passed`

- [ ] **Étape 6 : Commit (à faire toi-même)**

```bash
git add tests/conftest.py tests/test_clean.py src/clean.py
git commit -m "feat: nettoie les annonces (prix, salles de bain, filtres)"
```

---

### Tâche 3 : Créer les variables (features)

**Fichiers :**
- Créer : `tests/test_features.py`, `src/features.py`

**Contexte :** la formule de haversine donne la distance « à vol d'oiseau » entre deux points GPS. Pour trouver la station la plus proche, on calcule d'un coup une matrice de distances (annonces × 68 stations) avec numpy, puis on prend le minimum de chaque ligne. `amenities` est une liste JSON sous forme de texte : on la lit avec `json.loads`.

- [ ] **Étape 1 : Écrire les tests qui échouent** — `tests/test_features.py`

```python
import pytest

from src.clean import clean_listings
from src.features import FEATURES, add_features, distance_to_nearest_station, haversine_km


def test_haversine_same_point_is_zero():
    assert haversine_km(45.5, -73.5, 45.5, -73.5) == pytest.approx(0.0)


def test_haversine_montreal_to_quebec_city():
    # Place Ville-Marie -> Château Frontenac : environ 233 km à vol d'oiseau
    assert haversine_km(45.5017, -73.5673, 46.8119, -71.2050) == pytest.approx(233, abs=5)


def test_distance_to_nearest_station_on_a_station(stations):
    distances = distance_to_nearest_station([45.515299, 45.60], [-73.561273, -73.50], stations)
    assert distances[0] == pytest.approx(0.0, abs=0.01)  # pile sur Berri-UQAM
    assert distances[1] > 5  # loin de toutes les stations


def test_add_features_creates_model_columns(raw_listings, stations):
    df = add_features(clean_listings(raw_listings), stations)
    assert set(FEATURES) <= set(df.columns)
    assert "amenities" not in df.columns
    first = df.iloc[0]  # équipements : ["Wifi", "Kitchen"] ou ["Air conditioning"]
    assert first["n_amenities"] == first["has_wifi"] + first["has_kitchen"] + first["has_ac"]
```

- [ ] **Étape 2 : Vérifier qu'ils échouent**

```bash
python -m pytest tests/test_features.py -v
```

Attendu : ÉCHEC avec `ModuleNotFoundError: No module named 'src.features'`

- [ ] **Étape 3 : Écrire `src/features.py`**

`FEATURES` est la liste officielle des colonnes que voit le modèle. Le modèle, l'entraînement et l'app l'importent tous d'ici.

```python
"""Création des variables utilisées par le modèle."""
import json

import numpy as np
import pandas as pd

from src.config import DOWNTOWN_LAT, DOWNTOWN_LON

EARTH_RADIUS_KM = 6371.0

# Nom de la colonne -> mot-clé cherché dans la liste des équipements.
AMENITY_FLAGS = {
    "has_wifi": "wifi",
    "has_ac": "air conditioning",
    "has_parking": "parking",
    "has_kitchen": "kitchen",
}

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "latitude",
    "longitude",
    "dist_metro_km",
    "dist_downtown_km",
    "n_amenities",
    "bath_shared",
    *AMENITY_FLAGS,
]
CATEGORICAL_FEATURES = ["neighbourhood_cleansed", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance à vol d'oiseau (km) entre deux points GPS. Accepte des tableaux numpy."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_to_nearest_station(lat, lon, stations: pd.DataFrame) -> np.ndarray:
    """Pour chaque point, distance (km) à la station de métro la plus proche."""
    lat = np.asarray(lat, dtype=float)[:, None]  # une ligne par annonce, une colonne par station
    lon = np.asarray(lon, dtype=float)[:, None]
    distances = haversine_km(lat, lon, stations["lat"].to_numpy(), stations["lon"].to_numpy())
    return distances.min(axis=1)


def add_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_metro_km"] = distance_to_nearest_station(df["latitude"], df["longitude"], stations)
    df["dist_downtown_km"] = haversine_km(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(), DOWNTOWN_LAT, DOWNTOWN_LON
    )
    amenities = df["amenities"].fillna("[]").apply(json.loads)
    df["n_amenities"] = amenities.apply(len)
    amenities_text = amenities.apply(lambda items: " | ".join(items).lower())
    for column, keyword in AMENITY_FLAGS.items():
        df[column] = amenities_text.str.contains(keyword, regex=False).astype(int)
    return df.drop(columns=["amenities"])
```

- [ ] **Étape 4 : Vérifier que les tests passent**

```bash
python -m pytest tests/test_features.py -v
```

Attendu : `4 passed`

- [ ] **Étape 5 : Commit (à faire toi-même)**

```bash
git add tests/test_features.py src/features.py
git commit -m "feat: ajoute les distances au métro et au centre-ville et les équipements"
```

---

### Tâche 4 : Définir les modèles

**Fichiers :**
- Créer : `tests/test_model.py`, `src/model.py`

**Contexte, pour bien comprendre :**
- **`Pipeline`** : enchaîne le prétraitement et le modèle dans un seul objet. `fit` et `predict` appliquent exactement les mêmes transformations, à l'entraînement comme dans l'app.
- **`ColumnTransformer`** : prétraite différemment les colonnes numériques (valeurs manquantes remplacées par la médiane, standardisées pour Ridge) et catégorielles (encodage one-hot ; les quartiers de moins de 20 annonces sont regroupés).
- **`TransformedTargetRegressor`** : le modèle apprend `log(prix)`, ce qui réduit l'effet des logements très chers, mais `predict()` renvoie directement des dollars grâce à `np.exp`.
- **Référence naïve** : un « modèle » qui prédit simplement le prix médian du quartier. Un vrai modèle doit faire nettement mieux.

- [ ] **Étape 1 : Écrire les tests qui échouent** — `tests/test_model.py`

```python
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_listings
from src.features import FEATURES, add_features
from src.model import NeighbourhoodMedianBaseline, build_models


def test_baseline_predicts_neighbourhood_median():
    X = pd.DataFrame({"neighbourhood_cleansed": ["A", "A", "A", "B"]})
    y = pd.Series([100.0, 200.0, 300.0, 50.0])
    baseline = NeighbourhoodMedianBaseline().fit(X, y)
    new = pd.DataFrame({"neighbourhood_cleansed": ["A", "B", "Inconnu"]})
    np.testing.assert_allclose(baseline.predict(new), [200.0, 50.0, 150.0])


@pytest.mark.parametrize("name", list(build_models()))
def test_models_train_and_predict_dollars(name, raw_listings, stations):
    df = add_features(clean_listings(raw_listings), stations)
    model = build_models()[name]
    model.fit(df[FEATURES], df["price"])
    predictions = model.predict(df[FEATURES])
    assert predictions.shape == (len(df),)
    assert np.all(predictions > 0)
    assert np.all(predictions < 2000)  # des dollars, pas des log(dollars)
```

`@pytest.mark.parametrize` lance le même test une fois par modèle : 4 tests d'un coup.

- [ ] **Étape 2 : Vérifier qu'ils échouent**

```bash
python -m pytest tests/test_model.py -v
```

Attendu : ÉCHEC avec `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Étape 3 : Écrire `src/model.py`**

```python
"""Définition des modèles comparés et de leur évaluation."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class NeighbourhoodMedianBaseline(RegressorMixin, BaseEstimator):
    """Référence naïve : prédit le prix médian du quartier (ou la médiane globale)."""

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.medians_ = y.groupby(X["neighbourhood_cleansed"]).median()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X):
        medians = X["neighbourhood_cleansed"].map(self.medians_)
        return medians.fillna(self.global_median_).to_numpy(dtype=float)


def build_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _log_price_model(preprocessor, regressor) -> TransformedTargetRegressor:
    """Le modèle apprend log(prix), mais predict() renvoie directement des dollars."""
    pipeline = Pipeline([("pre", preprocessor), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def build_models() -> dict:
    return {
        "Médiane du quartier": NeighbourhoodMedianBaseline(),
        "Ridge": _log_price_model(build_preprocessor(scale=True), Ridge(alpha=1.0)),
        "Random Forest": _log_price_model(
            build_preprocessor(scale=False),
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
        "Gradient Boosting": _log_price_model(
            build_preprocessor(scale=False),
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        ),
    }


# Hyperparamètres testés pour le meilleur modèle. Le préfixe regressor__model__ suit
# le chemin TransformedTargetRegressor -> Pipeline -> étape "model".
PARAM_DISTRIBUTIONS = {
    "Ridge": {"regressor__model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "Random Forest": {
        "regressor__model__max_features": [0.3, 0.5, 1.0],
        "regressor__model__min_samples_leaf": [2, 5, 10],
    },
    "Gradient Boosting": {
        "regressor__model__learning_rate": [0.03, 0.05, 0.1],
        "regressor__model__max_leaf_nodes": [15, 31, 63],
        "regressor__model__min_samples_leaf": [10, 20, 50],
        "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def cross_validate_models(models: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """MAE (en $) et R² moyens en validation croisée, du meilleur au moins bon."""
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=folds, scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}
        )
        rows.append(
            {"modèle": name, "MAE ($)": -scores["test_mae"].mean(), "R²": scores["test_r2"].mean()}
        )
    return pd.DataFrame(rows).sort_values("MAE ($)", ignore_index=True)
```

- [ ] **Étape 4 : Vérifier que les tests passent**

```bash
python -m pytest tests/test_model.py -v
```

Attendu : `5 passed`

- [ ] **Étape 5 : Lancer tous les tests**

```bash
python -m pytest
```

Attendu : `14 passed`

- [ ] **Étape 6 : Commit (à faire toi-même)**

```bash
git add tests/test_model.py src/model.py
git commit -m "feat: définit la référence naïve et les modèles Ridge, Random Forest, Gradient Boosting"
```

---

### Tâche 5 : Entraîner et sauvegarder le modèle final

**Fichiers :**
- Créer : `src/train.py`
- Généré : `models/model.joblib` (versionné)

**Déroulé :** chargement → séparation 80/20 → validation croisée des 4 modèles sur les 80 % → réglage des hyperparamètres du meilleur → **une seule** évaluation sur les 20 % de test → sauvegarde. Le fichier sauvegardé contient le modèle et les informations dont l'app a besoin : liste des quartiers, valeurs par défaut et erreur de test.

- [ ] **Étape 1 : Écrire `src/train.py`**

```python
"""Entraîne le modèle final et le sauvegarde dans models/model.joblib."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split

from src.clean import clean_listings
from src.config import LISTINGS_RAW, LISTINGS_URL, METRO_STATIONS, MODEL_PATH, RANDOM_STATE
from src.features import FEATURES, NUMERIC_FEATURES, add_features
from src.model import PARAM_DISTRIBUTIONS, build_models, cross_validate_models


def load_dataset() -> pd.DataFrame:
    """Annonces nettoyées + variables calculées, prêtes pour la modélisation."""
    if not LISTINGS_RAW.exists() or not METRO_STATIONS.exists():
        raise FileNotFoundError("Données absentes : lancez d'abord `python -m src.download`.")
    raw = pd.read_csv(LISTINGS_RAW)
    stations = pd.read_csv(METRO_STATIONS)
    return add_features(clean_listings(raw), stations)


def split(df: pd.DataFrame):
    """80 % entraînement / 20 % test. Le test n'est utilisé qu'à la toute fin."""
    return train_test_split(df[FEATURES], df["price"], test_size=0.2, random_state=RANDOM_STATE)


def tune(name: str, model, X_train, y_train):
    params = PARAM_DISTRIBUTIONS[name]
    search = RandomizedSearchCV(
        model,
        params,
        n_iter=min(20, len(ParameterGrid(params))),
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Meilleurs hyperparamètres : {search.best_params_}")
    return search.best_estimator_


def main() -> None:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    print(f"{len(X_train)} annonces d'entraînement, {len(X_test)} de test")

    scores = cross_validate_models(build_models(), X_train, y_train)
    print(scores.to_string(index=False))

    best_name = scores.iloc[0]["modèle"]
    print(f"Meilleur modèle en validation croisée : {best_name}")
    model = tune(best_name, build_models()[best_name], X_train, y_train)

    predictions = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)
    # Erreur relative médiane : la moitié des estimations sont plus proches que ça du vrai prix.
    test_median_rel_error = float(np.median(np.abs(y_test - predictions) / y_test))
    print(f"Jeu de test : MAE = {test_mae:.2f} $, R² = {test_r2:.3f}, erreur relative médiane = {test_median_rel_error:.0%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": best_name,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_median_rel_error": test_median_rel_error,
            "neighbourhoods": sorted(df["neighbourhood_cleansed"].unique()),
            "defaults": df[NUMERIC_FEATURES].median().to_dict(),
            "data_source": LISTINGS_URL,
        },
        MODEL_PATH,
        compress=3,
    )
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Étape 2 : Lancer l'entraînement**

```bash
python -m src.train
```

Attendu (environ 30 secondes ; les chiffres devraient être identiques, puisque l'instantané et `RANDOM_STATE` sont fixés) :

```
7332 annonces d'entraînement, 1833 de test
             modèle    MAE ($)       R²
      Random Forest  54.739152 0.642981
  Gradient Boosting  55.516062 0.640565
              Ridge  67.277646 0.482950
Médiane du quartier 106.552417 0.017076
Meilleur modèle en validation croisée : Random Forest
Meilleurs hyperparamètres : {'regressor__model__min_samples_leaf': 2, 'regressor__model__max_features': 0.5}
Jeu de test : MAE = 52.24 $, R² = 0.687, erreur relative médiane = 21%
Modèle sauvegardé dans ...\models\model.joblib
```

Si tes chiffres diffèrent un peu (autre version d'une bibliothèque), ce n'est pas grave : utilise **tes** chiffres dans le README.

- [ ] **Étape 3 : Vérifier la taille du modèle**

```bash
python -c "from src.config import MODEL_PATH; print(round(MODEL_PATH.stat().st_size / 1e6, 1), 'Mo')"
```

Attendu : environ `18.6 Mo`. GitHub accepte les fichiers jusqu'à 100 Mo.

- [ ] **Étape 4 : Commit (à faire toi-même)**

```bash
git add src/train.py models/model.joblib
git commit -m "feat: entraîne, évalue et sauvegarde le modèle final"
```

---

### Tâche 6 : Notebook d'exploration

**Fichiers :**
- Créer : `notebooks/01_exploration.ipynb`
- Générés : `figures/price_distribution.png`, `figures/price_by_neighbourhood.png`, `figures/price_vs_capacity_metro.png`, `figures/minimum_nights.png`, `figures/price_map.png`

**Comment créer le notebook :** lance `jupyter lab` (ou ouvre le dossier dans VS Code), crée `notebooks/01_exploration.ipynb` et choisis le noyau Python de l'environnement `airbnb-mtl`. Chaque bloc ci-dessous est **une cellule**. Les blocs « Markdown » sont des cellules de texte : reformule-les avec tes mots et ajoute tes observations après chaque graphique. C'est ce que les recruteurs lisent.

- [ ] **Étape 1 : Cellule Markdown**

```markdown
# Exploration des annonces Airbnb à Montréal

Objectif : comprendre les données avant de modéliser le prix par nuit.
Source : Inside Airbnb, instantané du 15 juin 2026 (licence CC BY 4.0).
```

- [ ] **Étape 2 : Cellule de configuration**

```python
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.clean import COLUMNS, parse_price
from src.config import LISTINGS_RAW, METRO_STATIONS, ROOT
from src.train import load_dataset

sns.set_theme(style="whitegrid")
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
```

- [ ] **Étape 3 : Données brutes**

```python
raw = pd.read_csv(LISTINGS_RAW)
print(raw.shape)
raw[COLUMNS].head()
```

Attendu : `(10656, 90)`

- [ ] **Étape 4 : Valeurs manquantes**

```python
raw[COLUMNS].isna().mean().sort_values(ascending=False).round(3)
```

Attendu : `bedrooms` et `beds` autour de 17-18 %, `price` à 11 %.

- [ ] **Étape 5 : Distribution brute des prix et des types**

```python
raw_prices = parse_price(raw["price"])
print(raw_prices.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(1))
raw["room_type"].value_counts()
```

Cellule Markdown à écrire juste après : explique pourquoi on garde 20 $ – 1 000 $ (1 % des prix sont sous 18 $, 1 % au-dessus de 1 233 $ : probablement des erreurs ou des cas très particuliers), et pourquoi on garde seulement les logements entiers et les chambres privées (les chambres d'hôtel et les chambres partagées représentent moins de 1 % des annonces).

- [ ] **Étape 6 : Prix et log du prix**

```python
df = load_dataset()
print(f"{len(df)} annonces gardées sur {len(raw)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["price"], bins=60, ax=axes[0])
axes[0].set(title="Prix par nuit", xlabel="Prix ($)")
sns.histplot(np.log(df["price"]), bins=60, ax=axes[1])
axes[1].set(title="Log du prix par nuit", xlabel="log(prix)")
fig.tight_layout()
fig.savefig(FIGURES / "price_distribution.png", dpi=120)
```

Attendu : `9165 annonces gardées sur 10656`. À noter en Markdown : la distribution du prix est très étirée vers la droite ; celle du log est bien plus symétrique. C'est pour ça que le modèle prédit le log du prix.

- [ ] **Étape 7 : Prix par quartier**

```python
top = df["neighbourhood_cleansed"].value_counts().head(15).index
by_hood = (
    df[df["neighbourhood_cleansed"].isin(top)]
    .groupby("neighbourhood_cleansed")["price"]
    .median()
    .sort_values()
)
fig, ax = plt.subplots(figsize=(8, 6))
by_hood.plot.barh(ax=ax)
ax.set(title="Prix médian par nuit — 15 quartiers avec le plus d'annonces", xlabel="Prix médian ($)", ylabel="")
fig.tight_layout()
fig.savefig(FIGURES / "price_by_neighbourhood.png", dpi=120)
```

- [ ] **Étape 8 : Capacité et distance au métro**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df[df["accommodates"] <= 8], x="accommodates", y="price", hue="room_type", showfliers=False, ax=axes[0])
axes[0].set(title="Prix selon la capacité", xlabel="Voyageurs", ylabel="Prix ($)")
metro_bins = pd.cut(df["dist_metro_km"], [0, 0.25, 0.5, 1, 2, 50], labels=["<250 m", "250-500 m", "0,5-1 km", "1-2 km", ">2 km"])
sns.boxplot(x=metro_bins, y=df["price"], showfliers=False, ax=axes[1])
axes[1].set(title="Prix selon la distance au métro", xlabel="Distance au métro", ylabel="Prix ($)")
fig.tight_layout()
fig.savefig(FIGURES / "price_vs_capacity_metro.png", dpi=120)
```

- [ ] **Étape 9 : Le nombre minimum de nuits, deux marchés**

```python
print(df["minimum_nights"].value_counts().head(6))
monthly = df["minimum_nights"] >= 31
print(df.groupby(monthly.rename("31 nuits et plus"))["price"].agg(["size", "median"]))

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df.assign(sejour=np.where(monthly, "31 nuits et plus", "Moins de 31 nuits")), x="price", hue="sejour", bins=60, log_scale=True, ax=ax)
ax.set(title="Prix selon la durée minimale de séjour", xlabel="Prix par nuit ($, échelle log)")
fig.tight_layout()
fig.savefig(FIGURES / "minimum_nights.png", dpi=120)
```

Attendu : environ la moitié des annonces imposent 31 nuits ou plus, avec un prix médian autour de 90 $, contre environ 245 $ pour les autres. À noter en Markdown : ce sont presque deux marchés, la location au mois et la location touristique. Une hypothèse à vérifier et à citer avec une source si tu la gardes : la réglementation québécoise sur l'hébergement touristique de courte durée pousse beaucoup d'hôtes vers des séjours de 31 nuits et plus.

- [ ] **Étape 10 : Carte des prix (image pour le README)**

```python
stations = pd.read_csv(METRO_STATIONS)
fig, ax = plt.subplots(figsize=(8, 8))
points = ax.scatter(df["longitude"], df["latitude"], c=np.log(df["price"]), cmap="viridis", s=4, alpha=0.6)
ax.scatter(stations["lon"], stations["lat"], marker="^", color="red", s=25, label="Stations de métro")
fig.colorbar(points, ax=ax, label="log(prix)")
ax.set(title="Annonces Airbnb à Montréal, colorées par prix", xlabel="Longitude", ylabel="Latitude")
ax.legend()
ax.set_aspect(1 / np.cos(np.radians(45.5)))
fig.tight_layout()
fig.savefig(FIGURES / "price_map.png", dpi=120)
```

- [ ] **Étape 11 : Carte interactive**

```python
sample = df.sample(1500, random_state=42)
carte = folium.Map(location=[45.51, -73.58], zoom_start=12, tiles="cartodbpositron")
for _, row in sample.iterrows():
    folium.CircleMarker(
        [row["latitude"], row["longitude"]],
        radius=3,
        color="crimson" if row["price"] > df["price"].median() else "steelblue",
        fill=True,
        popup=f"{row['price']:.0f} $ — {row['neighbourhood_cleansed']}",
    ).add_to(carte)
carte
```

Rouge : au-dessus du prix médian ; bleu : en dessous. Clique sur un point pour voir son prix. GitHub n'affiche pas ce type de carte, d'où l'image de l'étape 10.

- [ ] **Étape 12 : Corrélations**

```python
numeric = ["price", "accommodates", "bedrooms", "beds", "bathrooms", "dist_metro_km", "dist_downtown_km", "n_amenities", "minimum_nights"]
df[numeric].corr(method="spearman")["price"].drop("price").sort_values()
```

Attendu : `minimum_nights` est la plus négative (≈ -0,65) et `accommodates` la plus positive (≈ 0,58). La corrélation de Spearman compare les rangs plutôt que les valeurs : elle est moins sensible aux valeurs extrêmes.

- [ ] **Étape 13 : Conclusion en Markdown**

Écris 3 à 5 phrases sur ce que tu retiens : quelles variables semblent compter, et ce qui t'a surpris.

- [ ] **Étape 14 : Relancer tout le notebook et vérifier**

Menu *Run → Restart Kernel and Run All Cells*. Aucune cellule ne doit être en erreur, et `figures/` doit contenir 5 images.

- [ ] **Étape 15 : Commit (à faire toi-même)**

```bash
git add notebooks/01_exploration.ipynb figures/
git commit -m "docs: ajoute le notebook d'exploration et ses graphiques"
```

---

### Tâche 7 : Notebook de modélisation

**Fichiers :**
- Créer : `notebooks/02_modelisation.ipynb`
- Générés : `figures/model_comparison.png`, `figures/feature_importance.png`, `figures/predicted_vs_actual.png`

Prérequis : `models/model.joblib` existe (Tâche 5).

- [ ] **Étape 1 : Cellule Markdown**

```markdown
# Modélisation du prix par nuit

On compare une référence naïve à trois modèles en validation croisée (5 plis), puis on évalue
le modèle final une seule fois sur le jeu de test (20 %) et on interprète ses prédictions.
```

- [ ] **Étape 2 : Configuration et données**

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import MODEL_PATH, RANDOM_STATE, ROOT
from src.model import NeighbourhoodMedianBaseline, build_models, cross_validate_models
from src.train import load_dataset, split

sns.set_theme(style="whitegrid")
FIGURES = ROOT / "figures"

df = load_dataset()
X_train, X_test, y_train, y_test = split(df)
len(X_train), len(X_test)
```

Attendu : `(7332, 1833)`. Grâce au même `random_state`, c'est exactement la séparation utilisée par `train.py`.

- [ ] **Étape 3 : Comparaison des modèles**

```python
scores = cross_validate_models(build_models(), X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.barplot(data=scores, x="MAE ($)", y="modèle", ax=ax)
ax.set(title="Erreur moyenne en validation croisée (plus bas = mieux)", ylabel="")
fig.tight_layout()
fig.savefig(FIGURES / "model_comparison.png", dpi=120)
scores.round(3)
```

À noter en Markdown : les deux modèles à base d'arbres (Random Forest, Gradient Boosting) sont au coude à coude et divisent l'erreur de la référence naïve par environ deux. Ridge, linéaire, ne capte pas aussi bien les interactions.

- [ ] **Étape 4 : Évaluation finale sur le jeu de test**

```python
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
baseline = NeighbourhoodMedianBaseline().fit(X_train, y_train)

pd.DataFrame(
    {
        "modèle": ["Médiane du quartier", artifact["model_name"]],
        "MAE test ($)": [mean_absolute_error(y_test, baseline.predict(X_test)), mean_absolute_error(y_test, model.predict(X_test))],
        "R² test": [r2_score(y_test, baseline.predict(X_test)), r2_score(y_test, model.predict(X_test))],
    }
).round(3)
```

Attendu : environ 104 $ de MAE pour la référence et environ 52 $ pour le Random Forest (R² ≈ 0,69).

- [ ] **Étape 5 : Importance des variables par permutation**

Le principe : on mélange au hasard une colonne du jeu de test et on mesure de combien l'erreur augmente. Plus elle augmente, plus le modèle dépend de cette variable.

```python
importance = permutation_importance(
    model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
)
importances = pd.Series(importance.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
importances.plot.barh(ax=ax)
ax.set(title="Importance par permutation", xlabel="Hausse de l'erreur moyenne ($) quand on mélange la variable")
fig.tight_layout()
fig.savefig(FIGURES / "feature_importance.png", dpi=120)
importances.sort_values(ascending=False).round(2)
```

Attendu : `minimum_nights` loin devant (≈ +43 $), puis `accommodates`, `bathrooms` et `dist_downtown_km`. À noter : le quartier (`neighbourhood_cleansed`) compte peu, parce que la latitude, la longitude et la distance au centre portent déjà cette information.

- [ ] **Étape 6 : Prix prédit contre prix réel**

```python
predictions = model.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, predictions, s=5, alpha=0.4)
ax.plot([20, 1000], [20, 1000], color="red", label="Prédiction parfaite")
ax.set(xscale="log", yscale="log", xlabel="Prix réel ($)", ylabel="Prix prédit ($)", title="Prix prédit vs prix réel (jeu de test)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "predicted_vs_actual.png", dpi=120)
```

- [ ] **Étape 7 : Analyse des erreurs**

```python
errors = X_test.assign(price=y_test, predicted=predictions, abs_error=np.abs(y_test - predictions))
by_hood = (
    errors.groupby("neighbourhood_cleansed")
    .agg(annonces=("price", "size"), prix_median=("price", "median"), mae=("abs_error", "mean"))
    .query("annonces >= 20")
    .sort_values("mae", ascending=False)
)
by_hood.round(1)
```

```python
print(errors.groupby("room_type")["abs_error"].mean().round(1))
errors["price_bin"] = pd.cut(errors["price"], [20, 100, 200, 400, 1000])
errors.groupby("price_bin", observed=True)["abs_error"].agg(["size", "mean"]).round(1)
```

Attendu : l'erreur est la plus forte dans les quartiers chers (Plateau ≈ 66 $) et pour les logements à plus de 400 $ (≈ 176 $). Le modèle a tendance à « tirer vers la moyenne » les logements haut de gamme, pour lesquels il a peu d'exemples.

- [ ] **Étape 8 : Conclusion et limites en Markdown**

Résume les résultats, puis liste les limites : un seul instantané (pas de saisonnalité) ; le prix affiché n'est pas forcément le prix payé ; les descriptions et photos ne sont pas utilisées ; peu d'exemples de logements haut de gamme.

- [ ] **Étape 9 : Relancer tout le notebook**

*Restart Kernel and Run All Cells* : aucune erreur, 3 nouvelles images dans `figures/`.

- [ ] **Étape 10 : Commit (à faire toi-même)**

```bash
git add notebooks/02_modelisation.ipynb figures/
git commit -m "docs: ajoute le notebook de modélisation, l'importance des variables et l'analyse des erreurs"
```

---

### Tâche 8 : App Streamlit

**Fichiers :**
- Créer : `tests/test_app.py`, `streamlit_app.py`

**Contexte :** l'utilisateur de l'app ne connaît pas les coordonnées GPS de son logement. On lui demande donc la station de métro la plus proche et le temps de marche, et on utilise les coordonnées de la station, ce qui est une approximation. La durée minimale de séjour est proposée sous forme de deux choix (1 ou 31 nuits), puisque ce sont les deux cas les plus courants.

- [ ] **Étape 1 : Écrire le test qui échoue** — `tests/test_app.py`

`AppTest` exécute l'app sans navigateur et permet d'inspecter ce qu'elle affiche.

```python
import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PATH, ROOT


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="lancez d'abord python -m src.train")
def test_app_shows_a_price():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value.endswith("$")
```

- [ ] **Étape 2 : Vérifier qu'il échoue**

```bash
python -m pytest tests/test_app.py -v
```

Attendu : ÉCHEC avec `FileNotFoundError: AppTest script not found`

- [ ] **Étape 3 : Écrire `streamlit_app.py`** (à la racine du projet)

```python
"""Application de démo : estimer le prix d'une nuit Airbnb à Montréal."""
import joblib
import pandas as pd
import streamlit as st

from src.config import DOWNTOWN_LAT, DOWNTOWN_LON, METRO_STATIONS, MODEL_PATH
from src.features import FEATURES, haversine_km

WALKING_KM_PER_MINUTE = 0.08  # environ 5 km/h
ROOM_TYPES = {"Logement entier": "Entire home/apt", "Chambre privée": "Private room"}
STAY_TYPES = {"Courts séjours (dès 1 nuit)": 1, "Location au mois (31 nuits et plus)": 31}


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_stations() -> pd.DataFrame:
    return pd.read_csv(METRO_STATIONS)


st.set_page_config(page_title="Prix Airbnb Montréal", page_icon="🏠")
st.title("🏠 Combien vaut une nuit à Montréal ?")
st.write("Estimation du prix par nuit d'un logement Airbnb, à partir de ses caractéristiques.")

if not MODEL_PATH.exists():
    st.error("Modèle introuvable : lancez d'abord `python -m src.train`.")
    st.stop()

artifact = load_artifact()
stations = load_stations()
defaults = artifact["defaults"]

with st.sidebar:
    st.header("Le logement")
    neighbourhood = st.selectbox("Quartier", artifact["neighbourhoods"])
    room_label = st.radio("Type", list(ROOM_TYPES))
    accommodates = st.slider("Voyageurs", 1, 16, 2)
    bedrooms = st.slider("Chambres", 0, 10, 1)
    beds = st.slider("Lits", 1, 16, 1)
    bathrooms = st.select_slider("Salles de bain", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], value=1.0)
    bath_shared = st.checkbox("Salle de bain partagée")
    stay_label = st.radio("Durée minimale de séjour", list(STAY_TYPES))

    st.header("Emplacement")
    station_name = st.selectbox("Station de métro la plus proche", stations["name"])
    walking_minutes = st.slider("Minutes à pied du métro", 0, 30, 5)

    st.header("Équipements")
    has_wifi = st.checkbox("Wi-Fi", value=True)
    has_ac = st.checkbox("Climatisation", value=True)
    has_parking = st.checkbox("Stationnement")
    has_kitchen = st.checkbox("Cuisine", value=True)
    n_amenities = st.slider("Nombre total d'équipements", 0, 100, int(defaults["n_amenities"]))

station = stations.set_index("name").loc[station_name]
listing = pd.DataFrame(
    [
        {
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "bathrooms": bathrooms,
            "minimum_nights": STAY_TYPES[stay_label],
            "latitude": station["lat"],
            "longitude": station["lon"],
            "dist_metro_km": walking_minutes * WALKING_KM_PER_MINUTE,
            "dist_downtown_km": haversine_km(station["lat"], station["lon"], DOWNTOWN_LAT, DOWNTOWN_LON),
            "n_amenities": n_amenities,
            "bath_shared": int(bath_shared),
            "has_wifi": int(has_wifi),
            "has_ac": int(has_ac),
            "has_parking": int(has_parking),
            "has_kitchen": int(has_kitchen),
            "neighbourhood_cleansed": neighbourhood,
            "room_type": ROOM_TYPES[room_label],
        }
    ]
)[FEATURES]

price = float(artifact["model"].predict(listing)[0])
rel_error = artifact["test_median_rel_error"]

st.metric("Prix estimé par nuit", f"{price:.0f} $")
st.caption(
    f"Fourchette indicative : {price * (1 - rel_error):.0f} $ – {price * (1 + rel_error):.0f} $. "
    f"Sur le jeu de test, la moitié des estimations sont à moins de {rel_error:.0%} du vrai prix."
)
st.map(pd.DataFrame({"lat": [station["lat"]], "lon": [station["lon"]]}), zoom=13)
st.caption(
    f"Modèle : {artifact['model_name']} · Données : Inside Airbnb (CC BY 4.0) et STM."
)
```

- [ ] **Étape 4 : Vérifier que le test passe**

```bash
python -m pytest tests/test_app.py -v
```

Attendu : `1 passed`

- [ ] **Étape 5 : Essayer l'app dans le navigateur**

```bash
streamlit run streamlit_app.py
```

Une page s'ouvre sur `http://localhost:8501`. Vérifications à faire à la main :
- Avec les réglages par défaut (logement entier, 2 voyageurs, courts séjours), le prix est autour de 190 $.
- En passant à « Location au mois », le prix baisse nettement (environ 90 $).
- En choisissant « Chambre privée » et « Location au mois », il tombe autour de 50 $, avec une fourchette d'environ 41 $ à 62 $.

Arrête l'app avec `Ctrl+C` dans le terminal.

- [ ] **Étape 6 : Prendre une capture d'écran pour le README**

Enregistre une capture de l'app dans `figures/app_screenshot.png`.

- [ ] **Étape 7 : Lancer tous les tests**

```bash
python -m pytest
```

Attendu : `15 passed`

- [ ] **Étape 8 : Commit (à faire toi-même)**

```bash
git add tests/test_app.py streamlit_app.py figures/app_screenshot.png
git commit -m "feat: ajoute l'app Streamlit d'estimation du prix"
```

---

### Tâche 9 : README

**Fichiers :**
- Créer : `README.md`

- [ ] **Étape 1 : Écrire `README.md`**

Les chiffres ci-dessous sont ceux du prototype : remplace-les par les tiens s'ils diffèrent (Tâche 5, étape 2, et Tâche 7). Le lien de l'app sera ajouté à la Tâche 10 : d'ici là, garde la ligne `**App en ligne :**` telle quelle.

````markdown
# 🏠 Prix des Airbnb à Montréal

Prédire le prix par nuit d'un logement Airbnb à Montréal à partir de ses caractéristiques
et de son emplacement : données publiques, nettoyage, exploration, modèles scikit-learn
et app Streamlit.

**App en ligne :** lien ajouté après le déploiement

![Capture de l'app](figures/app_screenshot.png)

## La question

Combien peut coûter une nuit dans un Airbnb à Montréal, selon sa taille, ses équipements
et sa distance au métro et au centre-ville ?

## Les données

| Source | Contenu | Licence |
|--------|---------|---------|
| [Inside Airbnb](https://insideairbnb.com/get-the-data/) | 10 656 annonces de Montréal, instantané du 15 juin 2026 | CC BY 4.0 |
| [STM — données GTFS](https://www.stm.info/fr/a-propos/developpeurs) | Position des 68 stations de métro | Licence ouverte STM |

Après nettoyage (prix manquants, prix hors de 20 $ – 1 000 $, chambres d'hôtel et partagées) :
**9 165 annonces**.

## Ce que montrent les données

![Prix selon la durée minimale de séjour](figures/minimum_nights.png)

- **Deux marchés :** environ la moitié des annonces imposent 31 nuits minimum. Leur prix médian
  est de 90 $ la nuit, contre 245 $ pour les courts séjours.
- La capacité (nombre de voyageurs) est la caractéristique du logement la plus liée au prix.
- Les prix baissent en s'éloignant du centre-ville.

![Carte des prix](figures/price_map.png)

## Modélisation

- Cible : log du prix (distribution très étirée), reconverti en dollars pour l'évaluation.
- 80 % entraînement / 20 % test ; comparaison en validation croisée à 5 plis.
- Tout le prétraitement est dans un `Pipeline` scikit-learn, pour éviter les fuites de données.
- Les colonnes calculées à partir du prix (revenus estimés, etc.) sont exclues.

| Modèle | MAE validation croisée | R² |
|--------|-----------------------:|---:|
| Médiane du quartier (référence) | 106,6 $ | 0,02 |
| Ridge | 67,3 $ | 0,48 |
| Gradient Boosting | 55,5 $ | 0,64 |
| **Random Forest** | **54,7 $** | **0,64** |

**Sur le jeu de test**, le Random Forest réglé obtient une **MAE de 52 $** (R² = 0,69) contre
104 $ pour la référence naïve. La moitié de ses estimations sont à moins de 21 % du vrai prix.

![Importance des variables](figures/feature_importance.png)

La durée minimale de séjour est de loin la variable la plus importante, suivie de la capacité,
du nombre de salles de bain et de la distance au centre-ville.

## Limites et pistes d'amélioration

- Un seul instantané : pas de saisonnalité (Grand Prix, festivals, hiver).
- Le modèle se trompe davantage sur les logements haut de gamme (plus de 400 $), peu représentés.
- Pistes : utiliser les descriptions (NLP), plusieurs instantanés dans l'année, des données
  de quartier (commerces, parcs), et prédire séparément les deux marchés.

## Reproduire le projet

Prérequis : Python 3.12 (ou un environnement conda : `conda create -n airbnb-mtl python=3.12`).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux : source .venv/bin/activate)
pip install -r requirements-dev.txt

python -m src.download            # télécharge les données
python -m src.train               # entraîne et sauvegarde le modèle
streamlit run streamlit_app.py    # lance l'app
```

Tests : `python -m pytest`

## Structure

```
src/            code réutilisable (téléchargement, nettoyage, variables, modèles, entraînement)
notebooks/      exploration et modélisation commentées
tests/          tests pytest
figures/        graphiques utilisés dans ce README
models/         modèle entraîné
streamlit_app.py
```
````

- [ ] **Étape 2 : Vérifier les liens des images**

```bash
python -c "import re, pathlib; missing = [p for p in re.findall(r'\((figures/[^)]+)\)', pathlib.Path('README.md').read_text(encoding='utf-8')) if not pathlib.Path(p).exists()]; print(missing or 'OK')"
```

Attendu : `OK`

- [ ] **Étape 3 : Commit et envoi (à faire toi-même)**

```bash
git add README.md
git commit -m "docs: rédige le README du projet"
git push
```

Ouvre ensuite ton repo sur GitHub pour vérifier que le README et les images s'affichent bien.

---

### Tâche 10 : Déployer l'app sur Streamlit Community Cloud

**Fichiers :**
- Modifier : `README.md` (ligne `**App en ligne :**`)

Ces étapes se font dans ton navigateur, avec ton compte : c'est à toi de les faire.

- [ ] **Étape 1 : Vérifier que tout est sur GitHub**

```bash
git status
```

Attendu : `nothing to commit, working tree clean` et `Your branch is up to date with 'origin/main'`.

- [ ] **Étape 2 : Créer l'app**

1. Va sur [share.streamlit.io](https://share.streamlit.io) et connecte-toi avec ton compte GitHub.
2. Clique sur **Create app**, puis choisis de déployer depuis un repo GitHub existant.
3. Repository : `STcyber-gif/Montreal-airbnb-price-prediction` ; Branch : `main` ; Main file path : `streamlit_app.py`.
4. Choisis une URL courte, par exemple `airbnb-montreal`.
5. Dans **Advanced settings**, choisis Python **3.12** (la même version que sur ton PC).
6. Clique sur **Deploy**. La première installation prend quelques minutes.

- [ ] **Étape 3 : Tester l'app en ligne**

Refais les vérifications de la Tâche 8, étape 5. Si l'app affiche une erreur, ouvre **Manage app** (en bas à droite) pour lire les journaux.

- [ ] **Étape 4 : Ajouter le lien dans le README**

Remplace la ligne :

```markdown
**App en ligne :** lien ajouté après le déploiement
```

par (avec ton URL réelle) :

```markdown
**App en ligne :** [airbnb-montreal.streamlit.app](https://airbnb-montreal.streamlit.app)
```

- [ ] **Étape 5 : Commit et envoi (à faire toi-même)**

```bash
git add README.md
git commit -m "docs: ajoute le lien vers l'app déployée"
git push
```

- [ ] **Étape 6 : Finitions sur GitHub**

Sur la page du repo, clique sur la roue dentée à côté de **About** : ajoute une description d'une ligne, le lien de l'app dans *Website* et des *topics* (`machine-learning`, `data-science`, `python`, `scikit-learn`, `streamlit`, `montreal`, `airbnb`). Pense aussi à épingler le repo sur ton profil (*Customize your pins*).

---

## Récapitulatif

| Tâche | Résultat | Tests |
|-------|----------|-------|
| 0 | Python, environnement, configuration | — |
| 1 | Données téléchargées, 68 stations | 1 |
| 2 | Nettoyage | 4 |
| 3 | Variables (distances, équipements) | 4 |
| 4 | Modèles et référence naïve | 5 |
| 5 | Modèle entraîné et sauvegardé | — |
| 6 | Notebook d'exploration + 5 graphiques | — |
| 7 | Notebook de modélisation + 3 graphiques | — |
| 8 | App Streamlit | 1 |
| 9 | README | — |
| 10 | App en ligne | — |
