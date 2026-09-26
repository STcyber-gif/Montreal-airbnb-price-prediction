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
