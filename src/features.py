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