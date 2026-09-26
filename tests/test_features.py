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