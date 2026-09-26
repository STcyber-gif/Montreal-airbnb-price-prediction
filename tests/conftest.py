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
