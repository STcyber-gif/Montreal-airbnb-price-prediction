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