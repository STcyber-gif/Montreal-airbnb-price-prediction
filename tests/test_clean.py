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
