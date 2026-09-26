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