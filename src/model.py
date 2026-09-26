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