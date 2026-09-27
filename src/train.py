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