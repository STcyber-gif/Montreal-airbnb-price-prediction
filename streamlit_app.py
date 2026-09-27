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
st.title("Combien vaut une nuit à Montréal ?")
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