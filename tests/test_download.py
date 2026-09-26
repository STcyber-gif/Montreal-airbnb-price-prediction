import pandas as pd

from src.download import extract_metro_stations


def test_extract_metro_stations_keeps_one_row_per_station():
    # Extrait simplifié du fichier stops.txt de la STM (tout est lu en texte).
    stops = pd.DataFrame(
        {
            "stop_id": ["STATION_M118", "43", "43-01", "STATION_M700", "70", "51234"],
            "stop_name": [
                "STATION ANGRIGNON",
                "Station Angrignon",
                "Station Angrignon",
                "STATION LONGUEUIL",
                "Station Longueuil-Université de Sherbrooke -Zone B",
                "Arrêt de bus Sherbrooke / Saint-Denis",
            ],
            "stop_lat": ["45.446397", "45.446466", "45.446319", "45.525", "45.524", "45.518"],
            "stop_lon": ["-73.603293", "-73.603118", "-73.603835", "-73.522", "-73.521", "-73.568"],
            "location_type": ["1", "0", "2", "1", "0", "0"],
            "parent_station": [None, "STATION_M118", "STATION_M118", None, "STATION_M700", None],
        }
    )
    stations = extract_metro_stations(stops)
    assert stations["name"].tolist() == ["Angrignon", "Longueuil-Université de Sherbrooke"]
    assert stations.loc[0, "lat"] == 45.446466
    assert list(stations.columns) == ["name", "lat", "lon"]
