"""Télécharge les données brutes : annonces Airbnb et stations de métro."""
import io
import urllib.request
import zipfile

import pandas as pd

from src.config import GTFS_URL, LISTINGS_RAW, LISTINGS_URL, METRO_STATIONS


def download_listings() -> None:
    LISTINGS_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {LISTINGS_URL} ...")
    urllib.request.urlretrieve(LISTINGS_URL, LISTINGS_RAW)
    print(f"-> {LISTINGS_RAW}")


def extract_metro_stations(stops: pd.DataFrame) -> pd.DataFrame:
    """Garde une ligne par station de métro à partir du fichier stops.txt du GTFS."""
    parent = stops["parent_station"].fillna("")
    is_metro_platform = (stops["location_type"] == "0") & parent.str.startswith("STATION_M")
    stations = stops[is_metro_platform].drop_duplicates("parent_station")
    names = (
        stations["stop_name"]
        .str.removeprefix("Station ")
        .str.replace(r"\s*-Zone B$", "", regex=True)
    )
    return pd.DataFrame(
        {
            "name": names.to_numpy(),
            "lat": stations["stop_lat"].astype(float).to_numpy(),
            "lon": stations["stop_lon"].astype(float).to_numpy(),
        }
    ).sort_values("name", ignore_index=True)


def download_metro_stations() -> None:
    METRO_STATIONS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {GTFS_URL} (~45 Mo) ...")
    with urllib.request.urlopen(GTFS_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("stops.txt") as f:
        stops = pd.read_csv(f, dtype=str)
    stations = extract_metro_stations(stops)
    stations.to_csv(METRO_STATIONS, index=False)
    print(f"-> {METRO_STATIONS} ({len(stations)} stations)")


if __name__ == "__main__":
    download_listings()
    download_metro_stations()
