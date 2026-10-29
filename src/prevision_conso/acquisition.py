"""Acquisition des données éCO2mix nationales (RTE, via la plateforme ODRÉ)."""
from pathlib import Path

import pandas as pd
import requests

URL_EXPORT = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "eco2mix-national-cons-def/exports/csv"
)

# Racine du projet : acquisition.py -> prevision_conso -> src -> racine
RACINE = Path(__file__).resolve().parents[2]
CHEMIN_CSV = RACINE / "data" / "raw" / "eco2mix_national.csv"
CHEMIN_PARQUET = RACINE / "data" / "processed" / "eco2mix_national.parquet"


def telecharger_eco2mix(date_debut: str, chemin_csv: Path) -> None:
    """Télécharge la consommation nationale depuis date_debut et l'enregistre telle quelle (CSV brut)."""
    params = {
        "select": "date_heure,consommation",            # seulement les 2 colonnes utiles
        "where": f"date_heure >= date'{date_debut}'",   # filtre côté serveur
        "timezone": "UTC",                              # pas de changement d'heure
    }
    chemin_csv.parent.mkdir(parents=True, exist_ok=True)

    # stream=True : écriture morceau par morceau, sans tout charger en mémoire
    # timeout=(10, 300) : 10 s pour se connecter, 300 s maximum sans recevoir de données
    with requests.get(URL_EXPORT, params=params, stream=True, timeout=(10, 300)) as reponse:
        reponse.raise_for_status()  # erreur si le code HTTP n'est pas 2xx
        with open(chemin_csv, "wb") as fichier:
            for bloc in reponse.iter_content(chunk_size=1024 * 1024):
                fichier.write(bloc)

    taille_mo = chemin_csv.stat().st_size / 1e6
    print(f"CSV brut enregistré : {chemin_csv} ({taille_mo:.1f} Mo)")


def csv_vers_parquet(chemin_csv: Path, chemin_parquet: Path) -> pd.DataFrame:
    """Lit le CSV brut, type les colonnes, trie par date et enregistre un fichier Parquet."""
    # 1. Lecture : les exports ODRÉ utilisent le point-virgule comme séparateur
    df = pd.read_csv(chemin_csv, sep=";")

    # 2. Conversion du texte en vraie date, en précisant que c'est de l'UTC
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)

    # 3. Tri chronologique, puis renumérotation des lignes de 0 à n-1
    df = df.sort_values("date_heure").reset_index(drop=True)

    chemin_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chemin_parquet, index=False)
    print(f"Parquet enregistré : {chemin_parquet} ({len(df)} lignes)")
    return df


if __name__ == "__main__":
    if CHEMIN_CSV.exists():
        print(f"CSV déjà présent, pas de nouveau téléchargement : {CHEMIN_CSV}")
    else:
        telecharger_eco2mix("2019-01-01", CHEMIN_CSV)

    df = csv_vers_parquet(CHEMIN_CSV, CHEMIN_PARQUET)
    print(df.shape)
    print(df.head())
    print(df.dtypes)