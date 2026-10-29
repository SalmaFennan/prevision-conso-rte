"""Préparation de la série de consommation : pas de 30 min, grille complète, sans doublon."""
from pathlib import Path

import pandas as pd


def charger_serie(chemin_parquet: Path) -> pd.Series:
    """Charge le Parquet et renvoie une série demi-horaire propre, indexée en UTC.

    La source enregistre 48 points par jour en heure locale, quel que soit le changement d'heure :
    - fin mars, l'heure inexistante crée des doublons (fusionnés par la moyenne) ;
    - fin octobre, l'heure répétée manque (2 points interpolés).
    """
    df = pd.read_parquet(chemin_parquet)

    # Pas réel des données définitives : minutes 0 et 30 uniquement
    df = df[df["date_heure"].dt.minute.isin([0, 30])]
    serie = df.set_index("date_heure")["consommation"]

    # Doublons fusionnés, grille complète, trous courts interpolés
    serie = serie.groupby(level=0).mean()
    grille = pd.date_range(serie.index.min(), serie.index.max(), freq="30min")
    serie = serie.reindex(grille).interpolate(method="time", limit=2)

    serie.index.name = "date_heure"
    serie.name = "consommation"
    return serie