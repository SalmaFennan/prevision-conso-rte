"""Baselines naïves : même heure J-7 et même heure J-1."""
import pandas as pd

from prevision_conso.acquisition import CHEMIN_PARQUET
from prevision_conso.metriques import mae, mape, rmse
from prevision_conso.preparation import charger_serie

DEBUT_TEST = pd.Timestamp("2025-09-01", tz="Europe/Paris")
PAS_PAR_JOUR = 48  # 24 h x 2 pas de 30 min


def prevision_naive(serie: pd.Series, nb_jours: int) -> pd.Series:
    """Prévoit la valeur observée nb_jours plus tôt, à la même heure.

    Le décalage en nombre de lignes n'est valable que parce que la grille est complète.
    """
    return serie.shift(nb_jours * PAS_PAR_JOUR)


def main() -> None:
    serie = charger_serie(CHEMIN_PARQUET)
    test = serie[serie.index >= DEBUT_TEST]
    print(f"Période de test : {test.index.min()} -> {test.index.max()} ({len(test)} points)\n")

    for nom, nb_jours in [("Naïve J-7", 7), ("Naïve J-1", 1)]:
        prevu = prevision_naive(serie, nb_jours).loc[test.index]
        print(f"{nom:10s} | MAE {mae(test, prevu):8.0f} MW | "
              f"RMSE {rmse(test, prevu):8.0f} MW | MAPE {mape(test, prevu):5.2f} %")


if __name__ == "__main__":
    main()