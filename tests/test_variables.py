import numpy as np
import pandas as pd

from prevision_conso.variables import construire_variables


def _serie_test(debut: str = "2025-01-01", nb_jours: int = 10) -> pd.Series:
    """Série fabriquée 0, 1, 2, ... : la valeur d'un point = sa position."""
    index = pd.date_range(debut, periods=nb_jours * 48, freq="30min", tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="consommation")


def test_decalages_sans_fuite():
    serie = _serie_test()
    X, y = construire_variables(serie)

    # Le décalage de 24 h doit valoir la consommation observée 48 pas plus tôt
    assert (X["lag_24h"] == y - 48).all()
    assert (X["lag_168h"] == y - 336).all()
    # Aucune valeur manquante, et seule la première semaine est retirée
    assert X.isna().sum().sum() == 0
    assert len(X) == len(serie) - 336


def test_variables_calendaires_en_heure_de_paris():
    X, _ = construire_variables(_serie_test(debut="2024-12-20", nb_jours=20))

    # 1er janvier 2025 à 12:00 UTC = 13:00 à Paris (heure d'hiver, UTC+1), jour férié
    nouvel_an = X.loc[pd.Timestamp("2025-01-01 12:00", tz="UTC")]
    assert nouvel_an["heure"] == 13.0
    assert nouvel_an["est_ferie"] == 1


    # Samedi 4 janvier 2025 : week-end
    samedi = X.loc[pd.Timestamp("2025-01-04 12:00", tz="UTC")]
    assert samedi["est_week_end"] == 1