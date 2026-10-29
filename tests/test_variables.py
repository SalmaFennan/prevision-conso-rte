import numpy as np
import pandas as pd

from prevision_conso.variables import construire_variables


def _serie_test(nb_jours: int = 10) -> pd.Series:
    """Série fabriquée 0, 1, 2, ... : la valeur d'un point = sa position."""
    index = pd.date_range("2025-01-01", periods=nb_jours * 48, freq="30min", tz="UTC")
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