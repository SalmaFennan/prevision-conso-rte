import pytest

from prevision_conso.metriques import mae, mape, rmse


def test_mae_cas_simple():
    assert mae([100, 200], [110, 170]) == 20      # erreurs de 10 et 30 -> moyenne 20


def test_rmse_penalise_les_grosses_erreurs():
    # erreurs de 3 et 4 : RMSE = racine((9 + 16) / 2) ≈ 3,54, supérieure à la MAE (3,5)
    assert rmse([10, 10], [13, 6]) == pytest.approx(12.5 ** 0.5)
    assert rmse([10, 10], [13, 6]) > mae([10, 10], [13, 6])


def test_mape_en_pourcentage():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)   # 10 % d'erreur sur chaque point