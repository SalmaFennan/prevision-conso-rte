from prevision_conso.metriques import mae


def test_mae_cas_simple():
    assert mae([100, 200], [110, 170]) == 20   # erreurs de 10 et 30 -> moyenne 20