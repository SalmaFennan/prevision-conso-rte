import pandas as pd

from prevision_conso.preparation import charger_serie


def test_charger_serie_corrige_doublons_et_trous(tmp_path):
    # Petites données fabriquées : 1 doublon (00:30), 1 point à 15 min, 1 trou (01:00)
    dates = pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:30",
         "2025-01-01 00:30", "2025-01-01 01:30"],
        utc=True,
    )
    conso = [100.0, None, 200.0, 220.0, 300.0]
    chemin = tmp_path / "test.parquet"
    pd.DataFrame({"date_heure": dates, "consommation": conso}).to_parquet(chemin, index=False)

    serie = charger_serie(chemin)

    assert len(serie) == 4                       # 00:00, 00:30, 01:00, 01:30
    assert not serie.index.duplicated().any()
    assert serie.isna().sum() == 0
    assert serie.loc[pd.Timestamp("2025-01-01 00:30", tz="UTC")] == 210.0   # moyenne du doublon
    assert serie.loc[pd.Timestamp("2025-01-01 01:00", tz="UTC")] == 255.0   # interpolation