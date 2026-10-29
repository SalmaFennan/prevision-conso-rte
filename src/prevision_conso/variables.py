"""Construction des variables explicatives (features) pour la prévision."""
import holidays
import numpy as np
import pandas as pd

PAS_PAR_JOUR = 48          # 24 h x 2 pas de 30 min
FUSEAU = "Europe/Paris"


def construire_variables(serie: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Renvoie (X, y) à partir de la série demi-horaire propre (index UTC).

    - Variables calendaires calculées en heure de Paris (rythme de vie local).
    - Décalages calculés sur la grille UTC complète (48 pas = 24 h réelles).
    - Aucune variable n'utilise d'information postérieure à l'instant prévu.
    """
    df = serie.to_frame("consommation")
    local = df.index.tz_convert(FUSEAU)

    # 1. Calendrier (heure de Paris)
    df["heure"] = local.hour + local.minute / 60
    df["jour_semaine"] = local.day_of_week          # 0 = lundi, 6 = dimanche
    df["mois"] = local.month
    df["jour_annee"] = local.day_of_year
    df["est_week_end"] = (df["jour_semaine"] >= 5).astype(int)

    feries = holidays.France(years=range(local.year.min(), local.year.max() + 1))
    df["est_ferie"] = np.array([jour in feries for jour in local.date], dtype=int)

    # 2. Encodage cyclique (utile pour Ridge)
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)

    df["annee_sin"] = np.sin(2 * np.pi * df["jour_annee"] / 365.25)
    df["annee_cos"] = np.cos(2 * np.pi * df["jour_annee"] / 365.25)

    # 3. Décalages (lags) : uniquement du passé
    df["lag_24h"] = df["consommation"].shift(PAS_PAR_JOUR)
    df["lag_168h"] = df["consommation"].shift(7 * PAS_PAR_JOUR)

    # Les premières lignes n'ont pas de passé disponible : on les retire
    df = df.dropna()

    # y = valeur réelle à prédire ; X = variables explicatives
    y = df.pop("consommation")
    return df, y