"""Analyse des erreurs du modèle retenu et graphiques pour le README."""
import matplotlib

matplotlib.use("Agg")   # enregistre les figures sans les afficher à l'écran

import holidays
import matplotlib.pyplot as plt
import pandas as pd

from prevision_conso.modeles import CHEMIN_PREDICTIONS, DOSSIER_RAPPORTS

MODELE = "Random Forest"      # choisi sur la validation croisée
REFERENCE = "Naïve J-1"
FUSEAU = "Europe/Paris"
DOSSIER_FIGURES = DOSSIER_RAPPORTS / "figures"
NOMS_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def charger_predictions() -> pd.DataFrame:
    """Prédictions du jeu de test, en heure de Paris pour l'analyse."""
    predictions = pd.read_parquet(CHEMIN_PREDICTIONS)
    predictions.index = predictions.index.tz_convert(FUSEAU)
    return predictions


def erreurs_absolues(predictions: pd.DataFrame) -> pd.DataFrame:
    """Erreur absolue du modèle et de la référence, point par point."""
    return pd.DataFrame({
        nom: (predictions["reel"] - predictions[nom]).abs() for nom in [MODELE, REFERENCE]
    })



def graphique_prevu_vs_reel(predictions: pd.DataFrame) -> None:
    """Une semaine d'hiver et une semaine d'été : réel, modèle et baseline."""
    semaines = {
        "Semaine d'hiver (12-18 janvier 2026)": ("2026-01-12", "2026-01-18"),
        "Semaine d'été (6-12 juillet 2026)": ("2026-07-06", "2026-07-12"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, (titre, (debut, fin)) in zip(axes, semaines.items()):
        extrait = predictions.loc[debut:fin]
        ax.plot(extrait.index, extrait["reel"], label="Réel", linewidth=2)
        ax.plot(extrait.index, extrait[MODELE], label=MODELE, linestyle="--")
        ax.plot(extrait.index, extrait[REFERENCE], label=REFERENCE, alpha=0.5)
        ax.set_title(titre)
        ax.set_ylabel("MW")
        ax.legend()
    fig.tight_layout()
    fig.savefig(DOSSIER_FIGURES / "prevu_vs_reel.png", dpi=120)
    plt.close(fig)


def graphique_barres(tableau: pd.DataFrame, titre: str, xlabel: str, nom_fichier: str) -> None:
    """Diagramme en barres de la MAE, modèle contre référence."""
    ax = tableau.plot(kind="bar", figsize=(10, 4))
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (MW)")
    plt.tight_layout()
    plt.savefig(DOSSIER_FIGURES / nom_fichier, dpi=120)
    plt.close()


def main() -> None:

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    predictions = charger_predictions()
    erreurs = erreurs_absolues(predictions)
    pd.set_option("display.width", 200)

    # 1. Biais : le modèle sous-estime-t-il ou surestime-t-il en moyenne ?
    biais = (predictions["reel"] - predictions[MODELE]).mean()
    print(f"Biais de {MODELE} : {biais:+.0f} MW (positif = sous-estimation)\n")

    # 2. Erreur par jour de la semaine
    par_jour = erreurs.groupby(predictions.index.day_of_week).mean().round(0)
    par_jour.index = NOMS_JOURS
    print("=== MAE par jour de la semaine ===")
    print(par_jour, "\n")
    graphique_barres(par_jour, "MAE par jour de la semaine", "Jour", "mae_par_jour.png")

    # 3. Erreur par heure (heure de Paris)
    par_heure = erreurs.groupby(predictions.index.hour).mean().round(0)
    graphique_barres(par_heure, "MAE par heure (heure de Paris)", "Heure", "mae_par_heure.png")
    print("=== Heures les plus difficiles pour le modèle ===")
    print(par_heure[MODELE].nlargest(3), "\n")

    # 4. Erreur par mois
    par_mois = erreurs.groupby(predictions.index.month).mean().round(0)
    print("=== MAE par mois ===")
    print(par_mois, "\n")

    # 5. Les 10 journées les plus mal prévues
    feries = holidays.France(years=[2025, 2026])
    erreur_jour = erreurs[MODELE].resample("D").mean()
    pires = erreur_jour.nlargest(10).round(0).to_frame("mae_du_jour")
    pires["jour"] = [NOMS_JOURS[d.day_of_week] for d in pires.index]

    pires["ferie"] = [feries.get(d.date(), "") for d in pires.index]
    pires.index = pires.index.date
    print("=== Les 10 journées les plus mal prévues ===")
    print(pires)

    # 6. Graphique principal du README
    graphique_prevu_vs_reel(predictions)
    print(f"\nFigures enregistrées dans {DOSSIER_FIGURES}")


if __name__ == "__main__":
    main()