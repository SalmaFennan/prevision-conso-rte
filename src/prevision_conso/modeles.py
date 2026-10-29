"""Entraînement et comparaison des modèles : Ridge, Random Forest, XGBoost.

- Jeu d'entraînement : avant DEBUT_TEST ; jeu de test : l'année qui suit (utilisé une seule fois).
- Validation croisée temporelle (TimeSeriesSplit) uniquement sur le jeu d'entraînement.
- Les baselines sont évaluées sur la même période de test, pour une comparaison équitable.
"""
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from prevision_conso.acquisition import CHEMIN_PARQUET, RACINE
from prevision_conso.baseline import DEBUT_TEST, PAS_PAR_JOUR
from prevision_conso.metriques import mae, mape, rmse
from prevision_conso.preparation import charger_serie
from prevision_conso.variables import construire_variables

DOSSIER_RAPPORTS = RACINE / "reports"
CHEMIN_PREDICTIONS = RACINE / "data" / "processed" / "predictions_test.parquet"


def definir_modeles() -> dict:
    """Les trois modèles comparés, avec des hyperparamètres raisonnables par défaut."""
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=18, min_samples_leaf=10,

            n_jobs=-1, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        ),
    }


def separer_train_test(X: pd.DataFrame, y: pd.Series):
    """Découpe temporelle : tout ce qui précède DEBUT_TEST sert à l'entraînement."""
    est_test = X.index >= DEBUT_TEST
    return X[~est_test], X[est_test], y[~est_test], y[est_test]


def valider(modele, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    """Validation croisée temporelle : on apprend sur le passé et on valide sur la suite."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=PAS_PAR_JOUR)
    scores = cross_validate(
        modele, X_train, y_train, cv=tscv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "mape": "neg_mean_absolute_percentage_error",
        },
    )
    # scikit-learn renvoie des scores négatifs ("plus grand = meilleur") : on les remet en positif
    return {
        "cv_mae": -scores["test_mae"].mean(),
        "cv_mae_ecart_type": scores["test_mae"].std(),
        "cv_rmse": -scores["test_rmse"].mean(),
        "cv_mape": -scores["test_mape"].mean() * 100,

    }


def evaluer(y_reel, y_prevu) -> dict:
    """Métriques sur le jeu de test."""
    return {
        "test_mae": mae(y_reel, y_prevu),
        "test_rmse": rmse(y_reel, y_prevu),
        "test_mape": mape(y_reel, y_prevu),
    }


def main() -> None:
    serie = charger_serie(CHEMIN_PARQUET)
    X, y = construire_variables(serie)
    X_train, X_test, y_train, y_test = separer_train_test(X, y)
    print(f"Entraînement : {len(X_train)} points | Test : {len(X_test)} points\n")

    resultats = []
    predictions = pd.DataFrame({"reel": y_test})

    # 1. Baselines sur la même période de test (lag_24h = prévision J-1, lag_168h = J-7)
    for nom, colonne in [("Naïve J-7", "lag_168h"), ("Naïve J-1", "lag_24h")]:
        predictions[nom] = X_test[colonne]
        resultats.append({"modele": nom, **evaluer(y_test, X_test[colonne])})

    # 2. Modèles : validation croisée, puis entraînement final et évaluation sur le test
    importances = {}
    for nom, modele in definir_modeles().items():
        debut = time.perf_counter()
        print(f"{nom} : validation croisée temporelle puis entraînement...")
        cv = valider(modele, X_train, y_train)

        modele.fit(X_train, y_train)
        prevu = modele.predict(X_test)
        predictions[nom] = prevu
        duree = round(time.perf_counter() - debut)
        resultats.append({"modele": nom, **cv, **evaluer(y_test, prevu), "duree_s": duree})
        if hasattr(modele, "feature_importances_"):
            importances[nom] = pd.Series(modele.feature_importances_, index=X.columns)

    # 3. Sauvegarde et affichage
    DOSSIER_RAPPORTS.mkdir(exist_ok=True)
    tableau = pd.DataFrame(resultats).set_index("modele").round(2)
    tableau.to_csv(DOSSIER_RAPPORTS / "resultats_modeles.csv")
    tableau_importances = pd.DataFrame(importances).round(3).sort_values("XGBoost", ascending=False)
    tableau_importances.to_csv(DOSSIER_RAPPORTS / "importances_variables.csv")
    predictions.to_parquet(CHEMIN_PREDICTIONS)   # utilisé à l'étape 7 pour les graphiques

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Résultats ===")
    print(tableau)
    print("\n=== Importance des variables ===")
    print(tableau_importances)


if __name__ == "__main__":
    main()