
"""Métriques d'évaluation des prévisions."""
import numpy as np


def mae(y_reel, y_prevu) -> float:
    """Erreur absolue moyenne, en MW."""
    y_reel = np.asarray(y_reel, dtype=float)
    y_prevu = np.asarray(y_prevu, dtype=float)
    return float(np.mean(np.abs(y_reel - y_prevu)))


def rmse(y_reel, y_prevu) -> float:
    """Racine de l'erreur quadratique moyenne, en MW."""
    y_reel = np.asarray(y_reel, dtype=float)
    y_prevu = np.asarray(y_prevu, dtype=float)
    return float(np.sqrt(np.mean((y_reel - y_prevu) ** 2)))


def mape(y_reel, y_prevu) -> float:
    """Erreur absolue moyenne en pourcentage."""
    y_reel = np.asarray(y_reel, dtype=float)
    y_prevu = np.asarray(y_prevu, dtype=float)
    return float(np.mean(np.abs((y_reel - y_prevu) / y_reel)) * 100)
