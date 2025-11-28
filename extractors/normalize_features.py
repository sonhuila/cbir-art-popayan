"""
Módulo de normalización para diccionarios de características.

Proporciona funciones para normalizar diferentes tipos de vectores
de características. La función principal, normalize_feature_dict,
actúa como un enrutador que aplica la normalización adecuada a cada
característica según su tipo.
"""

import numpy as np

eps = 1e-10


def normalize_histogram(h):
    """
    Normaliza un histograma para que la suma de sus componentes sea 1.

    Args:
        h: El histograma de entrada como un vector de NumPy

    Returns:
        Histograma normalizado
    """
    return h / (np.sum(h) + eps)


def l2_normalize(v):
    """
    Realiza la normalización L2 en un vector (lo escala a longitud unitaria).

    Args:
        v: El vector de entrada

    Returns:
        Vector normalizado
    """
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def normalize_feature_dict(feat):
    """
    Aplica la normalización adecuada a cada característica en un diccionario

    Itera a través de un diccionario de características y aplica una técnica
    de normalización específica basada en la clave del diccionario.

    Args:
        feat: Un diccionario de características cruda (no normalizadas).

    Returns:
        dict: Un nuevo diccionario con las características normalizadas.
    """

    feat_norm = {}

    if "lbp_histogram" in feat:
        feat_norm["lbp_histogram"] = normalize_histogram(feat["lbp_histogram"])

    if "color_moments" in feat:
        feat_norm["color_moments"] = l2_normalize(feat["color_moments"])

    if "haralick_features" in feat:
        feat_norm["haralick_features"] = l2_normalize(feat["haralick_features"])

    if "orb" in feat:
        feat_norm["orb"] = feat["orb"]

    return feat_norm
