import numpy as np

from .normalize_features import normalize_feature_dict
from .combine_features import combine_features

def build_feature_vector(raw_features):
    """
    raw_features: diccionario con los vectores SIN normalizar
    Devuelve un vector final listo para guardar o usar en similitud
    """
    # 1. Normalización por descriptor
    normalized = normalize_feature_dict(raw_features)

    # Si no hay características normalizadas, devuelve un vector vacío.
    if not normalized:
        return np.array([])

    # 2. Combinación (concatenación)
    combined = combine_features(normalized)

    # 3. Normalización final opcional
    # Asegurarse de que el vector no sea nulo para evitar división por cero
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined_norm = combined / norm
    else:
        combined_norm = combined

    return combined_norm