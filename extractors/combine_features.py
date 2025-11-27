import numpy as np

def combine_features(features_dict):
    """
    Combina todas las características del diccionario en un único vector.
    """
    # Ordenar por clave para asegurar un orden consistente
    feature_vectors = [features_dict[key] for key in sorted(features_dict.keys())]
    
    if not feature_vectors:
        return np.array([])
        
    return np.concatenate(feature_vectors)