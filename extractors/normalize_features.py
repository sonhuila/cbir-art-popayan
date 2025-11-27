import numpy as np

eps = 1e-10


def normalize_histogram(h):
    """Normaliza histogramas para sum=1."""
    return h / (np.sum(h) + eps)


def l2_normalize(v):
    """Normalización L2."""
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def normalize_feature_dict(feat):
    """
    Normaliza cada descriptor según su tipo.
    IMPORTANTE:
    - ORB NO se normaliza (se mantiene binario).
    """

    feat_norm = {}

    # =============== Histogramas ===============
    if "lbp_histogram" in feat:
        feat_norm["lbp_histogram"] = normalize_histogram(feat["lbp_histogram"])

    # =============== Vectores numéricos ===============
    if "color_moments" in feat:
        feat_norm["color_moments"] = l2_normalize(feat["color_moments"])

    if "haralick_features" in feat:
        feat_norm["haralick_features"] = l2_normalize(feat["haralick_features"])

    # =============== ORB (NO normalizar) ===============
    if "orb" in feat:
        feat_norm["orb"] = feat["orb"]   # tal cual

    return feat_norm
