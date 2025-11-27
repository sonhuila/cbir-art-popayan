import numpy as np

eps = 1e-10

def chi_square(h1, h2):
    num = (h1 - h2)**2
    den = h1 + h2 + eps
    return 0.5 * np.sum(num / den)   # a veces se usa factor 0.5

def bhattacharyya(h1, h2):
    # h1, h2 deben estar normalizados (suman 1)
    bc = np.sum(np.sqrt(h1 * h2))
    return np.sqrt(max(0.0, 1 - bc))  # Hellinger distance

def l2_dist(x, y):
    return np.linalg.norm(x - y)

def l1_dist(x, y):
    return np.sum(np.abs(x - y))

def cosine_dist(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < eps or ny < eps:
        return 1.0
    return 1.0 - (np.dot(x, y) / (nx * ny))

def hamming_dist(bin1, bin2):
    # bin vectors of 0/1 or booleans
    return np.mean(np.not_equal(bin1, bin2))
