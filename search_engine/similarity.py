import numpy as np
from scipy.spatial import distance

def find_similar(query_features, db_features, filenames, top_n=5):
    dists = [distance.euclidean(query_features, f) for f in db_features]
    idxs = np.argsort(dists)[:top_n]
    return [(filenames[i], dists[i]) for i in idxs]
