import numpy as np

def weighted_distance(dist_dict, weights):
    """
    Combina distancias con pesos.
    """
    total = 0
    for key in dist_dict:
        total += weights[key] * dist_dict[key]
    return total


def compute_global_distance(query_feats, db_feats, weights, distance_fns):
    """
    Calcula la distancia global entre dos imágenes usando varios descriptores.
    query_feats y db_feats son diccionarios:
      { 'hsv': vec, 'moments': vec, 'lbp': vec, 'haralick': vec }
    """
    dist_dict = {}

    for key in query_feats:
        dist_dict[key] = distance_fns[key](query_feats[key], db_feats[key])
    
    global_dist = weighted_distance(dist_dict, weights)
    return global_dist


def rank_images(query_features, database, weights, distance_fns, top_k=10):
    """
    Ordena todas las imágenes del dataset según su similitud con la query.
    database = [
        { 'id': '001', 'features': {...} },
        { 'id': '002', 'features': {...} }
    ]
    """
    results = []

    for item in database:
        dist = compute_global_distance(
            query_features,
            item['features'],
            weights,
            distance_fns
        )
        results.append((item['id'], dist))

    # Menor distancia = más similar
    results.sort(key=lambda x: x[1])

    return results[:top_k]


def get_top_ids(ranking):
    """
    Devuelve solo los IDs del ranking.
    """
    return [item[0] for item in ranking]
