"""
    Motor de ranking para la búsqueda de imágenes por similitud.

    Este módulo calcula una distancia final ponderada para ordenar 
    imágenes. Es responsable de tomar un diccionario de características 
    de una imagen  de consulta y compararlo con toda una base de datos, 
    devolviendo una lista ordenada de los resultados más similares.

    El flujo principal es manejado por la función `rank_images`, que utiliza
    funciones auxiliares para calcular distancias individuales y combinarlas
    según pesos definidos.
"""

import numpy as np

def weighted_distance(dist_dict, weights):
    """
    Calcula la distancia final como una suma ponderada de distancias individuales.

    Args:
        dist_dict (dict): Diccionario con distancias individuales por descriptor.
        weights (dict): Diccionario que asigna un peso o importancia a cada tipo de característica.

    Returns:
        float: Distancia ponderada total.
    """
    total = 0
    for key in dist_dict:
        total += weights[key] * dist_dict[key]
    return total


def compute_global_distance(query_feats, db_feats, weights, distance_fns):
    """
    Calcula la distancia global entre dos items (consulta y base de datos).

    Esta función orquesta el cálculo de la distancia para un solo par de imágenes,
    desplegando a las funciones de distancia y luego combinando los resultados con
    weighted_distance.

    Args:
        query_feats (dict): Diccionario de características de la imagen de consulta.
        db_feats (dict): Diccionario de características de la imagen en la base de datos.
        weights (dict): Diccionario de pesos para cada tipo de característica.
        distance_fns (dict): Diccionario que mapea cada característica a su función
                            de distancias correspondiente.

    Returns:
        float: Distancia global ponderada entre la consulta y la imagen de la base de datos
    """
    dist_dict = {}

    for key in query_feats:
        dist_dict[key] = distance_fns[key](query_feats[key], db_feats[key])
    
    global_dist = weighted_distance(dist_dict, weights)
    return global_dist


def rank_images(query_features, database, weights, distance_fns, top_k):
    """
    Ordena todas las imágenes del dataset según su similitud con la consulta.
    
    Esta es la función principal del módulo. Itera sobre toda la base de datos,
    calcula una distancia final para cada imagen usando compute_global_distance,
    y devuelve una lista ordenada de los 'top_k' resultados más similares.

    Args:
        query_features Diccionario de características de la imagen de consulta.
        database: Lista completa de items de la base de datos.
        weights: Diccionario de pesos para la suma ponderada.
        distance_fns: Diccionario que mapea cada característica a su función
                            de distancias correspondiente.
        top_k: Número de resultados más similares a devolver.

    Returns:
        list: Lista de tuplas de solo el top_k (image_id, distance) ordenadas de menor a mayor distancia.
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
    Extrae y devuelve solo los IDs de imagen del ranking ordenado.

    Es una función de utilidad para convertir el resultado de rank_images
    en una lista simple de identificadores de imagen.

    Args:
        ranking: Lista de tuplas (image_id, distance) ordenadas.

    Returns:
        List[str]: Lista de IDs de imagen ordenados por similitud.
    """
    return [item[0] for item in ranking]

def rank_images_by_single_vector(query_vector, db_vectors, distance_fn, top_k=20):
    """
    Calcula la distancia entre un vector de consulta y una lista de vectores de la base de datos,
    y devuelve los 'top_k' resultados más cercanos.

    Args:
        query_vector (np.array): El vector de características concatenado de la imagen de consulta.
        db_vectors (list): Una lista de tuplas (item_id, vector_concatenado) de la base de datos.
        distance_fn (function): La función de distancia a utilizar (ej. l2_dist).
        top_k (int): El número de resultados a devolver.

    Returns:
        list: Una lista de tuplas (distancia, item_id) para los 'top_k' mejores resultados.
    """
    if query_vector.size == 0:
        return []

    results = []
    for item_id, db_vector in db_vectors:
        if db_vector.size == 0:
            continue
        # Asegurarse de que los vectores tengan la misma longitud
        if query_vector.shape == db_vector.shape:
            dist = distance_fn(query_vector, db_vector)
            results.append((dist, item_id))

    # Ordenar los resultados por distancia (ascendente)
    results.sort(key=lambda x: x[0])

    return results[:top_k]