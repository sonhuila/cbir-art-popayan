"""
Colección de funciones de distancia para comparar vectores de características.

Este módulo proporciona varias métricas para calcular la distancia o disimilitud
entre dos vectores numéricos. Cada función está optimizada para un tipo diferente
de descriptor de características.

Estas funciones son utilizadas por el motor de ranking.
"""

import numpy as np

# Constante para la estabilidad numérica en divisiones
eps = 1e-10

def chi_square(h1, h2):
    """
    Calcula la distancia Chi-cuadrado para comparar distribuciones
    de frecuencia.

    Args:
        h1: Primer histograma.
        h2: Segundo histograma.

    Returns:
        float: Distancia Chi-cuadrado.
    """
    num = (h1 - h2)**2
    den = h1 + h2 + eps
    return 0.5 * np.sum(num / den)

def l2_dist(x, y):
    """
    Calcula la distancia Euclidiana entre dos vectores.

    Es la distancia en línea recta entre dos puntos en un espacio multidimensional.
    
    Args:
        x: El primer vector.
        y: El segundo vector.

    Returns:
        float: Distancia Euclidiana.
    """
    return np.linalg.norm(x - y)

def hamming_dist(bin1, bin2):
    """
    Calcula la distancia de Hamming normalizadas para vectores binarios.

    Mide la proporción de posiciones en las que dos vectores difieren. Es
    la métrica ideal para descriptores binarios.

    Args:
        bin1: El primer vector binario.
        bin2: El segundo vector binario.
    
    Returns:
        float: Distancia de Hamming normalizada entre 0 y 1.
    """
    return np.mean(np.not_equal(bin1, bin2))
