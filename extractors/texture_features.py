"""
Módulo para la extracción de características de textura.

Contiene funciones para extraer descriptores de textura basado en
Patrone Binarios Locales (LBP) y matrices de co-ocurrencia (Haralick).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops

def extract_lbp(img):
    """
    Extrae el histograma de Patrones Binarios Locales (LBP) de una imagen.

    LBP es un descriptor de textura que codifica la relación de un píxel con
    sus vecinos. Esta función calcula el histograma LBP.

    Args:
        img: Imagen de entrada en formato BGR.
    
    Returns:
        hist: Histograma LBP de la imagen.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59))
    return hist

def extract_haralick(img):
    """
    Extrae características de textura de Haralick a partir de una GLCM.

    Calcula la Matriz de Co-ocurrencia de Niveles de Gris y extrae cuatro
    propiedades estadísticas: contraste, homogeneidad, energía y correlación.

    Args:
        img: Imagen de entrada en formato BGR.

    Returns:
        features: Vector de características de Haralick.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return np.array(features)
