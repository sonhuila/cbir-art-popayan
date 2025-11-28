"""
    Módulo para la extracción de características de color

    Contiene funciones para analizar y cuantificar las propiedades
    de color de una imagen, convirtiéndolas en vectores númericos.
"""

import cv2
import numpy as np

def extract_color_moments(img):
    """
    Extrae los primero tres momentos estadíticos de cada canal de color.

    Calcula la media, la desviación estándar y la asímetria (skewness) para
    cada uno de los tres canales de color (B, G, R) de la imagen. Esto resulta
    en un vector de 9 características qeu resume la distribución de color.

    Args:
        img: La imgane de entrada en formato BGR.

    Returns:
        Un array numpy de 9 elementos que contiene los momentos de color.
    """
    features = []
    for i in range(3):
        channel = img[:, :, i].ravel()
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-8)
        features.extend([mean, std, skew])
    return np.array(features)
