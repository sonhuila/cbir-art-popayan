"""
Módulo de extracción de características de puntos clave (keypoints)

Utiliza el algoritmo ORB para detectar puntos de interés en la imagen
y generar un descriptor agregado
"""

import cv2
import numpy as np

def extract_orb(img):
    """
    Detecta descriptores ORB y los agrega en un único vector de características.

    El método detecta múltiples puntos clave y sus correspondientes descriptores.
    Para obtener un único vector de longitud fija, se calcula la media de todos
    los descriptores.

    Args:
        img: La imagen de entrada de formato BGR.

    Returns:
        Un vector de NumPy de 32 elementos que representa el descriptor ORB
        promedio de la imagen. Si no se detectan puntos clave, devuelve
        un vector de ceros del mismo tamaño.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros(32)
    return np.mean(des, axis=0)
