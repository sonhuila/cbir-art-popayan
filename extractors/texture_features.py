import cv2
import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

# --- Textura 1: LBP ---
def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(59))
    hist = hist / np.sum(hist)
    return hist

# --- Textura 2: Haralick ---
def extract_haralick(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    features = [
        greycoprops(glcm, 'contrast')[0, 0],
        greycoprops(glcm, 'homogeneity')[0, 0],
        greycoprops(glcm, 'energy')[0, 0],
        greycoprops(glcm, 'correlation')[0, 0]
    ]
    return np.array(features)
