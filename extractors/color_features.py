import cv2
import numpy as np

def extract_color_moments(img):
    # Calcula media, desviación y asimetría por canal RGB
    features = []
    for i in range(3):
        channel = img[:, :, i].ravel()
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-8)
        features.extend([mean, std, skew])
    return np.array(features)
