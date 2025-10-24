import os
import cv2
import numpy as np
from extractors.color_features import extract_color_moments
from extractors.texture_features import extract_lbp, extract_haralick
from extractors.keypoint_features import extract_orb

def extract_features(img):
    color = extract_color_moments(img)
    lbp = extract_lbp(img)
    har = extract_haralick(img)
    orb = extract_orb(img)
    return np.concatenate((color, lbp, har, orb))

def process_dataset(dataset_path="dataset/wikiart"):
    features, filenames = [], []
    for file in os.listdir(dataset_path):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(dataset_path, file)
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))
            vec = extract_features(img)
            features.append(vec)
            filenames.append(file)
    np.save("features.npy", features)
    np.save("filenames.npy", filenames)
    print(f"✅ {len(features)} imágenes procesadas y guardadas.")

if __name__ == "__main__":
    process_dataset()
