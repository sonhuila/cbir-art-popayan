import cv2
import numpy as np

def extract_orb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros(32)
    return np.mean(des, axis=0)
