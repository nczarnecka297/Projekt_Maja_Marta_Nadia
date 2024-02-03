import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage import exposure

def preprocessing(video_path):
    
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    processed_frames = []

    if not cap.isOpened():
        print("Nie udało się otworzyć pliku wideo. Sprawdź ścieżkę.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(gray)

        filtered = cv2.GaussianBlur(clahe_applied, (5, 5), 0)
        filtered = cv2.medianBlur(filtered, 5)

        edged = cv2.Canny(filtered, 100, 200)

        processed_frames.append(edged)

    cap.release()
    return processed_frames

def extract_features(processed_frames):

    orb = cv2.ORB_create()

    features = []

    for frame in processed_frames:
        keypoints, descriptors = orb.detectAndCompute(frame, None)

        if descriptors is None:
            features.append([])
        else:
            features.append(descriptors.flatten())

    return np.array(features, dtype=object)
