from joblib import load
import cv2
import numpy as np
import os
from skimage import exposure
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont

error_label = None
def preprocessing(video_path):
    
    cap = cv2.VideoCapture(video_path)
    current_frame = 0
    processed_frames = []

    if not cap.isOpened():
        error_label.config(text="Error: unable to open file.")
        return []

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
    if not processed_frames:
        error_label.config(text="Error: unable to process video.")
    else:
        error_label.config(text="")
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

    mean_features_length = int(np.mean([len(f) for f in features]))
    padded_features = []
    for feature in features:
        if len(feature) < mean_features_length:
            padded_feature = np.pad(feature, (0, mean_features_length - len(feature)), 'constant')
        else:
            padded_feature = feature[:mean_features_length]
        padded_features.append(padded_feature)

    return np.array(padded_features, dtype=object)

model = load('sign_language_classifier.joblib')
vectorizer = load('vectorizer.joblib')
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        processed_frames = preprocessing(file_path)
        video_features = extract_features(processed_frames)
        video_features_flattened = np.concatenate(video_features)

        features_as_string = " ".join(map(str, video_features_flattened))
        features_transformed = vectorizer.transform([features_as_string])

        translation = model.predict(features_transformed)
        result_label.config(text=f"Translation: {translation[0]}")

root = tk.Tk()
root.title("American Sign Language Video Translator")
root.geometry("400x200")

root.configure(bg="#E6E6FA")
font = tkfont.Font(family="Roboto", size=12)

result_label = tk.Label(root, text="", font=("Arial", 12), bg="#E6E6FA")
result_label.pack(pady=20)

browse_button = tk.Button(root, text="Choose video file ('*.mp4')", font=("Arial", 14), command=browse_file, bg="#9370DB")
browse_button.pack()

error_label = tk.Label(root, text="", font=font, bg="#E6E6FA", fg="red")
error_label.pack()

root.mainloop()