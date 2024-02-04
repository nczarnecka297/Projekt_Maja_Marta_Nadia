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

    """
    Preprocesses a video file by reducing noise and enhancing contrast.

    :param str video_path: The path to the video file to be processed.
    :return: A list of processed video frames.
    :rtype: list[numpy.ndarray]

    Note:
        The video frames are processed using techniques such as grayscale conversion,
        CLAHE (Contrast Limited Adaptive Histogram Equalization), Gaussian blur, median blur,
        and Canny edge detection.
        If an error occurs while opening or processing the video, an error message is displayed.
    """
    
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

    """
    Extracts features from a list of processed video frames.

    :param list[numpy.ndarray] processed_frames: A list of processed video frames.
    :return: Extracted features from the video frames.
    :rtype: numpy.ndarray

    Note:
        Features are extracted using the ORB (Oriented FAST and Rotated BRIEF) algorithm,
        and then flattened into a one-dimensional array.
        The length of features is adjusted to the mean length of all extracted features.
    """

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
scaler = load('scaler.joblib')
svd = load('svd.joblib')

def browse_file():
    
    """
    Open a file dialog to select an MP4 video file, process it, and display the translation.

    When this function is called, a file dialog is opened to allow the user to select an MP4 video file.
    The selected video file is then preprocessed, and its features are extracted.
    The extracted features are transformed using a vectorizer, and a translation is predicted using a pre-trained model.
    The resulting translation is displayed in the GUI.

    :return: None
    :rtype: None
    """
    
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        video_name = os.path.basename(file_path)
        selected_video_label.config(text=f"Selected video: {video_name}", fg="blue")
        
        update_status("Translating...", "blue")

        processed_frames = preprocessing(file_path)
        video_features = extract_features(processed_frames)
        video_features_flattened = np.concatenate(video_features)

        features_as_strings = [" ".join(map(str, video_features_flattened))]
        features_transformed = vectorizer.transform(features_as_strings)
        
        X_scaled = scaler.transform(features_transformed)
        X_reduced = svd.transform(X_scaled)

        translation = model.predict(X_reduced)
        update_status(f"Translation: {translation[0]}", "purple")
        browse_button.config(state="normal")
        
def finish_translation(translation):
    """
    Finalizes the translation process by updating the status label with the translation result and re-enabling the browse button.

    :param translation: The translated text to be displayed.
    :type translation: str
    """
    update_status(f"Translation: {translation}", "purple")
    browse_button.config(state="normal")

def update_status(message, color="red"):
    """
    Updates the status label with a given message and changes its color. It also clears any previous error messages.

    :param message: The message to be displayed on the status label.
    :type message: str
    :param color: The color of the text to be displayed, defaults to "red".
    :type color: str
    """
    status_label.config(text=message, fg=color)
    error_label.config(text='')
    root.update_idletasks()
    
root = tk.Tk()
root.title("American Sign Language Video Translator")
root.geometry("500x250")

root.configure(bg="#E6E6FA")
font = tkfont.Font(family="Roboto", size=13)

selected_video_label = tk.Label(root, text="No video selected", font=font, bg="#E6E6FA", fg="blue")
selected_video_label.pack(pady=(10, 0))

browse_button = tk.Button(root, text="Choose video file ('*.mp4')", font=("Roboto", 14), command=lambda: [browse_button.config(state="disabled"), browse_file()], bg="#9370DB")
browse_button.pack(pady=(10, 5))

status_label = tk.Label(root, text="", font=font, bg="#E6E6FA")
status_label.pack(pady=(5, 10))

error_label = tk.Label(root, text="", font=font, bg="#E6E6FA", fg="red")
error_label.pack()

root.mainloop()
