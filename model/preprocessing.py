#funkcje na wstepne przetworzenie obrazu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump
import cv2
import numpy as np
import json
import os
from skimage.feature import hog
from skimage import exposure

data_folder = os.getcwd()
nslt_json_path = os.path.join(data_folder, 'nslt_100.json')
wlasl_json_path = os.path.join(data_folder, 'WLASL_v0.3.json')

with open(nslt_json_path, 'r') as file:
    nslt_data = json.load(file)

with open(wlasl_json_path, 'r') as file:
    wlasl_data = json.load(file)

def preprocessing(video_path, bbox, frame_start, frame_end):
    #trzeba bedzie zaktualizowac dokumentacje Maja
    """Przetwarza obraz z pliku wideo stosując redukcję szumów i poprawę kontrastu.
    :param video_path: Ścieżka do pliku wideo do przetworzenia.
    :type video_path: str
    """

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
        current_frame += 1

        if current_frame < frame_start or (frame_end != -1 and current_frame > frame_end):
            continue

        #tniemy do regionu zainteresowania ROI
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(roi), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            clahe_filtered = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe_filtered = clahe.apply(frame)

        yield clahe_filtered
        
        roi_filtered = cv2.medianBlur(clahe filtered, 5)
        roi_filtered = cv2.GaussianBlur(roi_filtered, (5, 5), 0)
        edged = cv2.Canny(roi_filtered, 100, 200)

        processed_frames.append(edged)

    cap.release()
    return processed_frames

def extract_features(processed_frames, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    video_features = []

    for frame in processed_frames:
        hog_features, hog_image = hog(frame,
                                      orientations=orientations,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      block_norm='L2-Hys',
                                      visualize=True)
        
        hog_features = exposure.rescale_intensity(hog_features, in_range=(0, 10))

        video_features.append(hog_features)

    video_features = np.concatenate(video_features, axis=0)

    return video_features

features = []
labels = []

for entry in wlasl_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        video_path = os.path.join(data_folder, f"{video_id}.mp4")

        if os.path.exists(video_path):
            processed_frames = process_video(video_path, bbox, frame_start, frame_end)
            video_features = extract_features(processed_frames)
            features.append(video_features)
            labels.append(gloss)
        else:
            print(f"Video {video_id} for gloss {gloss} not found in the data folder.")

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the machine learning pipeline
pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95), SVC(kernel='rbf', class_weight='balanced'))

# Train the model
pipeline.fit(X_train, y_train)

dump(pipeline, 'sign_language_classifier.joblib')

print(f"Training Accuracy: {pipeline.score(X_train, y_train)}")
print(f"Testing Accuracy: {pipeline.score(X_test, y_test)}")
