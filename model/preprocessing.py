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

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(gray)
        
        filtered = cv2.medianBlur(clahe_applied, 5)
        filtered = cv2.GaussianBlur(filtered, (5, 5), 0)
        edged = cv2.Canny(filtered, 100, 200)

        processed_frames.append(edged)

    cap.release()
    return processed_frames

def extract_features(processed_frames):

    features = []

    orb = cv2.ORB_create()

    for frame in processed_frames:
        keypoints, descriptors = orb.detectAndCompute(frame, None)

        if descriptors is None:
            features.append([])
        else:
            features.append(descriptors.flatten())

    return np.array(features, dtype=object)

features = []
labels = []

features_train = []
labels_train = []

features_test = []
labels_test = []

features_val = []
labels_val = []

subset_info = {}

for entry in wlasl_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        video_path = os.path.join(data_folder, f"{video_id}.mp4")

        if os.path.exists(video_path):
            processed_frames = preprocessing(video_path, bbox, frame_start, frame_end)
            video_features = extract_features(processed_frames)
            video_features_flattened = np.concatenate(video_features)
            features.append(video_features_flattened)
            labels.append(gloss)

            subset = subset_info.get(video_id)
            if subset:
                if subset == 'train':
                    features_train.append(video_features_flattened)
                    labels_train.append(gloss)
                elif subset == 'test':
                    features_test.append(video_features_flattened)
                    labels_test.append(gloss)
                elif subset == 'val':
                    features_val.append(video_features_flattened)
                    labels_val.append(gloss)

        else:
            print(f"Video {video_id} for gloss {gloss} not found in the data folder.")

features = np.array(features)
labels = np.array(labels)

min_length = min(len(f) for f in features)

features_train = np.array([f[:min_length] if len(f) > min_length else np.pad(f, (0, min_length - len(f)), 'constant') for f in features_train])
features_test = np.array([f[:min_length] if len(f) > min_length else np.pad(f, (0, min_length - len(f)), 'constant') for f in features_test])
features_val = np.array([f[:min_length] if len(f) > min_length else np.pad(f, (0, min_length - len(f)), 'constant') for f in features_val])

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
labels_val = np.array(labels_val)

pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95), SVC(kernel='rbf', class_weight='balanced'))

pipeline.fit(features_train, labels_train)

dump(pipeline, 'sign_language_classifier.joblib')

print(f"Training Accuracy: {pipeline.score(features_train, labels_train)}")
print(f"Testing Accuracy: {pipeline.score(features_test, labels_test)}")