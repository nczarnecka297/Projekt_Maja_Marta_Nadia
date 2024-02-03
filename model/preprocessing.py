from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump, load
import cv2
import numpy as np
import json
import os
from skimage import exposure
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

data_folder = os.path.join(r'data')
wlasl_json_path = os.path.join(data_folder, 'WLASL_v0.3.json')

with open(wlasl_json_path, 'r') as file:
    wlasl_data = json.load(file)

def preprocessing(video_path, bbox, frame_start, frame_end):
   
    """
    Process video frames by reducing noise, enhancing contrast, and applying edge detection.
    
    :param video_path: Path to the video file to be processed.
    :type video_path: str
    :param bbox: Bounding box coordinates (x, y, width, height) for the region of interest (ROI).
    :type bbox: tuple
    :param frame_start: Starting frame for processing (optional).
    :type frame_start: int
    :param frame_end: Ending frame for processing (optional).
    :type frame_end: int
    :return: List of processed frames.
    :rtype: list
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
    
    """
    Extract ORB (Oriented FAST and Rotated BRIEF) features from processed video frames.

    :param processed_frames: List of processed video frames.
    :type processed_frames: list
    :return: Extracted features as a NumPy array.
    :rtype: numpy.ndarray
    """

    features = []

    orb = cv2.ORB_create()

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

batch_size = 10
features_folder = 'processed_features'

if not os.path.exists(features_folder):
    os.makedirs(features_folder)

batch_features = []
batch_labels = []
batch_index = 1

for entry in wlasl_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        video_path = os.path.join(data_folder, f"{video_id}.mp4")

        if os.path.exists(video_path):
            print(f"Processing video {video_id} for gloss {gloss}...")
            processed_frames = preprocessing(video_path, bbox, frame_start, frame_end)
            video_features = extract_features(processed_frames)
            video_features_flattened = np.concatenate(video_features)
            batch_features.append(video_features_flattened)
            batch_labels.append(gloss)
            print("Done")
        else:
            pass#print(f"Video {video_id} for gloss {gloss} not found in the data folder.")

        if len(batch_features) >= batch_size:
            features_filename = os.path.join(features_folder, f"{batch_index}_batch_features.npy")
            print(f"Saving features to {features_filename}...")
            np.save(features_filename, np.array(batch_features))
            print(f"Saved {len(batch_features)} features.")
            batch_features = []
            batch_labels = []
            batch_index += 1

if batch_features:
    features_filename = os.path.join(features_folder, f"{batch_index}_batch_features.npy")
    np.save(features_filename, np.array(batch_features))

loaded_features = []
for filename in os.listdir(features_folder):
    if filename.endswith("_batch_features.npy"):
        batch_features = np.load(os.path.join(features_folder, filename), allow_pickle=True)
        loaded_features.extend(batch_features)

X_train, X_test, y_train, y_test = train_test_split(loaded_features, labels, test_size=0.2, random_state=42)

features_as_strings = [" ".join(map(str, feature)) for feature in X_train]
vectorizer = CountVectorizer()
features_transformed = vectorizer.fit_transform(features_as_strings)

explained_variance_ratio = TruncatedSVD(n_components=features_transformed.shape[1]).fit(features_transformed).explained_variance_ratio_
n_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95)

pipeline = make_pipeline(StandardScaler(with_mean=False), PCA(n_components=n_components), SVC(kernel='rbf', class_weight='balanced'))

print("Training model..")
pipeline.fit(features_transformed, y_train)

dump(pipeline, 'sign_language_classifier.joblib')

X_test_as_strings = [" ".join(map(str, feature)) for feature in X_test]
X_test_transformed = vectorizer.transform(X_test_as_strings)
test_accuracy = pipeline.score(X_test_transformed, y_test)

print(f"Accuracy on test data: {test_accuracy}")
