#funkcje na wstepne przetworzenie obrazu
import cv2
import numpy as np
import json
import os

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
