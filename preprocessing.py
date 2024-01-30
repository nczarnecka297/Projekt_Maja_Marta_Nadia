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

        roi_filtered = cv2.medianBlur(roi, 5)
        roi_filtered = cv2.GaussianBlur(roi_filtered, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(roi_filtered), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            clahe_filtered = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe_filtered = clahe.apply(frame)

        yield clahe_filtered

        processed_frames.append(roi_filtered)

    cap.release()

def segmentation(frame, use_canny=True, threshold1=100, threshold2=200):

    """Segmentuje rękę na obrazie stosując wybraną metodę segmentacji.

    :param frame: Ramka wideo do segmentacji.
    :type frame: numpy.ndarray
    :param use_canny: Jeśli True, używa detekcji krawędzi Canny, w przeciwnym razie stosuje binaryzację.
    :type use_canny: bool
    :param threshold1: Pierwszy próg dla operatora Canny.
    :type threshold1: int
    :param threshold2: Drugi próg dla operatora Canny.
    :type threshold2: int
    :return: Obraz z segmentowaną ręką.
    :rtype: numpy.ndarray
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if use_canny:
        # Użycie detekcji krawędzi Canny
        edged = cv2.Canny(gray, threshold1, threshold2)
        return edged
    else:
        # Binaryzacja obrazu
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary


def pca_to_frame(frame, n_components=50):

    """Stosuje PCA do pojedynczej ramki w celu redukcji wymiarowości i ekstrakcji cech.

    :param frame: Ramka wideo do przetworzenia.
    :type frame: numpy.ndarray
    :param n_components: Liczba głównych składowych do zachowania.
    :type n_components: int
    :return: Ramka po redukcji wymiarowości.
    :rtype: numpy.ndarray
    """

    flat_frame = frame.flatten().reshape(1, -1)
    pca = PCA(n_components=n_components)
    pca_frame = pca.fit_transform(flat_frame)

    return pca_frame
