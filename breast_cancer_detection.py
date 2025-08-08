import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops

# ========== Load Dataset ==========
try:
    X_train = np.load("MIAS_X_train_roi_multi.npy")
    y_train = np.load("MIAS_y_train_roi_multi.npy")
    X_test = np.load("MIAS_X_test_roi_multi.npy")
    y_test = np.load("MIAS_y_test_roi_multi.npy")
    X_valid = np.load("MIAS_X_valid_roi_multi.npy")
    y_valid = np.load("MIAS_y_valid_roi_multi.npy")
except FileNotFoundError:
    print("Dataset not found. Check the file paths.")
    exit()

# ========== Binarize Labels ==========
def binarize(y):
    return np.array([1 if i in [1, 2] else 0 for i in y])

# ========== Feature Extraction ==========
def extract_features(X, y_bin):
    features, labels = [], []

    for i in range(len(X)):
        img = X[i].astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspicious_count, suspicious_area = 0, 0
        circularity_list, solidity_list, extent_list = [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            extent = area / rect_area if rect_area > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0

            if circularity < 0.70:
                suspicious_count += 1
                suspicious_area += area
                circularity_list.append(circularity)
                solidity_list.append(solidity)
                extent_list.append(extent)

        glcm = graycomatrix(equalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]

        avg_circularity = np.mean(circularity_list) if circularity_list else 1.0
        avg_solidity = np.mean(solidity_list) if solidity_list else 1.0
        avg_extent = np.mean(extent_list) if extent_list else 1.0

        features.append([
            suspicious_count,
            suspicious_area,
            avg_circularity,
            avg_solidity,
            avg_extent,
            contrast,
            homogeneity,
            energy
        ])
        labels.append(y_bin[i])

    return np.array(features), np.array(labels)

# ========== Train and Evaluate ==========
y_train_bin = binarize(y_train)
y_test_bin = binarize(y_test)
y_valid_bin = binarize(y_valid)

features_train, labels_train = extract_features(X_train, y_train_bin)
features_valid, labels_valid = extract_features(X_valid, y_valid_bin)

svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(features_train, labels_train)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(features_train, labels_train)

svm_probs = svm_model.predict_proba(features_valid)
rf_probs = rf_model.predict_proba(features_valid)
avg_probs = (svm_probs + rf_probs) / 2
final_preds = np.argmax(avg_probs, axis=1)

accuracy = accuracy_score(labels_valid, final_preds)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
