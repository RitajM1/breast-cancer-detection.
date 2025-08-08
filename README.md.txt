# Breast Cancer Detection Using Image Processing

This project is part of the ARTI407 – Image Processing course at Imam Abdulrahman Bin Faisal University.

## 🔬 Project Summary

We developed a hybrid image processing and machine learning system to detect breast cancer in mammogram images using the MIAS dataset. The system preprocesses images, extracts shape and texture features, and classifies them using an ensemble of SVM and Random Forest classifiers.

- **Accuracy:** 85.11%
- **Dataset:** MIAS Mammography ROIs
- **Models Used:** SVM, Random Forest
- **Tools:** Python, OpenCV, scikit-learn, skimage

## ⚙️ Pipeline

1. Grayscale Conversion
2. Gaussian Blurring
3. Histogram Equalization
4. Otsu’s Thresholding
5. Morphological Closing
6. Feature Extraction (GLCM + Shape)
7. Classification (SVM + RF)

## 📁 Dataset

We used the publicly available **MIAS Mammography ROIs** dataset:
[Kaggle Link](https://www.kaggle.com/datasets/annkristinbalve/mias-mammography-rois)

Due to size restrictions, dataset files are not included in this repo.

## 🧪 Results

| Metric     | Value   |
|------------|---------|
| Accuracy   | 85.11%  |
| Precision  | 73.68%  |
| Recall     | 87.50%  |
| F1 Score   | 80.00%  |

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
