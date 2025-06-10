import cv2
import numpy as np
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from validasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image
from skimage.feature import graycomatrix, graycoprops
from ekstraksi_kombinasi import load_dataset_split, load_and_combine_features, visualize_process

# --- Ekstraksi Fitur Bentuk ---
def ekstraksi_fitur_bentuk(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in kontur:
        luas = cv2.contourArea(cnt)
        if luas < 100:
            continue
        keliling = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        rasio_aspek = w / h
        approx = cv2.approxPolyDP(cnt, 0.04 * keliling, True)
        sisi = len(approx)
        return [luas, rasio_aspek, sisi]
    return None

# --- Ekstraksi Fitur Tekstur ---
def ekstraksi_fitur_tekstur(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    fitur = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in properties:
        fitur.extend(graycoprops(glcm, prop).flatten())
    return np.array(fitur)

# --- Ekstraksi Fitur Warna ---
def ekstraksi_fitur_warna(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (200, 200))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    mean_rgb = cv2.mean(image)[:3]
    fitur = np.hstack((hist, mean_rgb))
    return fitur

# --- Load Dataset ---
def load_dataset(folder_dataset, ekstraksi_fitur_func):
    data = []
    label_list = []
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_func(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)
    return np.array(data), np.array(label_list)

# --- Main ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    citra_path = os.path.join(base_dir, "..", "citra")
    labels = ["kertas", "organik", "plastik"]

    # Load combined features from training and testing sets
    print("[INFO] Loading combined features from designated folders...")
    X_train, X_test, y_train, y_test = load_dataset_split(citra_path)

    print("\n[INFO] Training models with combined features...")
    
    # Train and evaluate KNN
    print("\n=== KNN Model (Combined Features) ===")
    model_knn = klasifikasi_knn(X_train, y_train, k=3)
    y_pred_knn = model_knn.predict(X_test)
    print("\nKNN Test Set Performance:")
    print(classification_report(y_test, y_pred_knn, target_names=labels))
    
    # Train and evaluate SVM
    print("\n=== SVM Model (Combined Features) ===")
    model_svm = klasifikasi_svm(X_train, y_train, kernel='rbf')
    y_pred_svm = model_svm.predict(X_test)
    print("\nSVM Test Set Performance:")
    print(classification_report(y_test, y_pred_svm, target_names=labels))

    # Create confusion matrix plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot KNN confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred_knn, labels=labels),
        display_labels=labels
    ).plot(ax=ax1, cmap='Blues')
    ax1.set_title('KNN Confusion Matrix (Combined Features)')
    
    # Plot SVM confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred_svm, labels=labels),
        display_labels=labels
    ).plot(ax=ax2, cmap='Blues')
    ax2.set_title('SVM Confusion Matrix (Combined Features)')
    
    plt.tight_layout()
    
    # Save confusion matrix plots
    output_dir = os.path.join(citra_path, "hasil_ekstraksi", "visualisasi_kombinasi")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()

    # Visualize process for one sample from each class
    for class_name in labels:
        test_folder = os.path.join(citra_path, "testing", class_name)
        if os.path.exists(test_folder):
            # Get first image from test folder
            test_images = os.listdir(test_folder)
            if test_images:
                test_image = os.path.join(test_folder, test_images[0])
                # Get prediction from both models
                features = load_and_combine_features(test_image)
                if features is not None:
                    features = features.reshape(1, -1)
                    features_scaled = StandardScaler().fit_transform(features)
                    knn_pred = model_knn.predict(features_scaled)[0]
                    svm_pred = model_svm.predict(features_scaled)[0]
                    pred_text = f"KNN: {knn_pred}\nSVM: {svm_pred}"
                    
                    # Create visualization
                    fig = visualize_process(test_image, pred_text)
                    plt.savefig(os.path.join(output_dir, f"process_{class_name}.png"))
                    plt.close()

    print("\n[INFO] Results saved to:", output_dir)