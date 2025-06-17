import cv2
import numpy as np
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from validasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image
from skimage.feature import graycomatrix, graycoprops
from ekstraksi_kombinasi import load_dataset_split, load_and_combine_features, visualize_process

# --- Optimasi Parameter SVM ---
def optimize_svm_params(X, y):
    """
    Mencari parameter optimal untuk model SVM menggunakan Grid Search
    
    Args:
        X: Fitur yang telah di-scale
        y: Label
    
    Returns:
        dict: Parameter terbaik untuk SVM
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print(f"\n[INFO] Best SVM parameters: {grid_search.best_params_}")
    print(f"[INFO] Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    
    return grid_search.best_params_

# --- Ekstraksi Fitur Bentuk ---
def ekstraksi_fitur_bentuk(image_path):
    """
    Mengekstrak fitur geometri dari citra dengan preprocessing dan fitur tambahan.
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        list: [luas, rasio_aspek, jumlah_sisi, kebulatan, eksentrisitas, solidity, extent, convexity] atau None jika gagal
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # Preprocessing yang lebih baik
    # 1. Resize untuk konsistensi
    image = cv2.resize(image, (300, 300))
    
    # 2. Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 3. Konversi ke grayscale dengan pembobotan yang lebih baik
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. Histogram equalization untuk meningkatkan kontras
    gray = cv2.equalizeHist(gray)
    
    # 5. Bilateral filtering untuk edge preservation
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 6. Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 7. Morphological operations
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Temukan kontur
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Ambil kontur terbesar
    cnt = max(contours, key=cv2.contourArea)
    
    # Ekstraksi fitur
    # 1. Luas
    luas = cv2.contourArea(cnt)
    if luas < 100:  # Filter objek terlalu kecil
        return None
        
    # 2. Keliling
    keliling = cv2.arcLength(cnt, True)
    
    # 3. Bounding Rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    rasio_aspek = float(w)/h
    
    # 4. Approx Poly
    epsilon = 0.04 * keliling
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    sisi = len(approx)
    
    # 5. Kebulatan (Circularity)
    kebulatan = (4 * np.pi * luas) / (keliling * keliling)
    
    # 6. Eksentrisitas
    if len(cnt) >= 5:  # Minimal 5 points needed for ellipse fitting
        try:
            (_, _), (MA, ma), angle = cv2.fitEllipse(cnt)
            # Pastikan MA (major axis) tidak nol dan ma/MA tidak lebih dari 1
            if MA > 0:
                ratio = min(ma/MA, 1.0)  # membatasi rasio ke maksimal 1
                eksentrisitas = np.sqrt(1 - ratio**2)
            else:
                eksentrisitas = 0
        except:
            eksentrisitas = 0
    else:
        eksentrisitas = 0
    
    # 7. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(luas)/hull_area if hull_area > 0 else 0
    
    # 8. Extent
    rect_area = w * h
    extent = float(luas)/rect_area if rect_area > 0 else 0
    
    # 9. Convexity
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = keliling / hull_perimeter if hull_perimeter > 0 else 0
    
    return [luas, rasio_aspek, sisi, kebulatan, eksentrisitas, solidity, extent, convexity]

# --- Ekstraksi Fitur Tekstur ---
def ekstraksi_fitur_tekstur(image_path):
    """
    Mengekstrak fitur tekstur dari citra menggunakan Gray Level Co-occurrence Matrix (GLCM).
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        np.array: Fitur tekstur yang diekstrak atau None jika gagal
    """
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
    """
    Mengekstrak fitur warna dari citra.
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        np.array: Fitur warna yang diekstrak atau None jika gagal
    """
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
    """
    Memuat dataset dari folder dan mengekstrak fitur menggunakan fungsi yang diberikan.
    
    Args:
        folder_dataset (str): Path ke folder dataset
        ekstraksi_fitur_func (function): Fungsi untuk mengekstrak fitur dari citra
        
    Returns:
        tuple: (data, label_list) - data fitur dan daftar label
    """
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

# --- Visualisasi Tahapan Preprocessing Citra ---
def visualize_preprocessing(image_path):
    """Visualisasi tahapan preprocessing citra"""
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return
    
    # Konversi BGR ke RGB untuk matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Proses ekstraksi fitur bentuk
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normal threshold
    _, binary_normal = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Otsu threshold
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Kontur dari Otsu threshold
    contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar kontur
    contour_img = rgb_image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Proses ekstraksi fitur warna
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Proses ekstraksi fitur tekstur
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    
    # Buat figure dengan 3x3 subplot
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Visualisasi Tahapan Preprocessing Citra', fontsize=16)
    
    # Original Image
    ax1 = plt.subplot(331)
    ax1.imshow(rgb_image)
    ax1.set_title('Citra Asli')
    ax1.axis('off')
    
    # Grayscale
    ax2 = plt.subplot(332)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Citra Grayscale')
    ax2.axis('off')
    
    # Normal Binary
    ax3 = plt.subplot(333)
    ax3.imshow(binary_normal, cmap='gray')
    ax3.set_title('Threshold Normal (127)')
    ax3.axis('off')
    
    # Otsu Binary
    ax4 = plt.subplot(334)
    ax4.imshow(binary_otsu, cmap='gray')
    ax4.set_title('Threshold Otsu')
    ax4.axis('off')
    
    # Contour Detection
    ax5 = plt.subplot(335)
    ax5.imshow(contour_img)
    ax5.set_title('Deteksi Kontur (Otsu)')
    ax5.axis('off')
    
    # HSV Color Space
    ax6 = plt.subplot(336)
    ax6.imshow(hsv_display)
    ax6.set_title('Ruang Warna HSV')
    ax6.axis('off')
    
    # HSV Histogram
    ax7 = plt.subplot(337)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    ax7.plot(hist_h)
    ax7.set_title('Histogram HSV (Hue)')
    
    # RGB Mean Values
    ax8 = plt.subplot(338)
    means = cv2.mean(image)[:3]
    ax8.bar(['B', 'G', 'R'], means[:3], color=['blue', 'green', 'red'])
    ax8.set_title('Nilai Rata-rata RGB')
    
    # Shape Features
    ax9 = plt.subplot(339)
    shape_info = ""
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w/h
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            sides = len(approx)
            shape_info = f"Luas: {area:.0f}\nRasio: {aspect_ratio:.2f}\nSisi: {sides}"
            break
    
    ax9.text(0.5, 0.5, shape_info, ha='center', va='center')
    ax9.set_title('Fitur Bentuk')
    ax9.axis('off')
    
    plt.tight_layout()
    return fig

# --- Voting dan Display Hasil ---
def get_voting_result(predictions):
    """Menghitung hasil voting dari semua classifier"""
    from collections import Counter
    vote_counter = Counter(predictions)
    return vote_counter.most_common(1)[0][0]  # Return label dengan vote terbanyak

def display_classification_results(results_dict):
    """Menampilkan hasil klasifikasi dalam format tabel"""
    # Header
    header = "| {:<15} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |".format(
        "Gambar", "Label", "Bentuk-KNN", "Bentuk-SVM", "Tekstur-KNN", "Tekstur-SVM", "Warna-KNN", "Warna-SVM", "Voting"
    )
    separator = "-" * len(header)
    
    print("\nHasil Klasifikasi:")
    print(separator)
    print(header)
    print(separator)
    
    # Rows
    for img_name, result in results_dict.items():
        row = "| {:<15} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |".format(
            img_name[:15],
            result['true_label'],
            result['predictions'].get('Bentuk-KNN', '-'),
            result['predictions'].get('Bentuk-SVM', '-'),
            result['predictions'].get('Tekstur-KNN', '-'),
            result['predictions'].get('Tekstur-SVM', '-'),
            result['predictions'].get('Warna-KNN', '-'),
            result['predictions'].get('Warna-SVM', '-'),
            result['voting']
        )
        print(row)
    print(separator)
    
    # Simpan hasil ke file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "citra", "hasil_ekstraksi")
    with open(os.path.join(output_dir, 'hasil_klasifikasi.txt'), 'w') as f:
        f.write("Hasil Klasifikasi:\n")
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for img_name, result in results_dict.items():
            row = "| {:<15} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |".format(
                img_name[:15],
                result['true_label'],
                result['predictions'].get('Bentuk-KNN', '-'),
                result['predictions'].get('Bentuk-SVM', '-'),
                result['predictions'].get('Tekstur-KNN', '-'),
                result['predictions'].get('Tekstur-SVM', '-'),
                result['predictions'].get('Warna-KNN', '-'),
                result['predictions'].get('Warna-SVM', '-'),
                result['voting']
            )
            f.write(row + "\n")
        f.write(separator)

def evaluate_single_image(image_path, models, scaler_bentuk, scaler_tekstur, scaler_warna):
    """Evaluasi satu gambar dengan semua model"""
    # Ekstrak fitur
    fitur_bentuk = ekstraksi_fitur_bentuk(image_path)
    fitur_tekstur = ekstraksi_fitur_tekstur(image_path)
    fitur_warna = ekstraksi_fitur_warna(image_path)
    
    predictions = {}
    
    # Prediksi menggunakan fitur bentuk
    if fitur_bentuk is not None:
        fitur_bentuk = scaler_bentuk.transform([fitur_bentuk])
        predictions['Bentuk-KNN'] = models['bentuk_knn'].predict(fitur_bentuk)[0]
        predictions['Bentuk-SVM'] = models['bentuk_svm'].predict(fitur_bentuk)[0]
    
    # Prediksi menggunakan fitur tekstur
    if fitur_tekstur is not None:
        fitur_tekstur = scaler_tekstur.transform([fitur_tekstur])
        predictions['Tekstur-KNN'] = models['tekstur_knn'].predict(fitur_tekstur)[0]
        predictions['Tekstur-SVM'] = models['tekstur_svm'].predict(fitur_tekstur)[0]
    
    # Prediksi menggunakan fitur warna
    if fitur_warna is not None:
        fitur_warna = scaler_warna.transform([fitur_warna])
        predictions['Warna-KNN'] = models['warna_knn'].predict(fitur_warna)[0]
        predictions['Warna-SVM'] = models['warna_svm'].predict(fitur_warna)[0]
    
    # Hitung voting
    if predictions:
        voting_result = get_voting_result(list(predictions.values()))
    else:
        voting_result = "Tidak dapat menentukan"
    
    return predictions, voting_result

# --- Training SVM dengan Parameter Optimal ---
def train_optimized_svm(X_train, y_train):
    """
    Melatih model SVM dengan parameter optimal
    
    Args:
        X_train: Fitur training yang telah di-scale
        y_train: Label training
    
    Returns:
        SVC: Model SVM yang telah dilatih
    """
    # Parameter grid untuk optimasi
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3, 4],
        'class_weight': ['balanced', None]
    }
    
    # Grid search dengan cross validation
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"\n[INFO] Parameter SVM terbaik: {grid_search.best_params_}")
    print(f"[INFO] Akurasi cross-validation terbaik: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

# --- Load Dataset dengan Augmentasi ---
def load_augmented_dataset(folder_dataset):
    """
    Memuat dataset dengan augmentasi data
    
    Args:
        folder_dataset: Path ke folder dataset
    
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
            
        for img_file in os.listdir(folder_label):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder_label, img_file)
            
            # Ekstrak fitur dari gambar asli
            features_orig = ekstraksi_fitur_bentuk(img_path)
            if features_orig is not None:
                features.append(features_orig)
                labels.append(label)
                
            # Load gambar untuk augmentasi
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Augmentasi data
            # 1. Flip horizontal
            img_flip = cv2.flip(image, 1)
            features_flip = ekstraksi_fitur_bentuk(img_path)
            if features_flip is not None:
                features.append(features_flip)
                labels.append(label)
            
            # 2. Rotasi 90 derajat
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
            img_rot = cv2.warpAffine(image, M, (cols, rows))
            features_rot = ekstraksi_fitur_bentuk(img_path)
            if features_rot is not None:
                features.append(features_rot)
                labels.append(label)
            
            # 3. Scaling
            img_scale = cv2.resize(image, None, fx=0.8, fy=0.8)
            features_scale = ekstraksi_fitur_bentuk(img_path)
            if features_scale is not None:
                features.append(features_scale)
                labels.append(label)
    
    return np.array(features), np.array(labels)

# --- Main ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    citra_path = os.path.join(base_dir, "..", "citra")
    labels = ["kertas", "organik", "plastik"]

    # Tampilkan proses preprocessing untuk satu contoh dari setiap kelas
    print("[INFO] Menampilkan visualisasi preprocessing...")
    for label in labels:
        test_folder = os.path.join(citra_path, "testing", label)
        if os.path.exists(test_folder):
            test_images = os.listdir(test_folder)
            if test_images:
                test_image = os.path.join(test_folder, test_images[0])
                fig = visualize_preprocessing(test_image)
                plt.savefig(os.path.join(citra_path, "hasil_ekstraksi", "visualisasi_kombinasi", f"preprocessing_{label}.png"))
                plt.close()

    # Train models untuk setiap fitur
    print("\n[INFO] Training models untuk setiap fitur...")
    
    # Load dan training dataset bentuk
    print("\n[INFO] Loading dan training dataset bentuk...")
    dataset_path = os.path.join(citra_path, "training")
    bentuk_features, bentuk_labels = load_dataset(dataset_path, ekstraksi_fitur_bentuk)
    scaler_bentuk = StandardScaler().fit(bentuk_features)
    bentuk_features_scaled = scaler_bentuk.transform(bentuk_features)
    # Train KNN model untuk fitur bentuk
    model_bentuk_knn = klasifikasi_knn(bentuk_features_scaled, bentuk_labels, k=3)
    
    # Optimize dan train SVM model untuk fitur bentuk
    print("\n[INFO] Optimizing SVM parameters for shape features...")
    best_params = optimize_svm_params(bentuk_features_scaled, bentuk_labels)
    model_bentuk_svm = SVC(**best_params, probability=True)
    model_bentuk_svm.fit(bentuk_features_scaled, bentuk_labels)
    
    # Load dan training dataset tekstur
    print("\n[INFO] Loading dan training dataset tekstur...")
    tekstur_features, tekstur_labels = load_dataset(dataset_path, ekstraksi_fitur_tekstur)
    scaler_tekstur = StandardScaler().fit(tekstur_features)
    tekstur_features_scaled = scaler_tekstur.transform(tekstur_features)
    model_tekstur_knn = klasifikasi_knn(tekstur_features_scaled, tekstur_labels, k=3)
    model_tekstur_svm = klasifikasi_svm(tekstur_features_scaled, tekstur_labels, kernel='linear')
    
    # Load dan training dataset warna
    print("\n[INFO] Loading dan training dataset warna...")
    warna_features, warna_labels = load_dataset(dataset_path, ekstraksi_fitur_warna)
    scaler_warna = StandardScaler().fit(warna_features)
    warna_features_scaled = scaler_warna.transform(warna_features)
    model_warna_knn = klasifikasi_knn(warna_features_scaled, warna_labels, k=3)
    model_warna_svm = klasifikasi_svm(warna_features_scaled, warna_labels, kernel='linear')
    
    # Kumpulkan semua model
    models = {
        'bentuk_knn': model_bentuk_knn,
        'bentuk_svm': model_bentuk_svm,
        'tekstur_knn': model_tekstur_knn,
        'tekstur_svm': model_tekstur_svm,
        'warna_knn': model_warna_knn,
        'warna_svm': model_warna_svm
    }
    
    # Evaluasi testing dataset
    print("\n[INFO] Evaluasi testing dataset...")
    all_results = {}
    
    for label in labels:
        test_folder = os.path.join(citra_path, "testing", label)
        if os.path.exists(test_folder):
            for image_name in os.listdir(test_folder):
                image_path = os.path.join(test_folder, image_name)
                predictions, voting = evaluate_single_image(
                    image_path, 
                    models,
                    scaler_bentuk,
                    scaler_tekstur,
                    scaler_warna
                )
                all_results[image_name] = {
                    'true_label': label,
                    'predictions': predictions,
                    'voting': voting
                }
    
    # Tampilkan hasil dalam format tabel
    display_classification_results(all_results)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(citra_path, "hasil_ekstraksi", "visualisasi_kombinasi")
    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] Results saved to:", output_dir)