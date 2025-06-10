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
    model_bentuk_knn = klasifikasi_knn(bentuk_features_scaled, bentuk_labels, k=3)
    model_bentuk_svm = klasifikasi_svm(bentuk_features_scaled, bentuk_labels, kernel='linear')
    
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