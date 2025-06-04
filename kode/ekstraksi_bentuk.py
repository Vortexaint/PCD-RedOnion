import cv2
import numpy as np
import os
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_svm(features, labels):
    """
    Melatih model SVM dengan fitur yang sudah dinormalisasi
    """
    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(features)
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(fitur_scaled, labels)
    return svm_model, scaler

def predict_image_svm(image_path, svm_model, scaler):
    """
    Prediksi gambar menggunakan model SVM
    """
    fitur = ekstraksi_fitur_bentuk(image_path)
    if fitur is not None:
        fitur_scaled = scaler.transform([fitur])
        label_prediksi = svm_model.predict(fitur_scaled)[0]
        confidence = np.max(svm_model.predict_proba(fitur_scaled)) * 100
        return label_prediksi, confidence
    return None, None

def ekstraksi_fitur_bentuk(image_path):
    """
    Ekstraksi fitur bentuk menggunakan kontur dan properti geometri
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None
        
    # Resize gambar
    image = cv2.resize(image, (200, 200))
    
    # Konversi ke grayscale dan threshold
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Temukan kontur
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ambil kontur terbesar
    if len(kontur) == 0:
        print(f"Warning: Tidak ada kontur ditemukan di {image_path}")
        return None
        
    cnt = max(kontur, key=cv2.contourArea)
    luas = cv2.contourArea(cnt)
    
    # Filter kontur kecil
    if luas < 100:
        print(f"Warning: Kontur terlalu kecil di {image_path}")
        return None
        
    # Ekstrak fitur bentuk
    keliling = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    rasio_aspek = float(w) / h if h != 0 else 0
    circularity = (4 * np.pi * luas) / (keliling * keliling) if keliling != 0 else 0
    approx = cv2.approxPolyDP(cnt, 0.04 * keliling, True)
    sisi = len(approx)
    
    # Return fitur bentuk [luas, rasio_aspek, sisi, circularity]
    return [luas, rasio_aspek, sisi, circularity]

def proses_folder_dataset(folder_dataset, output_file):
    """
    Proses semua gambar dalam folder dataset
    """
    data = []
    label_list = []
    
    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue

        print(f"[INFO] Memproses folder {label}...")
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_bentuk(path_file)
            
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)

    # Simpan hasil ke CSV
    with open(output_file, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        header = ['luas', 'rasio_aspek', 'sisi', 'circularity', 'label']
        writer.writerow(header)
        for fitur, label in zip(data, label_list):
            writer.writerow(np.append(fitur, label))
    
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_file}")
    return np.array(data), np.array(label_list)

def load_dataset_from_csv(csv_path):
    """
    Load dataset dari file CSV
    """
    features = []
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            features.append([float(x) for x in row[:-1]])
            labels.append(row[-1])
    return np.array(features), np.array(labels)

def hitung_jarak(fitur1, fitur2):
    """
    Menghitung jarak Euclidean antara dua fitur
    """
    return np.sqrt(np.sum((np.array(fitur1) - np.array(fitur2)) ** 2))

def knn_predict(fitur_test, fitur_training, label_training, k=3):
    """
    Klasifikasi menggunakan k-Nearest Neighbors dengan normalisasi dan pembobotan jarak
    """
    # Konversi ke numpy array jika belum
    fitur_training = np.array(fitur_training)
    fitur_test = np.array(fitur_test)
    
    # Normalisasi fitur
    mean = np.mean(fitur_training, axis=0)
    std = np.std(fitur_training, axis=0)
    fitur_training_norm = (fitur_training - mean) / (std + 1e-10)
    fitur_test_norm = (fitur_test - mean) / (std + 1e-10)
    
    # Hitung jarak ke semua data training
    jarak = []
    for fitur_train, label in zip(fitur_training_norm, label_training):
        d = hitung_jarak(fitur_test_norm, fitur_train)
        jarak.append((d, label))
    
    # Urutkan berdasarkan jarak
    jarak.sort(key=lambda x: x[0])
    
    # Ambil k tetangga terdekat
    k_terdekat = jarak[:k]
    
    # Voting mayoritas dengan pembobotan berdasarkan jarak
    label_count = {}
    total_weights = {}
    
    for d, label in k_terdekat:
        weight = 1.0 / (d + 1e-10)  # Inverse distance weighting
        if label not in label_count:
            label_count[label] = weight
            total_weights[label] = weight
        else:
            label_count[label] += weight
            total_weights[label] += weight
    
    # Hitung confidence untuk setiap kelas
    total_weight = sum(total_weights.values())
    confidences = {label: count/total_weight for label, count in label_count.items()}
    
    # Return label dengan vote terbanyak dan confidence
    label_prediksi = max(confidences.items(), key=lambda x: x[1])[0]
    confidence = confidences[label_prediksi] * 100
    
    return label_prediksi, confidence

def predict_image(image_path, features_training, labels_training):
    """
    Prediksi kelas untuk gambar baru
    """
    fitur = ekstraksi_fitur_bentuk(image_path)
    if fitur is not None:
        label_prediksi, confidence = knn_predict(fitur, features_training, labels_training, k=3)
        return label_prediksi, confidence
    return None, None

def visualisasi_hasil(image_path, knn_result, svm_result, output_path):
    """
    Membuat visualisasi hasil klasifikasi dari KNN dan SVM dan menyimpannya sebagai PNG
    """
    # Unpack hasil
    knn_label, knn_confidence = knn_result
    svm_label, svm_confidence = svm_result
    
    # Baca gambar
    image = cv2.imread(image_path)
    image = cv2.resize(image, (400, 400))
    
    # Proses gambar untuk deteksi bentuk
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Buat salinan gambar untuk visualisasi kontur
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    
    # Buat canvas untuk output
    output = np.zeros((600, 800, 3), dtype=np.uint8)
    output[0:400, 0:400] = image  # Original image
    output[0:400, 400:800] = contour_img  # Contour visualization
    
    # Tambahkan text hasil prediksi
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, f"Original", (10, 430), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Deteksi Kontur", (410, 430), font, 0.7, (255,255,255), 2)
    
    # Hasil KNN
    cv2.putText(output, "Hasil KNN:", (10, 470), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Prediksi: {knn_label}", (10, 500), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Confidence: {knn_confidence:.2f}%", (10, 530), font, 0.7, (255,255,255), 2)
    
    # Hasil SVM
    cv2.putText(output, "Hasil SVM:", (410, 470), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Prediksi: {svm_label}", (410, 500), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Confidence: {svm_confidence:.2f}%", (410, 530), font, 0.7, (255,255,255), 2)
    
    # Simpan gambar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"[INFO] Hasil visualisasi disimpan ke: {output_path}")

if __name__ == "__main__":
    # Gunakan absolute path untuk dataset
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(current_dir, "citra", "training")
    output_file = os.path.join(current_dir, "citra", "hasil_ekstraksi", "fitur_bentuk.csv")
    
    # Buat direktori hasil jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Proses dataset
    print("[INFO] Memulai ekstraksi fitur bentuk dari dataset...")
    fitur_training, label_training = proses_folder_dataset(dataset_path, output_file)
    
    # Train SVM model
    svm_model, scaler = train_svm(fitur_training, label_training)
    
    # Test beberapa gambar
    test_dir = os.path.join(current_dir, "citra", "testing")
    output_dir = os.path.join(current_dir, "citra", "hasil_ekstraksi", "visualisasi_bentuk")
    print("\n[INFO] Hasil Klasifikasi:")
    print("-" * 50)
    
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            test_image = os.path.join(label_dir, file)
            
            # Get predictions from both models
            knn_result = predict_image(test_image, fitur_training, label_training)
            svm_result = predict_image_svm(test_image, svm_model, scaler)
            
            if knn_result[0] and svm_result[0]:
                # Simpan visualisasi
                output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_hasil.png")
                visualisasi_hasil(test_image, knn_result, svm_result, output_path)
                
                print(f"[HASIL] {os.path.basename(test_image)}")
                print(f"[KNN] Prediksi: {knn_result[0]}, Confidence: {knn_result[1]:.2f}%")
                print(f"[SVM] Prediksi: {svm_result[0]}, Confidence: {svm_result[1]:.2f}%")
                print(f"[INFO] Label sebenarnya: {label}")
                print("-" * 50)
