import cv2
import numpy as np
import os
import csv

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

if __name__ == "__main__":
    # Gunakan absolute path untuk dataset
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(current_dir, "citra", "training")
    output_file = os.path.join(current_dir, "citra", "hasil_ekstraksi", "fitur_bentuk.csv")
    
    # Buat direktori hasil jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Proses dataset
    print("[INFO] Memulai ekstraksi fitur bentuk dari dataset...")
    proses_folder_dataset(dataset_path, output_file)
    
    # Load dataset
    print("\n[INFO] Loading dataset...")
    features_training, labels_training = load_dataset_from_csv(output_file)
    
    # Test beberapa gambar
    test_dir = os.path.join(current_dir, "citra", "testing")
    print("\n[INFO] Hasil Klasifikasi:")
    print("-" * 40)
    
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            test_image = os.path.join(label_dir, file)
            label_prediksi, confidence = predict_image(test_image, features_training, labels_training)
            if label_prediksi:
                print(f"[HASIL] {os.path.basename(test_image)} = {label_prediksi}")
                print(f"[INFO] Confidence: {confidence:.2f}%")
                print(f"[INFO] Label sebenarnya: {label}")
                print("-" * 40)