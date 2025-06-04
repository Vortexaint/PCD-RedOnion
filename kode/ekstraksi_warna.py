import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_svm(features, labels):
    """
    Melatih model SVM dengan normalisasi fitur
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
    fitur = ekstraksi_fitur_warna(image_path)
    if fitur is not None:
        fitur_scaled = scaler.transform([fitur])
        label_prediksi = svm_model.predict(fitur_scaled)[0]
        confidence = np.max(svm_model.predict_proba(fitur_scaled)) * 100
        return label_prediksi, confidence
    return None, None

def ekstraksi_fitur_warna(image_path):
    """
    Ekstraksi fitur warna menggunakan histogram warna HSV dan statistik
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None

    # Resize gambar
    image = cv2.resize(image, (200, 200))
    
    # Konversi ke HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Hitung histogram untuk setiap channel
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    
    # Normalisasi histogram
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # Hitung statistik untuk setiap channel
    stats_h = [np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])]
    stats_s = [np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])]
    stats_v = [np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])]
    
    # Gabungkan semua fitur
    fitur = np.concatenate([hist_h, hist_s, hist_v, stats_h, stats_s, stats_v])
    
    return fitur

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
            fitur = ekstraksi_fitur_warna(path_file)
            
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)

    # Simpan hasil ke file numpy
    np.savez(output_file, 
             fitur=np.array(data), 
             label=np.array(label_list))
    
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_file}")
    return np.array(data), np.array(label_list)

def hitung_jarak(fitur1, fitur2):
    """
    Menghitung jarak Euclidean antara dua fitur
    """
    return np.sqrt(np.sum((fitur1 - fitur2) ** 2))

def klasifikasi_gambar(fitur_test, fitur_training, label_training):
    """
    Klasifikasi menggunakan metode nearest neighbor
    """
    jarak_min = float('inf')
    label_prediksi = None
    
    for fitur_train, label in zip(fitur_training, label_training):
        jarak = hitung_jarak(fitur_test, fitur_train)
        if jarak < jarak_min:
            jarak_min = jarak
            label_prediksi = label
            
    return label_prediksi, jarak_min

def knn_predict(fitur_test, fitur_training, label_training, k=3):
    """
    Klasifikasi menggunakan k-Nearest Neighbors dengan normalisasi dan pembobotan jarak
    """
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
    confidences = {label: count/sum(total_weights.values()) for label, count in label_count.items()}
    
    # Return label dengan vote terbanyak dan confidence
    label_prediksi = max(confidences.items(), key=lambda x: x[1])[0]
    confidence = confidences[label_prediksi] * 100
    
    return label_prediksi, confidence

def predict_image(image_path, fitur_training, label_training):
    """
    Prediksi kelas untuk gambar baru
    """
    fitur = ekstraksi_fitur_warna(image_path)
    if fitur is not None:
        label_prediksi, confidence = knn_predict(fitur, fitur_training, label_training, k=3)
        return label_prediksi, confidence
    return None, None

def visualisasi_hasil(image_path, label_prediksi, confidence, output_path):
    """
    Membuat visualisasi hasil klasifikasi dan menyimpannya sebagai PNG
    """
    # Baca gambar
    image = cv2.imread(image_path)
    image = cv2.resize(image, (400, 400))
    
    # Buat canvas untuk output
    output = np.zeros((500, 400, 3), dtype=np.uint8)
    output[0:400, :] = image
    
    # Tambahkan text hasil prediksi
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, f"Prediksi: {label_prediksi}", (10, 430), font, 0.7, (255,255,255), 2)
    cv2.putText(output, f"Confidence: {confidence:.2f}%", (10, 460), font, 0.7, (255,255,255), 2)
    
    # Simpan gambar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"[INFO] Hasil visualisasi disimpan ke: {output_path}")

if __name__ == "__main__":
    # Gunakan absolute path untuk dataset
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(current_dir, "citra", "training")
    output_file = os.path.join(current_dir, "citra", "hasil_ekstraksi", "fitur_warna.npz")
    
    # Buat direktori hasil jika belum ada
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Proses dataset
    print("[INFO] Memulai ekstraksi fitur warna dari dataset...")
    fitur_training, label_training = proses_folder_dataset(dataset_path, output_file)
    
    # Test beberapa gambar
    test_dir = os.path.join(current_dir, "citra", "testing")
    output_dir = os.path.join(current_dir, "citra", "hasil_ekstraksi", "visualisasi_warna")
    print("\n[INFO] Hasil Klasifikasi:")
    print("-" * 40)
    
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            test_image = os.path.join(label_dir, file)
            label_prediksi, confidence = predict_image(test_image, fitur_training, label_training)
            if label_prediksi:
                # Simpan visualisasi
                output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_hasil.png")
                visualisasi_hasil(test_image, label_prediksi, confidence, output_path)
                
                print(f"[HASIL] {os.path.basename(test_image)} = {label_prediksi}")
                print(f"[INFO] Confidence: {confidence:.2f}%")
                print(f"[INFO] Label sebenarnya: {label}")
                print("-" * 40)
