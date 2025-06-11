import cv2
import numpy as np
import os
import csv
from validasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image

def ekstraksi_fitur_bentuk(image_path, show_conversion=False):
    """
    Fungsi untuk mengekstrak fitur bentuk dari citra.
    Parameters:
        image_path (str): Path ke file citra
        show_conversion (bool): Flag untuk menampilkan hasil konversi
    Returns:
        list: [luas, rasio_aspek, jumlah_sisi] atau None jika gagal
    """
    # Baca citra dari file
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None

    # Tahap 1: Konversi ke grayscale
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tahap 2: Konversi ke citra biner menggunakan metode Otsu
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tampilkan hasil konversi jika diminta
    if show_conversion:
        cv2.imshow('Citra Asli', image)
        cv2.imshow('Citra Grayscale', abu)
        cv2.imshow('Citra Biner', biner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Tahap 3: Deteksi kontur dari citra biner
    kontur, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tahap 4: Ekstraksi fitur dari kontur terbesar
    for cnt in kontur:
        # Hitung luas area kontur
        luas = cv2.contourArea(cnt)
        if luas < 100:  # Skip kontur yang terlalu kecil
            continue
            
        # Hitung keliling kontur (perimeter)
        keliling = cv2.arcLength(cnt, True)
        
        # Dapatkan bounding box dan hitung rasio aspek
        x, y, w, h = cv2.boundingRect(cnt)
        rasio_aspek = w / h
        
        # Aproksimasi bentuk poligon dan hitung jumlah sisi
        # Parameter 0.04 * keliling menentukan akurasi aproksimasi
        approx = cv2.approxPolyDP(cnt, 0.04 * keliling, True)
        sisi = len(approx)
        
        # Return fitur bentuk [luas, rasio aspek, jumlah sisi]
        return [luas, rasio_aspek, sisi]
    return None

def proses_folder_dataset_bentuk(folder_dataset, output_csv):
    """
    Memproses seluruh dataset dan menyimpan hasil ekstraksi fitur ke CSV
    Parameters:
        folder_dataset (str): Path ke folder dataset
        output_csv (str): Path untuk file output CSV
    """
    data = []
    label_list = []
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_bentuk(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)
    # Simpan hasil ekstraksi ke file CSV
    with open(output_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for fitur, label in zip(data, label_list):
            writer.writerow(np.append(fitur, label))
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_csv}")

def load_dataset_from_csv(csv_path):
    """
    Membaca dataset fitur dan label dari file CSV
    Parameters:
        csv_path (str): Path ke file CSV
    Returns:
        tuple: (features, labels) dalam format numpy array
    """
    features = []
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Konversi string ke float untuk fitur numerik
            features.append([float(x) for x in row[:-1]])
            labels.append(row[-1])
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Inisialisasi path untuk dataset dan output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "citra", "training")
    output_csv_path = os.path.join(base_dir, "..", "citra", "hasil_ekstraksi", "fitur_bentuk.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Tahap 1: Ekstraksi fitur dari seluruh dataset
    print("[INFO] Memulai ekstraksi fitur bentuk dari dataset...")
    proses_folder_dataset_bentuk(dataset_path, output_csv_path)

    # Tahap 2: Load dataset untuk training
    print("\n[INFO] Loading dataset...")
    features_training, labels_training = load_dataset_from_csv(output_csv_path)

    # Tahap 3: Training model klasifikasi
    print("\n[INFO] Training KNN dan SVM...")
    # Training model KNN dengan k=3
    model_knn = klasifikasi_knn(features_training, labels_training, k=3)
    # Training model SVM dengan kernel linear
    model_svm = klasifikasi_svm(features_training, labels_training, kernel='linear')

    # Tahap 4: Evaluasi model pada dataset
    print("\n[INFO] Mendeteksi seluruh dataset...")
    print("-" * 50)
    
    # Iterasi untuk setiap kelas sampah
    for label in ["kertas", "organik", "plastik"]:
        folder_path = os.path.join(dataset_path, label)
        print(f"\n[INFO] Memproses folder {label}...")
        print("-" * 40)
        
        # Evaluasi setiap gambar dalam folder
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                # Ekstrak fitur dari gambar
                image_path = os.path.join(folder_path, image_name)
                fitur = ekstraksi_fitur_bentuk(image_path)
                
                if fitur is not None:
                    # Lakukan prediksi menggunakan kedua model
                    hasil_prediksi_knn = prediksi_single_image(model_knn, fitur)
                    hasil_prediksi_svm = prediksi_single_image(model_svm, fitur)
                    
                    # Tampilkan hasil prediksi
                    print(f"Gambar: {image_name}")
                    print(f"[KNN] Prediksi = {hasil_prediksi_knn}")
                    print(f"[SVM] Prediksi = {hasil_prediksi_svm}")
                    print(f"Label sebenarnya = {label}")
                    print("-" * 40)
                    
    # Visualisasi hasil konversi untuk satu contoh gambar
    contoh_gambar = None
    for label in ["kertas", "organik", "plastik"]:
        folder_path = os.path.join(dataset_path, label)
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                contoh_gambar = os.path.join(folder_path, image_name)
                break
        if contoh_gambar:
            break
            
    # Tampilkan visualisasi tahapan konversi
    if contoh_gambar:
        print(f"\n[INFO] Menampilkan hasil konversi citra untuk: {os.path.basename(contoh_gambar)}")
        ekstraksi_fitur_bentuk(contoh_gambar, show_conversion=True)