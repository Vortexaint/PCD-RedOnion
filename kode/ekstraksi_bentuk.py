import cv2
import numpy as np
import os
import csv
from validasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image

def ekstraksi_fitur_bentuk(image_path, show_conversion=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None
    abu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show_conversion:
        cv2.imshow('Citra Asli', image)
        cv2.imshow('Citra Grayscale', abu)
        cv2.imshow('Citra Biner', biner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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

def proses_folder_dataset_bentuk(folder_dataset, output_csv):
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
    with open(output_csv, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for fitur, label in zip(data, label_list):
            writer.writerow(np.append(fitur, label))
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_csv}")

def load_dataset_from_csv(csv_path):
    features = []
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features.append([float(x) for x in row[:-1]])
            labels.append(row[-1])
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "citra", "training")  # Update path
    output_csv_path = os.path.join(base_dir, "..", "citra", "hasil_ekstraksi", "fitur_bentuk.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    print("[INFO] Memulai ekstraksi fitur bentuk dari dataset...")
    proses_folder_dataset_bentuk(dataset_path, output_csv_path)

    print("\n[INFO] Loading dataset...")
    features_training, labels_training = load_dataset_from_csv(output_csv_path)

    print("\n[INFO] Training KNN dan SVM...")
    model_knn = klasifikasi_knn(features_training, labels_training, k=3)
    model_svm = klasifikasi_svm(features_training, labels_training, kernel='linear')

    # Membaca semua gambar dari folder dataset
    print("\n[INFO] Mendeteksi seluruh dataset...")
    print("-" * 50)
    
    for label in ["kertas", "organik", "plastik"]:  # Update labels
        folder_path = os.path.join(dataset_path, label)
        print(f"\n[INFO] Memproses folder {label}...")
        print("-" * 40)
        
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                fitur = ekstraksi_fitur_bentuk(image_path)
                if fitur is not None:
                    hasil_prediksi_knn = prediksi_single_image(model_knn, fitur)
                    hasil_prediksi_svm = prediksi_single_image(model_svm, fitur)
                    print(f"Gambar: {image_name}")
                    print(f"[KNN] Prediksi = {hasil_prediksi_knn}")
                    print(f"[SVM] Prediksi = {hasil_prediksi_svm}")
                    print(f"Label sebenarnya = {label}")
                    print("-" * 40)
    # Tampilkan hasil konversi untuk satu gambar saja
    contoh_gambar = None
    for label in ["kertas", "organik", "plastik"]:  # Update labels
        folder_path = os.path.join(dataset_path, label)
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                contoh_gambar = os.path.join(folder_path, image_name)
                break
        if contoh_gambar:
            break
    if contoh_gambar:
        print(f"\n[INFO] Menampilkan hasil konversi citra untuk: {os.path.basename(contoh_gambar)}")
        ekstraksi_fitur_bentuk(contoh_gambar, show_conversion=True)