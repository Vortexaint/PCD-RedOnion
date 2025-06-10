import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
from validasi_model import klasifikasi_knn, klasifikasi_svm, prediksi_single_image

def ekstraksi_fitur_tekstur(image_path, show_conversion=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return None
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, biner = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show_conversion:
        cv2.imshow('Citra Asli', image)
        cv2.imshow('Citra Grayscale', gray)
        cv2.imshow('Citra Biner', biner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    fitur = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in properties:
        fitur.extend(graycoprops(glcm, prop).flatten())
    return np.array(fitur)

def proses_folder_dataset(folder_dataset, output_file):
    data = []
    label_list = []
    for label in os.listdir(folder_dataset):
        folder_label = os.path.join(folder_dataset, label)
        if not os.path.isdir(folder_label):
            continue
        print(f"[INFO] Memproses folder {label}...")
        for file in os.listdir(folder_label):
            path_file = os.path.join(folder_label, file)
            fitur = ekstraksi_fitur_tekstur(path_file)
            if fitur is not None:
                data.append(fitur)
                label_list.append(label)
    np.savez(output_file, fitur=np.array(data), label=np.array(label_list))
    print(f"[INFO] Ekstraksi selesai. Disimpan ke: {output_file}")
    return np.array(data), np.array(label_list)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "citra", "training")  # Update path
    output_file = os.path.join(current_dir, "..", "citra", "hasil_ekstraksi", "fitur_tekstur.npz")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("[INFO] Memulai ekstraksi fitur dari dataset...")
    fitur_training, label_training = proses_folder_dataset(dataset_path, output_file)

    print("\n[INFO] Training KNN dan SVM...")
    model_knn = klasifikasi_knn(fitur_training, label_training, k=3)
    model_svm = klasifikasi_svm(fitur_training, label_training, kernel='linear')

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
                fitur = ekstraksi_fitur_tekstur(image_path)
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
        ekstraksi_fitur_tekstur(contoh_gambar, show_conversion=True)