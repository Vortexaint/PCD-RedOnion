import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from ekstraksi_bentuk import ekstraksi_fitur_bentuk
from ekstraksi_tekstur import ekstraksi_fitur_tekstur
from ekstraksi_warna import ekstraksi_fitur_warna
from ekstraksi_kombinasi import visualize_process

def visualize_feature_extraction(image_path, feature_type, output_dir):
    """
    Visualisasi proses ekstraksi fitur dan menyimpan hasilnya
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return
    
    # Buat figure
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f'Visualisasi Ekstraksi Fitur {feature_type.capitalize()}', fontsize=16)
    
    if feature_type == "bentuk":
        # Visualisasi ekstraksi fitur bentuk
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = rgb_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        plt.subplot(221)
        plt.imshow(rgb_image)
        plt.title('Citra Asli')
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(gray, cmap='gray')
        plt.title('Citra Grayscale')
        plt.axis('off')
        
        plt.subplot(223)
        plt.imshow(binary, cmap='gray')
        plt.title('Citra Biner (Otsu)')
        plt.axis('off')
        
        plt.subplot(224)
        plt.imshow(contour_img)
        plt.title('Deteksi Kontur')
        plt.axis('off')
        
    elif feature_type == "tekstur":
        # Visualisasi ekstraksi fitur tekstur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm_props = {}
        for prop in props:
            glcm_props[prop] = graycoprops(glcm, prop)[0, 0]
        
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Citra Asli')
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(gray, cmap='gray')
        plt.title('Citra Grayscale')
        plt.axis('off')
        
        plt.subplot(223)
        plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
        plt.title('GLCM Matrix (d=1, θ=0°)')
        plt.colorbar()
        plt.axis('off')
        
        ax = plt.subplot(224)
        bars = plt.bar(range(len(props)), list(glcm_props.values()))
        plt.xticks(range(len(props)), props, rotation=45)
        plt.title('GLCM Properties')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
    elif feature_type == "warna":
        # Visualisasi ekstraksi fitur warna
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        plt.subplot(221)
        plt.imshow(rgb_image)
        plt.title('Citra Asli (RGB)')
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
        plt.title('Ruang Warna HSV')
        plt.axis('off')
        
        # Plot histogram HSV
        plt.subplot(223)
        hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        plt.plot(hist)
        plt.title('Histogram HSV (Hue)')
        plt.xlabel('Hue Value')
        plt.ylabel('Frequency')
        
        # Plot RGB mean values
        plt.subplot(224)
        means = cv2.mean(image)[:3]
        plt.bar(['Blue', 'Green', 'Red'], means[:3], color=['blue', 'green', 'red'])
        plt.title('Nilai Rata-rata RGB')
        
    plt.tight_layout()
    
    # Simpan visualisasi
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{feature_type}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualisasi {feature_type} disimpan ke: {output_path}")

def visualize_all_features(dataset_path):
    """
    Melakukan visualisasi untuk semua fitur pada dataset
    """
    # Buat direktori output
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "citra", "hasil_ekstraksi")
    
    feature_types = ["bentuk", "tekstur", "warna"]
    for feature_type in feature_types:
        output_dir = os.path.join(output_base, f"visualisasi_{feature_type}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nMemproses visualisasi fitur {feature_type}...")
        
        # Proses setiap kategori sampah
        for category in ["kertas", "organik", "plastik"]:
            category_path = os.path.join(dataset_path, category)
            if os.path.exists(category_path):
                print(f"\nMemproses kategori {category}...")
                # Ambil beberapa sampel gambar dari setiap kategori
                images = os.listdir(category_path)[:3]  # Batasi 3 sampel per kategori
                for image_name in images:
                    image_path = os.path.join(category_path, image_name)
                    visualize_feature_extraction(image_path, feature_type, output_dir)

    # Visualisasi kombinasi fitur
    kombinasi_dir = os.path.join(output_base, "visualisasi_kombinasi")
    os.makedirs(kombinasi_dir, exist_ok=True)
    print("\nMemproses visualisasi kombinasi fitur...")
    
    # Ambil satu sampel dari setiap kategori untuk visualisasi kombinasi
    for category in ["kertas", "organik", "plastik"]:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            images = os.listdir(category_path)
            if images:
                image_path = os.path.join(category_path, images[0])
                fig = visualize_process(image_path)
                output_path = os.path.join(kombinasi_dir, f"kombinasi_{category}.png")
                fig.savefig(output_path)
                plt.close(fig)
                print(f"Visualisasi kombinasi untuk {category} disimpan ke: {output_path}")

if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "citra", "training")
        
        print("Memulai proses visualisasi fitur...")
        print(f"Base directory: {base_dir}")
        print(f"Dataset path: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise Exception(f"Dataset path does not exist: {dataset_path}")
            
        visualize_all_features(dataset_path)
        print("\nProses visualisasi selesai!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
