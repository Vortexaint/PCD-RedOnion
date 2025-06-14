import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from ekstraksi_bentuk import ekstraksi_fitur_bentuk
from ekstraksi_tekstur import ekstraksi_fitur_tekstur
from ekstraksi_warna import ekstraksi_fitur_warna
from visualisasi import visualize_process, visualize_texture_shape_combination

def visualize_feature_extraction(image_path, feature_type, output_dir):
    """
    Membuat dan menyimpan visualisasi proses ekstraksi fitur.
    
    Args:
        image_path (str): Path ke file citra input
        feature_type (str): Jenis fitur ('bentuk', 'warna', atau 'tekstur')
        output_dir (str): Direktori untuk menyimpan hasil visualisasi
        
    Returns:
        None: File visualisasi akan disimpan di output_dir
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
        
    elif feature_type == "kombinasi-tekstur-warna":
        # Panggil fungsi visualisasi kombinasi tekstur dan warna
        visualize_texture_color_combination(image_path, output_dir)
        return
    
    elif feature_type == "kombinasi-tekstur-bentuk":
        # Visualisasi ekstraksi fitur tekstur dan bentuk
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Hitung fitur GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # Plot citra asli
        plt.subplot(221)
        plt.imshow(rgb_image)
        plt.title('Citra Asli')
        plt.axis('off')
        
        # Plot citra grayscale
        plt.subplot(222)
        plt.imshow(gray, cmap='gray')
        plt.title('Citra Grayscale')
        plt.axis('off')
        
        # Plot matriks GLCM
        plt.subplot(223)
        plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
        plt.title('GLCM Matrix (d=1, θ=0°)')
        plt.colorbar()
        plt.axis('off')
        
        # Plot kontur
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = rgb_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        plt.subplot(224)
        plt.imshow(contour_img)
        plt.title('Deteksi Kontur')
        plt.axis('off')
        
    plt.tight_layout()
    
    # Simpan visualisasi
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{feature_type}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualisasi {feature_type} disimpan ke: {output_path}")

def visualize_all_features(dataset_path):
    """
    Melakukan visualisasi untuk semua fitur pada dataset
    
    Args:
        dataset_path (str): Path ke direktori dataset
        
    Returns:
        None: Proses visualisasi dilakukan untuk setiap gambar dalam dataset
    """
    # Buat direktori output
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(base_dir, "citra", "hasil_ekstraksi")
      # List of all feature types including texture-color combination
    feature_types = ["bentuk", "tekstur", "warna", "kombinasi-tekstur-warna"]
    
    for feature_type in feature_types:
        # Setup output directory
        if feature_type == "kombinasi-tekstur-warna":
            output_dir = os.path.join(output_base, "visualisasi_tekstur_warna")
        else:
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

    # Visualisasi kombinasi semua fitur
    kombinasi_dir = os.path.join(output_base, "visualisasi_kombinasi")
    os.makedirs(kombinasi_dir, exist_ok=True)
    print("\nMemproses visualisasi kombinasi semua fitur...")
    
    # Ambil satu sampel dari setiap kategori untuk visualisasi kombinasi
    for category in ["kertas", "organik", "plastik"]:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            images = os.listdir(category_path)
            if images:
                image_path = os.path.join(category_path, images[0])
                fig = visualize_process(image_path)
                output_path = os.path.join(kombinasi_dir, f"preprocessing_{category}.png")
                fig.savefig(output_path)
                plt.close(fig)
                print(f"Visualisasi preprocessing untuk {category} disimpan ke: {output_path}")

def visualize_texture_color_combination(image_path, output_dir):
    """
    Membuat dan menyimpan visualisasi kombinasi fitur tekstur dan warna.
    
    Args:
        image_path (str): Path ke file citra input
        output_dir (str): Direktori untuk menyimpan hasil visualisasi
        
    Returns:
        None: File visualisasi akan disimpan di output_dir
    """
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: tidak dapat membaca {image_path}")
        return
    
    # Konversi ke RGB untuk display
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Konversi ke HSV untuk analisis warna
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Buat figure dengan ukuran lebih besar
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('Visualisasi Kombinasi Fitur Tekstur dan Warna', fontsize=16)
    
    # 1. Tampilkan citra asli
    plt.subplot(331)
    plt.imshow(rgb_image)
    plt.title('Citra Asli (RGB)')
    plt.axis('off')
    
    # 2. Tampilkan citra HSV
    plt.subplot(332)
    plt.imshow(hsv_image)
    plt.title('Citra HSV')
    plt.axis('off')
    
    # 3. Tampilkan histogram HSV
    plt.subplot(333)
    colors = ('h', 's', 'v')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([hsv_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=['r', 'g', 'b'][i], label=col.upper())
    plt.title('Histogram HSV')
    plt.legend()
    plt.xlabel('Nilai Bin')
    plt.ylabel('Frekuensi')
    
    # 4. Tampilkan citra grayscale untuk tekstur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(334)
    plt.imshow(gray, cmap='gray')
    plt.title('Citra Grayscale')
    plt.axis('off')
    
    # 5. Tampilkan GLCM untuk berbagai sudut
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    # 6. Tampilkan matriks GLCM untuk setiap sudut
    titles = ['GLCM 0°', 'GLCM 45°', 'GLCM 90°', 'GLCM 135°']
    for i in range(4):
        plt.subplot(335 + i)
        plt.imshow(glcm[:, :, 0, i], cmap='viridis')
        plt.title(titles[i])
        plt.colorbar()
        plt.axis('off')
    
    # 7. Tampilkan properti GLCM
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_props = {}
    for prop in props:
        glcm_props[prop] = [graycoprops(glcm, prop)[0, angle_idx] for angle_idx in range(4)]
    
    ax = plt.subplot(339)
    x = np.arange(len(props))
    width = 0.15
    for i in range(4):
        values = [glcm_props[prop][i] for prop in props]
        plt.bar(x + i*width, values, width, label=f'{titles[i]}')
    
    plt.xlabel('Properti GLCM')
    plt.ylabel('Nilai')
    plt.title('Properti GLCM untuk Setiap Sudut')
    plt.xticks(x + width*1.5, props, rotation=45)
    plt.legend()
    
    # Simpan visualisasi
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_kombinasi.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualisasi disimpan di: {output_path}")

if __name__ == "__main__":
    try:
        import argparse
        
        # Setup argument parser
        parser = argparse.ArgumentParser(description='Visualisasi Fitur Citra Sampah')
        parser.add_argument('--dataset', type=str, 
                          help='Path ke direktori dataset (default: direktori training)',
                          default=None)
        parser.add_argument('--feature-type', type=str, 
                          choices=['bentuk', 'tekstur', 'warna', 'kombinasi-tekstur-warna', 'all'],
                          help='Jenis fitur yang akan divisualisasikan',
                          default='all')
        parser.add_argument('--sample', type=int,
                          help='Jumlah sampel per kategori (default: 3)',
                          default=3)
        parser.add_argument('--category', type=str,
                          choices=['kertas', 'organik', 'plastik', 'all'],
                          help='Kategori sampah yang akan divisualisasikan',
                          default='all')
        
        args = parser.parse_args()
        
        # Setup paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = args.dataset or os.path.join(base_dir, "citra", "training")
        
        print("\n=== Visualisasi Fitur Citra Sampah ===")
        print(f"Base directory: {base_dir}")
        print(f"Dataset path: {dataset_path}")
        print(f"Jenis fitur: {args.feature_type}")
        print(f"Jumlah sampel: {args.sample}")
        print(f"Kategori: {args.category}")
        print("="*40 + "\n")
        
        if not os.path.exists(dataset_path):
            raise Exception(f"Dataset path tidak ditemukan: {dataset_path}")
        
        # Setup output directory
        output_base = os.path.join(base_dir, "citra", "hasil_ekstraksi")
        
        if args.feature_type == 'all':
            visualize_all_features(dataset_path)
        else:
            # Process specific feature type and category
            output_dir = os.path.join(output_base, f"visualisasi_{args.feature_type}")
            os.makedirs(output_dir, exist_ok=True)
            
            categories = ['kertas', 'organik', 'plastik'] if args.category == 'all' else [args.category]
            
            for category in categories:
                category_path = os.path.join(dataset_path, category)
                if os.path.exists(category_path):
                    print(f"\nMemproses kategori {category}...")
                    images = os.listdir(category_path)[:args.sample]
                    for image_name in images:
                        image_path = os.path.join(category_path, image_name)
                        visualize_feature_extraction(image_path, args.feature_type, output_dir)
                else:
                    print(f"Warning: Kategori {category} tidak ditemukan di {category_path}")
        
        print("\nProses visualisasi selesai!")
        print("\nProses visualisasi selesai!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
