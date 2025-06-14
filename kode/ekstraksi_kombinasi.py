import numpy as np
import os
from ekstraksi_warna import ekstraksi_fitur_warna
from ekstraksi_tekstur import ekstraksi_fitur_tekstur
from ekstraksi_bentuk import ekstraksi_fitur_bentuk
from validasi_model import klasifikasi_knn, klasifikasi_svm
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

def load_and_combine_features(image_path):
    """
    Mengekstrak dan menggabungkan semua fitur dari satu citra.
    
    Total fitur yang dihasilkan (538):
    - Fitur warna (515): Histogram HSV (512) + Mean RGB (3)
    - Fitur tekstur (20): 5 properti GLCM × 4 sudut
    - Fitur bentuk (3): luas, rasio aspek, jumlah sisi
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        numpy.array: Vektor fitur gabungan atau None jika gagal
    """
    try:
        # Extract color features (HSV histogram + RGB means = 515 features)
        color_features = ekstraksi_fitur_warna(image_path)
        if color_features is None or len(color_features) != 515:
            print(f"Warning: Invalid color features for {image_path}")
            return None
        
        # Extract texture features (5 GLCM properties × 4 angles = 20 features)
        texture_features = ekstraksi_fitur_tekstur(image_path)
        if texture_features is None or len(texture_features) != 20:
            print(f"Warning: Invalid texture features for {image_path}")
            return None
        
        # Extract shape features (area, aspect ratio, sides = 3 features)
        shape_features = ekstraksi_fitur_bentuk(image_path)
        if shape_features is None or len(shape_features) != 3:
            print(f"Warning: Invalid shape features for {image_path}")
            return None
            
        # Combine all features
        all_features = np.concatenate([
            color_features.flatten(),
            texture_features.flatten(),
            shape_features if isinstance(shape_features, np.ndarray) else np.array(shape_features)
        ])
        
        # Ensure we have the expected number of features (515 + 20 + 3 = 538)
        if len(all_features) != 538:
            print(f"Warning: Unexpected feature length {len(all_features)} for {image_path}")
            return None
            
        return all_features.astype(np.float64)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def combine_texture_color_features(image_path):
    """
    Mengekstrak dan menggabungkan fitur tekstur dan warna dari satu citra.
    
    Total fitur yang dihasilkan (535):
    - Fitur warna (515): Histogram HSV (512) + Mean RGB (3)
    - Fitur tekstur (20): 5 properti GLCM × 4 sudut
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        numpy.array: Vektor fitur gabungan tekstur dan warna atau None jika gagal
    """
    try:
        # Extract color features (HSV histogram + RGB means = 515 features)
        color_features = ekstraksi_fitur_warna(image_path)
        if color_features is None or len(color_features) != 515:
            print(f"Warning: Invalid color features for {image_path}")
            return None
        
        # Extract texture features (5 GLCM properties × 4 angles = 20 features)
        texture_features = ekstraksi_fitur_tekstur(image_path)
        if texture_features is None or len(texture_features) != 20:
            print(f"Warning: Invalid texture features for {image_path}")
            return None
            
        # Combine texture and color features
        combined_features = np.concatenate([
            color_features.flatten(),
            texture_features.flatten()
        ])
        
        # Ensure we have the expected number of features (515 + 20 = 535)
        if len(combined_features) != 535:
            print(f"Warning: Unexpected feature length {len(combined_features)} for {image_path}")
            return None
            
        return combined_features.astype(np.float64)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_dataset_split(base_path):
    """
    Load training and testing datasets from designated folders
    """
    # Initialize lists for features and labels
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Classes to process
    classes = ['kertas', 'organik', 'plastik']
    
    # Process training data
    training_path = os.path.join(base_path, 'training')
    for class_name in classes:
        class_path = os.path.join(training_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Training path {class_path} does not exist")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = load_and_combine_features(img_path)
            if features is not None:
                X_train.append(features)
                y_train.append(class_name)
    
    # Process testing data
    testing_path = os.path.join(base_path, 'testing')
    for class_name in classes:
        class_path = os.path.join(testing_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Testing path {class_path} does not exist")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = load_and_combine_features(img_path)
            if features is not None:
                X_test.append(features)
                y_test.append(class_name)
    
    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No valid features could be extracted from the dataset")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature vector length: {X_train.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def visualize_process(image_path, prediction=None):
    """
    Create a 3x3 visualization of the feature extraction process
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Feature Extraction Process', fontsize=16)
    
    # 1. Original Image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original Image')
    
    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    
    # 3. Binary (for shape features)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary Image')
    
    # 4. Contour Detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = rgb_image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    axes[1, 0].imshow(contour_img)
    axes[1, 0].set_title('Contour Detection')
    
    # 5. HSV Color Space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    axes[1, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[1, 1].set_title('HSV Color Space')
    
    # 6. GLCM Texture
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    axes[1, 2].imshow(glcm[:, :, 0, 0], cmap='viridis')
    axes[1, 2].set_title(f'GLCM (Contrast: {glcm_contrast:.2f})')
    
    # 7. HSV Histogram
    hsv_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    axes[2, 0].plot(hsv_hist)
    axes[2, 0].set_title('HSV Histogram (Hue)')
    
    # 8. RGB Means
    means = cv2.mean(image)[:3]
    bar_colors = ['red', 'green', 'blue']
    axes[2, 1].bar(['R', 'G', 'B'], means, color=bar_colors)
    axes[2, 1].set_title('RGB Mean Values')
    
    # 9. Classification Result
    if prediction is not None:
        axes[2, 2].text(0.5, 0.5, f'Predicted:\n{prediction}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8))
    axes[2, 2].set_title('Classification')
    axes[2, 2].axis('off')
    
    # Remove axes ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def save_texture_color_features(base_path):
    """
    Mengekstrak dan menyimpan fitur tekstur dan warna dari dataset
    """
    # Initialize lists for features and labels
    X_train_tc, y_train_tc = [], []
    X_test_tc, y_test_tc = [], []
    
    classes = ['kertas', 'organik', 'plastik']
    
    # Process training data
    print("\nMengekstrak fitur tekstur dan warna dari data training...")
    training_path = os.path.join(base_path, 'training')
    for class_name in classes:
        class_path = os.path.join(training_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Training path {class_path} tidak ditemukan")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = combine_texture_color_features(img_path)
            if features is not None:
                X_train_tc.append(features)
                y_train_tc.append(class_name)
    
    # Process testing data
    print("\nMengekstrak fitur tekstur dan warna dari data testing...")
    testing_path = os.path.join(base_path, 'testing')
    for class_name in classes:
        class_path = os.path.join(testing_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Testing path {class_path} tidak ditemukan")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = combine_texture_color_features(img_path)
            if features is not None:
                X_test_tc.append(features)
                y_test_tc.append(class_name)
    
    # Convert to numpy arrays
    X_train_tc = np.array(X_train_tc)
    X_test_tc = np.array(X_test_tc)
    y_train_tc = np.array(y_train_tc)
    y_test_tc = np.array(y_test_tc)
    
    print(f"\nStatistik dataset tekstur-warna:")
    print(f"Jumlah sampel training: {len(X_train_tc)}")
    print(f"Jumlah sampel testing: {len(X_test_tc)}")
    print(f"Dimensi fitur: {X_train_tc.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_tc_scaled = scaler.fit_transform(X_train_tc)
    X_test_tc_scaled = scaler.transform(X_test_tc)
    
    return X_train_tc_scaled, X_test_tc_scaled, y_train_tc, y_test_tc

def combine_texture_shape_features(image_path):
    """
    Mengekstrak dan menggabungkan fitur tekstur dan bentuk dari satu citra.
    
    Total fitur yang dihasilkan (23):
    - Fitur tekstur (20): 5 properti GLCM × 4 sudut
    - Fitur bentuk (3): luas, rasio aspek, jumlah sisi
    
    Args:
        image_path (str): Path ke file citra
        
    Returns:
        numpy.array: Vektor fitur gabungan tekstur dan bentuk atau None jika gagal
    """
    try:
        # Extract texture features (5 GLCM properties × 4 angles = 20 features)
        texture_features = ekstraksi_fitur_tekstur(image_path)
        if texture_features is None or len(texture_features) != 20:
            print(f"Warning: Invalid texture features for {image_path}")
            return None
        
        # Extract shape features (area, aspect ratio, sides = 3 features)
        shape_features = ekstraksi_fitur_bentuk(image_path)
        if shape_features is None or len(shape_features) != 3:
            print(f"Warning: Invalid shape features for {image_path}")
            return None
            
        # Combine texture and shape features
        combined_features = np.concatenate([
            texture_features.flatten(),
            shape_features if isinstance(shape_features, np.ndarray) else np.array(shape_features)
        ])
        
        # Ensure we have the expected number of features (20 + 3 = 23)
        if len(combined_features) != 23:
            print(f"Warning: Unexpected feature length {len(combined_features)} for {image_path}")
            return None
            
        return combined_features.astype(np.float64)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def save_texture_shape_features(base_path):
    """
    Mengekstrak dan menyimpan fitur tekstur dan bentuk dari dataset
    """
    # Initialize lists for features and labels
    X_train_ts, y_train_ts = [], []
    X_test_ts, y_test_ts = [], []
    
    classes = ['kertas', 'organik', 'plastik']
    
    # Process training data
    print("\nMengekstrak fitur tekstur dan bentuk dari data training...")
    training_path = os.path.join(base_path, 'training')
    for class_name in classes:
        class_path = os.path.join(training_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Training path {class_path} tidak ditemukan")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = combine_texture_shape_features(img_path)
            if features is not None:
                X_train_ts.append(features)
                y_train_ts.append(class_name)
    
    # Process testing data
    print("\nMengekstrak fitur tekstur dan bentuk dari data testing...")
    testing_path = os.path.join(base_path, 'testing')
    for class_name in classes:
        class_path = os.path.join(testing_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Testing path {class_path} tidak ditemukan")
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = combine_texture_shape_features(img_path)
            if features is not None:
                X_test_ts.append(features)
                y_test_ts.append(class_name)
    
    # Convert to numpy arrays
    X_train_ts = np.array(X_train_ts)
    X_test_ts = np.array(X_test_ts)
    y_train_ts = np.array(y_train_ts)
    y_test_ts = np.array(y_test_ts)
    
    print(f"\nStatistik dataset tekstur-bentuk:")
    print(f"Jumlah sampel training: {len(X_train_ts)}")
    print(f"Jumlah sampel testing: {len(X_test_ts)}")
    print(f"Dimensi fitur: {X_train_ts.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_ts_scaled = scaler.fit_transform(X_train_ts)
    X_test_ts_scaled = scaler.transform(X_test_ts)
    
    return X_train_ts_scaled, X_test_ts_scaled, y_train_ts, y_test_ts

def visualize_texture_shape_combination(image_path, output_dir):
    """
    Membuat dan menyimpan visualisasi kombinasi fitur tekstur dan bentuk.
    
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
    
    # Buat figure dengan ukuran lebih besar
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('Visualisasi Kombinasi Fitur Tekstur dan Bentuk', fontsize=16)
    
    # 1. Tampilkan citra asli
    plt.subplot(331)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title('Citra Asli (RGB)')
    plt.axis('off')
    
    # 2. Tampilkan citra grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(332)
    plt.imshow(gray, cmap='gray')
    plt.title('Citra Grayscale')
    plt.axis('off')
    
    # 3. Tampilkan citra biner
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.subplot(333)
    plt.imshow(binary, cmap='gray')
    plt.title('Citra Biner (Otsu)')
    plt.axis('off')
    
    # 4. Tampilkan deteksi kontur
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = rgb_image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    plt.subplot(334)
    plt.imshow(contour_img)
    plt.title('Deteksi Kontur')
    plt.axis('off')
    
    # 5. Tampilkan fitur bentuk
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        plt.subplot(335)
        plt.bar(['Area', 'Aspect Ratio', 'Perimeter'], 
                [area/1000, aspect_ratio, perimeter/100])
        plt.title('Fitur Bentuk')
    else:
        plt.subplot(335)
        plt.text(0.5, 0.5, 'No contours found', ha='center')
        plt.title('Fitur Bentuk')
    
    # 6-9. Tampilkan GLCM untuk berbagai sudut
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    
    titles = ['GLCM 0°', 'GLCM 45°', 'GLCM 90°', 'GLCM 135°']
    for i in range(4):
        plt.subplot(336 + i)
        plt.imshow(glcm[:, :, 0, i], cmap='viridis')
        plt.title(titles[i])
        plt.colorbar()
        plt.axis('off')
    
    # Simpan visualisasi
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tekstur_bentuk.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualisasi disimpan di: {output_path}")

if __name__ == "__main__":
    print("Memulai proses ekstraksi dan klasifikasi fitur...")
    
    # Get base path
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'citra')
    output_dir = os.path.join(base_path, 'hasil_ekstraksi')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ekstraksi dan klasifikasi semua fitur gabungan
    print("\n=== Ekstraksi dan Klasifikasi Semua Fitur ===")
    X_train, X_test, y_train, y_test = load_dataset_split(base_path)
    
    print("\nMelatih classifier KNN untuk semua fitur...")
    model_knn = klasifikasi_knn(X_train, y_train, k=3)
    y_pred_knn = model_knn.predict(X_test)
    
    print("\nMelatih classifier SVM untuk semua fitur...")
    model_svm = klasifikasi_svm(X_train, y_train, kernel='rbf')
    y_pred_svm = model_svm.predict(X_test)
    
    # Simpan hasil ekstraksi semua fitur
    np.savez(os.path.join(output_dir, 'fitur_kombinasi.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test,
             y_pred_knn=y_pred_knn, y_pred_svm=y_pred_svm)
    
    # 2. Ekstraksi dan klasifikasi fitur tekstur-warna
    print("\n=== Ekstraksi dan Klasifikasi Fitur Tekstur-Warna ===")
    X_train_tc, X_test_tc, y_train_tc, y_test_tc = save_texture_color_features(base_path)
    
    print("\nMelatih classifier KNN untuk fitur tekstur-warna...")
    model_knn_tc = klasifikasi_knn(X_train_tc, y_train_tc, k=3)
    y_pred_knn_tc = model_knn_tc.predict(X_test_tc)
    
    print("\nMelatih classifier SVM untuk fitur tekstur-warna...")
    model_svm_tc = klasifikasi_svm(X_train_tc, y_train_tc, kernel='rbf')
    y_pred_svm_tc = model_svm_tc.predict(X_test_tc)
    
    # Simpan hasil ekstraksi fitur tekstur-warna
    np.savez(os.path.join(output_dir, 'fitur_tekstur_warna.npz'),
             X_train=X_train_tc, X_test=X_test_tc,
             y_train=y_train_tc, y_test=y_test_tc,
             y_pred_knn=y_pred_knn_tc, y_pred_svm=y_pred_svm_tc)
    
    # 3. Ekstraksi dan klasifikasi fitur tekstur-bentuk
    print("\n=== Ekstraksi dan Klasifikasi Fitur Tekstur-Bentuk ===")
    X_train_ts, X_test_ts, y_train_ts, y_test_ts = save_texture_shape_features(base_path)
    
    print("\nMelatih classifier KNN untuk fitur tekstur-bentuk...")
    model_knn_ts = klasifikasi_knn(X_train_ts, y_train_ts, k=3)
    y_pred_knn_ts = model_knn_ts.predict(X_test_ts)
    
    print("\nMelatih classifier SVM untuk fitur tekstur-bentuk...")
    model_svm_ts = klasifikasi_svm(X_train_ts, y_train_ts, kernel='rbf')
    y_pred_svm_ts = model_svm_ts.predict(X_test_ts)
    
    # Simpan hasil ekstraksi fitur tekstur-bentuk
    np.savez(os.path.join(output_dir, 'fitur_tekstur_bentuk.npz'),
             X_train=X_train_ts, X_test=X_test_ts,
             y_train=y_train_ts, y_test=y_test_ts,
             y_pred_knn=y_pred_knn_ts, y_pred_svm=y_pred_svm_ts)
    
    # 4. Visualisasi untuk sampel dari setiap kategori
    print("\n=== Membuat Visualisasi ===")
    vis_output_dir = os.path.join(output_dir, 'visualisasi_tekstur_bentuk')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Buat visualisasi untuk satu sampel dari setiap kategori
    training_path = os.path.join(base_path, 'training')
    for category in ['kertas', 'organik', 'plastik']:
        category_path = os.path.join(training_path, category)
        if os.path.exists(category_path):
            images = os.listdir(category_path)
            if images:
                # Ambil sampel gambar pertama dari setiap kategori
                sample_image = images[0]
                image_path = os.path.join(category_path, sample_image)
                visualize_texture_shape_combination(image_path, vis_output_dir)
                print(f"Visualisasi tekstur-bentuk dibuat untuk kategori {category}")
    
    print("\nProses ekstraksi, klasifikasi, dan visualisasi selesai!")
    print(f"Hasil ekstraksi dan visualisasi disimpan di: {output_dir}")
