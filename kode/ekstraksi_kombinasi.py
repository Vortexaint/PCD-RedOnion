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
    Extract and combine features from a single image
    Returns a feature vector with fixed length:
    - Color features (515): HSV histogram (512) + RGB means (3)
    - Texture features (20): 5 GLCM properties × 4 angles
    - Shape features (3): area, aspect ratio, sides
    Total: 538 features
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

if __name__ == "__main__":
    print("Loading and combining features from training and testing sets...")
    
    # Get base path
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'citra')
    
    # Load and combine features
    X_train, X_test, y_train, y_test = load_dataset_split(base_path)
    
    print("\nDataset Statistics:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    print("\nTraining KNN classifier...")
    model_knn = klasifikasi_knn(X_train, y_train, k=3)
    y_pred_knn = model_knn.predict(X_test)
    
    print("\nTraining SVM classifier...")
    model_svm = klasifikasi_svm(X_train, y_train, kernel='rbf')
    y_pred_svm = model_svm.predict(X_test)
    
    # Save combined feature data
    output_dir = os.path.join(base_path, 'hasil_ekstraksi')
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'fitur_kombinasi.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
