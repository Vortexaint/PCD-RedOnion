import numpy as np
import os
from ekstraksi_warna import ekstraksi_fitur_warna
from ekstraksi_tekstur import ekstraksi_fitur_tekstur
from ekstraksi_bentuk import ekstraksi_fitur_bentuk
from validasi_model import klasifikasi_knn, klasifikasi_svm
from sklearn.preprocessing import StandardScaler

def load_and_combine_features(image_path):
    """
    Extract and combine features from a single image
    """
    all_features = []
    
    # Extract color features (HSV histogram + RGB means = 515 features)
    color_features = ekstraksi_fitur_warna(image_path)
    if color_features is not None:
        all_features.extend(color_features.flatten())
    
    # Extract texture features (5 GLCM properties × 4 angles = 20 features)
    texture_features = ekstraksi_fitur_tekstur(image_path)
    if texture_features is not None:
        all_features.extend(texture_features.flatten())
    
    # Extract shape features (area, aspect ratio, sides = 3 features)
    shape_features = ekstraksi_fitur_bentuk(image_path)
    if shape_features is not None:
        if isinstance(shape_features, list):
            all_features.extend(shape_features)
        else:
            all_features.extend(shape_features.flatten())
    
    # Check if we have all features
    if len(all_features) == 0:
        return None
        
    # Convert to numpy array with fixed shape
    return np.array(all_features, dtype=np.float64)

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
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

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
