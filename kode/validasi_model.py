import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_features():
    """Load extracted features from hasil_ekstraksi folder"""
    hasil_dir = Path("citra/hasil_ekstraksi")
    
    # Load color features
    warna = np.load(hasil_dir / "fitur_warna.npz")
    X_warna = warna['features']
    y = warna['labels']
    
    # Load shape features
    bentuk = np.load(hasil_dir / "fitur_bentuk.csv")
    X_bentuk = np.genfromtxt(hasil_dir / "fitur_bentuk.csv", delimiter=',')
    
    # Load texture features
    tekstur = np.load(hasil_dir / "fitur_tekstur.npz")
    X_tekstur = tekstur['features']
    
    # Combine all features
    X = np.hstack([X_warna, X_bentuk, X_tekstur])
    
    return X, y

def split_data(X, y, test_size=0.2):
    """Split data into training and testing sets"""
    # Get indices for test set
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(test_size * len(X))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train KNN model and evaluate its performance"""
    # Initialize and train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = knn.score(X_test, y_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Kertas', 'Organik', 'Plastik'])
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm, y_pred

def plot_confusion_matrix(cm):
    """Plot confusion matrix as heatmap"""
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5, 2.5], ['Kertas', 'Organik', 'Plastik'])
    plt.yticks([0.5, 1.5, 2.5], ['Kertas', 'Organik', 'Plastik'])
    
    # Save plot
    save_dir = Path("citra/hasil_ekstraksi/validasi")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

def main():
    print("Loading features...")
    X, y = load_features()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training and evaluating model...")
    accuracy, report, cm, y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)
    
    print("\nGenerating confusion matrix plot...")
    plot_confusion_matrix(cm)
    
    print("\nValidation complete! Results have been saved to citra/hasil_ekstraksi/validasi/")

if __name__ == "__main__":
    main()
