import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """
    Membuat visualisasi confusion matrix.
    
    Args:
        y_true (array): Label sebenarnya
        y_pred (array): Label hasil prediksi
        labels (list): Daftar nama kelas
        title (str): Judul plot
        
    Returns:
        None: Plot akan ditampilkan atau disimpan
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def klasifikasi_knn(X, y, k=3):
    """
    Melakukan klasifikasi menggunakan KNN (K-Nearest Neighbors).
    
    Args:
        X (array): Fitur input
        y (array): Label output
        k (int): Jumlah neighbor terdekat yang akan dipertimbangkan
        
    Returns:
        model: Model KNN yang telah dilatih
    """
    model = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Performance metrics
    acc = accuracy_score(y_test, y_pred)
    print("[KNN] Akurasi:", acc)
    print("\n[KNN] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['kertas', 'organik', 'plastik']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\n[KNN] Cross-validation scores:", cv_scores)
    print("[KNN] Mean CV accuracy: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['kertas', 'organik', 'plastik'], "KNN Confusion Matrix")
    
    return model

def klasifikasi_svm(X, y, kernel='rbf'):
    """
    Melakukan klasifikasi menggunakan SVM (Support Vector Machine).
    
    Args:
        X (array): Fitur input
        y (array): Label output
        kernel (str): Jenis kernel yang akan digunakan (contoh: 'linear', 'rbf', 'poly')
        
    Returns:
        model: Model SVM yang telah dilatih
    """
    model = SVC(kernel=kernel, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Performance metrics
    acc = accuracy_score(y_test, y_pred)
    print("[SVM] Akurasi:", acc)
    print("\n[SVM] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['kertas', 'organik', 'plastik']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\n[SVM] Cross-validation scores:", cv_scores)
    print("[SVM] Mean CV accuracy: {:.3f} (+/- {:.3f})".format(cv_scores.mean(), cv_scores.std() * 2))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['kertas', 'organik', 'plastik'], "SVM Confusion Matrix")
    
    return model

def prediksi_single_image(model, fitur, return_proba=False):
    """
    Melakukan prediksi untuk satu gambar/fitur tunggal.
    
    Args:
        model: Model yang telah dilatih (KNN atau SVM)
        fitur (array): Fitur gambar yang akan diprediksi
        return_proba (bool): Jika True, akan mengembalikan probabilitas untuk setiap kelas
        
    Returns:
        Prediksi kelas untuk gambar/fitur tersebut
    """
    fitur = np.array(fitur).reshape(1, -1)
    if return_proba and hasattr(model, 'predict_proba'):
        # Return class probabilities for each class
        probas = model.predict_proba(fitur)[0]
        classes = ['kertas', 'organik', 'plastik']
        return dict(zip(classes, probas))
    return model.predict(fitur)[0]