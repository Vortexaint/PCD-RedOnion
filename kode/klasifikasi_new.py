import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def get_color_distribution(color_features):
    """Get detailed color distribution from features."""
    # First 32 features are hue histogram
    hue_hist = color_features[:32]
    # Next 32 features are saturation histogram
    sat_hist = color_features[32:64]
    # Next 32 features are value histogram
    val_hist = color_features[64:96]
    
    # Get statistical features
    h_stats = color_features[96:101]  # mean, std, skewness, kurtosis, entropy
    s_stats = color_features[101:106]
    v_stats = color_features[106:111]
    
    color_dist = {}
    
    # Calculate percentage of each color range
    total_pixels = np.sum(hue_hist)
    if total_pixels > 0:
        # Yellow range (20-40 degrees in HSV)
        yellow_mask = slice(int(20/180*32), int(40/180*32))
        yellow_pixels = np.sum(hue_hist[yellow_mask])
        
        # Brown range (0-20 degrees in HSV)
        brown_mask = slice(0, int(20/180*32))
        brown_pixels = np.sum(hue_hist[brown_mask])
        
        # Green range (60-120 degrees in HSV)
        green_mask = slice(int(60/180*32), int(120/180*32))
        green_pixels = np.sum(hue_hist[green_mask])
        
        # Calculate white pixels (high value, low saturation)
        white_threshold = 0.8  # 80% of max value
        low_sat_threshold = 0.3  # 30% of max saturation
        white_pixels = np.sum(val_hist[int(white_threshold * 32):]) * \
                      np.sum(sat_hist[:int(low_sat_threshold * 32)]) / total_pixels
        
        # Store normalized values
        color_dist['yellow'] = yellow_pixels / total_pixels if s_stats[0] > 50 else 0
        color_dist['brown'] = brown_pixels / total_pixels if s_stats[0] > 30 and v_stats[0] < 200 else 0
        color_dist['green'] = green_pixels / total_pixels if s_stats[0] > 50 else 0
        color_dist['white'] = min(1.0, white_pixels)
        
        # Store color statistics
        color_dist['saturation'] = s_stats[0]  # mean saturation
        color_dist['value'] = v_stats[0]  # mean brightness
    
    return color_dist

def get_object_color(color_dist):
    """Determine the actual color of the object with enhanced detection"""
    # First check for achromatic colors (black, white, gray)
    if color_dist['saturation'] < 40:  # Low saturation threshold increased
        if color_dist['value'] > 180:
            return "Putih"
        elif color_dist['value'] < 70:
            return "Hitam"
        else:
            return "Abu-abu"
    
    # Initialize color scores
    color_scores = {
        "Kuning": color_dist['yellow'] * (color_dist['saturation'] / 255),
        "Coklat": color_dist['brown'] * (1 - color_dist['value'] / 255),
        "Hijau": color_dist['green'] * (color_dist['saturation'] / 255)
    }
    
    # Filter significant colors (>10% presence and good confidence)
    significant_colors = [
        color for color, score in color_scores.items()
        if score > 0.1
    ]
    
    if not significant_colors:
        if color_dist['value'] > 180 and color_dist['saturation'] < 60:
            return "Putih kekuningan"  # Light yellowish
        elif color_dist['value'] < 100 and color_dist['brown'] > 0.1:
            return "Coklat gelap"  # Dark brown
        else:
            return "Warna campuran"  # Mixed colors
    
    return "/".join(significant_colors)

def load_features(features_dir, category, is_training=True):
    """Load features for a specific category of waste."""
    prefix = "training" if is_training else "testing"
    features_dict = {}
    file_names = []
    
    for feature_type in ['color', 'shape', 'texture']:
        feature_files = list(features_dir.glob(f"{prefix}_{category}*_{feature_type}_features.npy"))
        for f in feature_files:
            sample_name = f.stem.replace(f"{prefix}_{category}_", "").replace(f"_{feature_type}_features", "")
            
            if sample_name not in features_dict:
                features_dict[sample_name] = {}
                file_names.append(sample_name)
            
            features_dict[sample_name][feature_type] = np.load(f)
    
    combined_features = []
    final_file_names = []
    color_features_list = []
    
    for name in sorted(set(file_names)):
        if all(feat_type in features_dict[name] for feat_type in ['color', 'shape', 'texture']):
            combined = np.concatenate([
                features_dict[name]['color'],
                features_dict[name]['shape'],
                features_dict[name]['texture']
            ])
            combined_features.append(combined)
            final_file_names.append(name)
            color_features_list.append(features_dict[name]['color'])
    
    return combined_features, final_file_names, color_features_list

def combine_features(is_training=True):
    """Combine features with enhanced weighting for shape and texture."""
    base_path = Path(__file__).parent.parent
    features_dir = base_path / 'citra' / 'hasil_ekstraksi'
    
    categories = ['organik', 'plastik', 'kertas']
    X = []
    y = []
    file_names = []
    color_features = []
    
    for label, category in enumerate(categories):
        category_features, category_files, category_colors = load_features(features_dir, category, is_training)
        
        # Process each sample
        for features, color in zip(category_features, category_colors):
            # Split features into color, shape, and texture
            color_part = features[:120]  # First 120 features are color
            shape_part = features[120:220]  # Next 100 features are shape
            texture_part = features[220:]  # Remaining features are texture
            
            # Apply feature importance weighting
            color_weighted = color_part * 0.3  # Reduce color importance
            shape_weighted = shape_part * 1.5  # Increase shape importance
            texture_weighted = texture_part * 1.2  # Slightly increase texture importance
            
            # Combine weighted features
            combined = np.concatenate([color_weighted, shape_weighted, texture_weighted])
            X.append(combined)
            
        y.extend([label] * len(category_features))
        file_names.extend([f"{category}_{f}" for f in category_files])
        color_features.extend(category_colors)
    
    return np.array(X), np.array(y), file_names, color_features

def print_confusion_matrix(y_true, y_pred, labels):
    """Print confusion matrix in a readable format."""
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriks Konfusi:")
    print("            Prediksi")
    print("Aktual    ", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print("\n" + "-" * 40)
    
    for i, label in enumerate(labels):
        print(f"{label:<10}", end="")
        for j in range(len(labels)):
            print(f"{cm[i,j]:>8}", end="")
        print()

def evaluate_classifier(clf_name, clf, X_train, y_train):
    """Evaluate classifier using cross-validation."""
    scores = cross_val_score(clf, X_train, y_train, cv=2)  # Using 2-fold CV due to small dataset
    print(f"\n{clf_name} Skor Validasi-silang: {scores}")
    print(f"Rata-rata Skor: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

def train_classifiers(X, y, color_features):
    """Train multiple classifiers with color-based feature importance."""
    print("\nAnalisis karakteristik data training...")
    
    # Analyze color distributions for each class
    class_color_stats = {0: [], 1: [], 2: []}  # For Organik, Plastik, Kertas
    for i, (features, label) in enumerate(zip(color_features, y)):
        color_dist = get_color_distribution(features)
        class_color_stats[label].append(color_dist)
    
    # Print class characteristics
    categories = ['Organik', 'Plastik', 'Kertas']
    for class_idx, stats in class_color_stats.items():
        if stats:
            avg_yellow = np.mean([s['yellow'] for s in stats])
            avg_brown = np.mean([s['brown'] for s in stats])
            avg_white = np.mean([s['white'] for s in stats])
            avg_sat = np.mean([s['saturation'] for s in stats])
            
            print(f"\nKarakteristik {categories[class_idx]}:")
            print(f"- Rata-rata kuning: {avg_yellow:.1%}")
            print(f"- Rata-rata coklat: {avg_brown:.1%}")
            print(f"- Rata-rata putih: {avg_white:.1%}")
            print(f"- Rata-rata saturasi: {avg_sat:.1f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize classifiers with optimized parameters
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=2,
            weights='distance'  # Weight points by their distance
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
    }
    
    # Train and evaluate each classifier
    trained_models = {}
    for name, clf in classifiers.items():
        print(f"\nMelatih {name}...")
        evaluate_classifier(name, clf, X_scaled, y)
        clf.fit(X_scaled, y)
        trained_models[name] = clf
    
    return trained_models, scaler

def get_class_confidence(color_dist, predicted_class):
    """Calculate confidence score based on color distribution and predicted class."""
    confidence = 0.0
    
    if predicted_class == 'Organik':
        # Organic waste tends to be yellow/brown
        yellow_score = color_dist['yellow'] * 0.4  # Yellow is important
        brown_score = color_dist['brown'] * 0.4    # Brown is equally important
        non_white_score = (1 - color_dist['white']) * 0.2  # Should not be too white
        
        confidence = max(yellow_score, brown_score) + non_white_score
        
        # Bonus for having both yellow and brown
        if color_dist['yellow'] > 0.2 and color_dist['brown'] > 0.2:
            confidence += 0.1
        
    elif predicted_class == 'Plastik':
        # Plastic tends to have clear colors and high saturation
        sat_score = (color_dist['saturation'] / 255) * 0.4
        brightness = (color_dist['value'] / 255)
        bright_score = brightness * 0.3 if brightness > 0.6 else 0
        color_variety = (1 - max(color_dist['yellow'], color_dist['brown'])) * 0.3
        
        confidence = sat_score + bright_score + color_variety
        
    elif predicted_class == 'Kertas':
        # Paper tends to be white/light colored
        white_score = color_dist['white'] * 0.6
        low_color_score = (1 - (color_dist['yellow'] + color_dist['brown'])) * 0.4
        
        confidence = white_score + low_color_score
        
        # Penalty for high saturation (paper usually has low saturation)
        if color_dist['saturation'] > 100:
            confidence *= 0.8
    
    return min(confidence, 1.0)

def predict_samples(models, scaler, X, file_names, color_features, y_true=None):
    """Predict using all models and output results."""
    X_scaled = scaler.transform(X)
    categories = ['Organik', 'Plastik', 'Kertas']
    
    for name, model in models.items():
        print(f"\nHasil Klasifikasi menggunakan {name}:")
        
        try:
            y_pred = model.predict(X_scaled)
            
            if name == 'Random Forest':
                importances = model.feature_importances_
                print("\nFitur Penting:")
                for i, importance in enumerate(importances):
                    if importance > 0.01:
                        print(f"Feature {i}: {importance:.4f}")
            
            print("\nDetail Prediksi:")
            for i, (pred, file_name, color_feat) in enumerate(zip(y_pred, file_names, color_features)):
                color_dist = get_color_distribution(color_feat)
                object_color = get_object_color(color_dist)
                confidence = get_class_confidence(color_dist, categories[pred])
                
                print(f"\n{file_name}:")
                print(f"- Kelas yang diprediksi: {categories[pred]}")
                print(f"- Warna benda: {object_color}")
                print(f"- Tingkat keyakinan: {confidence:.2%}")
                print(f"- Detail warna:")
                print(f"  Kuning: {color_dist['yellow']:.1%}")
                print(f"  Coklat: {color_dist['brown']:.1%}")
                print(f"  Putih: {color_dist['white']:.1%}")
                print(f"  Saturasi: {color_dist['saturation']:.1f}")
                print(f"  Kecerahan: {color_dist['value']:.1f}")
                
                # Check if prediction matches color expectations
                if categories[pred] == 'Organik':
                    if color_dist['yellow'] < 0.1 and color_dist['brown'] < 0.1:
                        print("⚠️ Peringatan: Sampah organik biasanya berwarna kuning/coklat")
                        print(f"   - Kuning terdeteksi: {color_dist['yellow']:.1%}")
                        print(f"   - Coklat terdeteksi: {color_dist['brown']:.1%}")
                        print("   - Coba periksa pencahayaan dan posisi objek")
                    
                elif categories[pred] == 'Plastik':
                    if color_dist['saturation'] < 20:
                        print("⚠️ Peringatan: Plastik biasanya memiliki warna yang lebih jelas")
                        print(f"   - Saturasi terdeteksi: {color_dist['saturation']:.1f}")
                        print("   - Coba tambah pencahayaan atau atur posisi objek")
                    
                elif categories[pred] == 'Kertas':
                    if color_dist['white'] < 0.3:
                        print("⚠️ Peringatan: Kertas biasanya berwarna putih/terang")
                        print(f"   - Putih terdeteksi: {color_dist['white']:.1%}")
                        print("   - Coba tambah pencahayaan atau kurangi bayangan")
            
            if y_true is not None:
                print(f"\nEvaluasi Model {name}:")
                print(classification_report(y_true, y_pred, target_names=categories))
                print_confusion_matrix(y_true, y_pred, categories)
        
        except Exception as e:
            print(f"Error during prediction with {name}: {str(e)}")

def save_models(models, scaler, output_dir):
    """Save all trained models and scaler."""
    for name, model in models.items():
        with open(output_dir / f'model_{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
    # Setup paths
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'kode'
    output_dir.mkdir(exist_ok=True)
    
    # Load and combine training features
    print("Loading training features...")
    X_train, y_train, _, color_features_train = combine_features(is_training=True)
    
    # Train classifiers with color features
    print("\nTraining classifiers...")
    models, scaler = train_classifiers(X_train, y_train, color_features_train)
    
    # Save models
    print("\nSaving models...")
    save_models(models, scaler, output_dir)
    
    # Load and predict testing features
    print("\nProcessing testing samples...")
    X_test, y_test, file_names_test, color_features_test = combine_features(is_training=False)
    predict_samples(models, scaler, X_test, file_names_test, color_features_test)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
