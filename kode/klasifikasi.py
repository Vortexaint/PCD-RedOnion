import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def load_features(features_dir, category):
    """Load features for a specific category of waste."""
    features = []
    for feature_type in ['color', 'shape', 'texture']:
        feature_files = list(features_dir.glob(f"*_{category}*_{feature_type}_features.npy"))
        if not feature_files:
            continue
        category_features = [np.load(f) for f in feature_files]
        features.extend(category_features)
    return np.array(features)

def combine_features():
    """Combine features from all categories and create labels."""
    base_path = Path(__file__).parent.parent
    features_dir = base_path / 'citra' / 'hasil_ekstraksi'
    
    categories = ['organik', 'plastik', 'kertas']
    X = []
    y = []
    
    for label, category in enumerate(categories):
        category_features = load_features(features_dir, category)
        X.extend(category_features)
        y.extend([label] * len(category_features))
    
    return np.array(X), np.array(y)

def train_classifier(X, y):
    """Train a Random Forest classifier."""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Organik', 'Plastik', 'Kertas']))
    
    return clf, scaler

def save_model(clf, scaler, output_dir):
    """Save the trained model and scaler."""
    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
    # Setup paths
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'kode'
    output_dir.mkdir(exist_ok=True)
    
    # Load and combine features
    print("Loading features...")
    X, y = combine_features()
    
    # Train and evaluate classifier
    print("\nTraining classifier...")
    clf, scaler = train_classifier(X, y)
    
    # Save model
    print("\nSaving model...")
    save_model(clf, scaler, output_dir)
    print("Done!")

if __name__ == '__main__':
    main()
