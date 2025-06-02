import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load and preprocess image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def extract_color_features(img):
    """Extract color features using HSV histogram."""
    # Calculate HSV histogram
    h_hist = cv2.calcHist([img], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([img], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([img], [2], None, [32], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Calculate mean values for each channel
    h_mean = np.mean(img[:,:,0])
    s_mean = np.mean(img[:,:,1])
    v_mean = np.mean(img[:,:,2])
    
    # Combine features
    features = np.concatenate([h_hist, s_hist, v_hist, [h_mean, s_mean, v_mean]])
    return features

def save_features(features, output_path):
    """Save extracted features to file."""
    np.save(output_path, features)

def main():
    # Setup paths
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'citra' / 'hasil_ekstraksi'
    output_dir.mkdir(exist_ok=True)
    
    # Process training and testing images
    for dataset in ['training', 'testing']:
        for category in ['organik', 'plastik', 'kertas']:
            input_dir = base_path / 'citra' / dataset / category
            print(f"Processing {dataset} {category} images...")
            
            # Process all images in the category
            for img_path in input_dir.glob('*.*'):
                try:
                    # Load and process image
                    img = load_image(img_path)
                    features = extract_color_features(img)
                    
                    # Save features with dataset and category in filename
                    output_path = output_dir / f"{dataset}_{category}_{img_path.stem}_color_features.npy"
                    save_features(features, output_path)
                    print(f"Processed {img_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == '__main__':
    main()
