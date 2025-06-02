import cv2
import numpy as np
import mahotas
from skimage.feature import local_binary_pattern
from pathlib import Path

def load_image(image_path):
    """Load and preprocess image for texture analysis."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def extract_glcm_features(gray_img):
    """Extract Haralick texture features using GLCM."""
    # Calculate Haralick features for different angles
    angles = [0, 45, 90, 135]  # mahotas uses degrees instead of radians
    haralick_features = mahotas.features.haralick(gray_img)
    return haralick_features.mean(axis=0)  # Average over angles

def extract_lbp_features(gray_img):
    """Extract Local Binary Pattern features."""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_texture_features(gray_img):
    """Combine GLCM and LBP features."""
    glcm_features = extract_glcm_features(gray_img)
    lbp_features = extract_lbp_features(gray_img)
    return np.concatenate([glcm_features, lbp_features])

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
                    gray_img = load_image(img_path)
                    features = extract_texture_features(gray_img)
                    
                    # Save features with dataset and category in filename
                    output_path = output_dir / f"{dataset}_{category}_{img_path.stem}_texture_features.npy"
                    save_features(features, output_path)
                    print(f"Processed {img_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == '__main__':
    main()
