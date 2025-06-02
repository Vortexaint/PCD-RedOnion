import cv2
import numpy as np
from pathlib import Path

def load_image(image_path):
    """Load and preprocess image for shape analysis."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def extract_shape_features(binary_img):
    """Extract shape features using contours and Hu moments."""
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image")
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate basic shape metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate shape features
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    compactness = (perimeter ** 2) / area
    
    # Calculate Hu Moments
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments)
    
    # Combine all features
    features = np.concatenate([
        [aspect_ratio, extent, compactness],
        hu_moments.flatten()
    ])
    
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
                    binary_img = load_image(img_path)
                    features = extract_shape_features(binary_img)
                    
                    # Save features with dataset and category in filename
                    output_path = output_dir / f"{dataset}_{category}_{img_path.stem}_shape_features.npy"
                    save_features(features, output_path)
                    print(f"Processed {img_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == '__main__':
    main()
