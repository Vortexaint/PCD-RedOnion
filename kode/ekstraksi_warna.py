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

def segment_object(img):
    """Segment the object from background using HSV color space."""
    # Convert to HSV
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    
    # Create mask using Otsu's thresholding on Value channel
    _, mask = cv2.threshold(blur[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def extract_color_features(img):
    """Extract enhanced color features using HSV color space and segmentation."""
    # Segment object
    mask = segment_object(img)
    
    # Calculate HSV histogram of segmented object
    h_hist = cv2.calcHist([img], [0], mask, [32], [0, 180])
    s_hist = cv2.calcHist([img], [1], mask, [32], [0, 256])
    v_hist = cv2.calcHist([img], [2], mask, [32], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Calculate statistical features for each channel
    h_stats = calculate_channel_stats(img[:,:,0], mask)
    s_stats = calculate_channel_stats(img[:,:,1], mask)
    v_stats = calculate_channel_stats(img[:,:,2], mask)
    
    # Calculate color moments
    moments = calculate_color_moments(img, mask)
    
    # Combine all features
    features = np.concatenate([
        h_hist, s_hist, v_hist,  # Color distribution (96 features)
        h_stats, s_stats, v_stats,  # Statistical features (15 features)
        moments  # Color moments (9 features)
    ])
    
    return features

def calculate_channel_stats(channel, mask):
    """Calculate statistical features for a channel."""
    # Apply mask
    masked_channel = channel[mask > 0]
    
    if len(masked_channel) == 0:
        return np.zeros(5)
    
    # Calculate statistics
    mean = np.mean(masked_channel)
    std = np.std(masked_channel)
    skewness = np.mean(((masked_channel - mean) / (std + 1e-10)) ** 3)
    kurtosis = np.mean(((masked_channel - mean) / (std + 1e-10)) ** 4) - 3
    entropy = -np.sum(np.histogram(masked_channel, bins=32, density=True)[0] * 
                     np.log2(np.histogram(masked_channel, bins=32, density=True)[0] + 1e-10))
    
    return np.array([mean, std, skewness, kurtosis, entropy])

def calculate_color_moments(img, mask):
    """Calculate color moments for the image."""
    moments = []
    
    for i in range(3):  # For each channel
        channel = img[:,:,i][mask > 0]
        if len(channel) == 0:
            moments.extend([0, 0, 0])
            continue
            
        # First moment - mean
        moment1 = np.mean(channel)
        # Second moment - standard deviation
        moment2 = np.std(channel)
        # Third moment - skewness
        moment3 = np.mean(((channel - moment1) / (moment2 + 1e-10)) ** 3)
        
        moments.extend([moment1, moment2, moment3])
    
    return np.array(moments)

def save_features(features, output_path):
    """Save extracted features to file."""
    np.save(output_path, features)

def main():
    # Setup paths
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'citra' / 'hasil_ekstraksi'
    output_dir.mkdir(exist_ok=True)
    
    # Process training and testing images
    total_processed = 0
    errors = []
    
    for dataset in ['training', 'testing']:
        for category in ['organik', 'plastik', 'kertas']:
            input_dir = base_path / 'citra' / dataset / category
            if not input_dir.exists():
                print(f"Warning: Directory {input_dir} does not exist")
                continue
                
            print(f"\nProcessing {dataset} {category} images...")
            category_count = 0
            
            # Process all images in the category
            for img_path in input_dir.glob('*.*'):
                if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    continue
                    
                try:
                    # Load and process image
                    img = load_image(img_path)
                    
                    # Print image shape and basic stats
                    print(f"\nAnalyzing {img_path.name}:")
                    print(f"Image size: {img.shape}")
                    print(f"Value range - H: [{np.min(img[:,:,0])}, {np.max(img[:,:,0])}], "
                          f"S: [{np.min(img[:,:,1])}, {np.max(img[:,:,1])}], "
                          f"V: [{np.min(img[:,:,2])}, {np.max(img[:,:,2])}]")
                    
                    features = extract_color_features(img)
                    
                    # Save features with dataset and category in filename
                    output_path = output_dir / f"{dataset}_{category}_{img_path.stem}_color_features.npy"
                    save_features(features, output_path)
                    print(f"✓ Successfully extracted features: {features.shape} dimensions")
                    
                    category_count += 1
                    total_processed += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {img_path.name}: {str(e)}"
                    print(f"✗ {error_msg}")
                    errors.append(error_msg)
            
            print(f"\nProcessed {category_count} images in {category}")
    
    print(f"\nSummary:")
    print(f"Total images processed: {total_processed}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        print("Error details:")
        for error in errors:
            print(f"- {error}")

if __name__ == '__main__':
    main()
