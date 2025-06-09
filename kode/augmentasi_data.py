import cv2
import numpy as np
import os
from pathlib import Path

def load_image(image_path):
    return cv2.imread(str(image_path))

def save_image(image, save_path):
    cv2.imwrite(str(save_path), image)

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def mirror_image(image, direction='horizontal'):
    if direction == 'horizontal':
        return cv2.flip(image, 1)  # 1 for horizontal flip
    else:
        return cv2.flip(image, 0)  # 0 for vertical flip

def augment_dataset():
    # Define paths
    base_path = Path("citra/asli")
    categories = ["kertas_samples", "organik_samples", "plastik_samples"]
    
    # Rotation angles
    angles = [90, 180, 270]
    
    for category in categories:
        source_dir = base_path / category
        # Create training directory if it doesn't exist
        train_dir = Path("citra/training") / category.replace("_samples", "")
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image in the category
        for img_path in source_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.png']:
                # Load image
                image = load_image(img_path)
                if image is None:
                    continue
                
                # Save original image to training directory
                original_name = img_path.stem
                save_image(image, train_dir / f"{original_name}_original{img_path.suffix}")
                
                # Generate rotated versions
                for angle in angles:
                    rotated = rotate_image(image, angle)
                    save_image(rotated, train_dir / f"{original_name}_rot{angle}{img_path.suffix}")
                
                # Generate mirrored versions
                mirrored_h = mirror_image(image, 'horizontal')
                save_image(mirrored_h, train_dir / f"{original_name}_flip_h{img_path.suffix}")
                
                mirrored_v = mirror_image(image, 'vertical')
                save_image(mirrored_v, train_dir / f"{original_name}_flip_v{img_path.suffix}")

if __name__ == "__main__":
    print("Memulai proses augmentasi data...")
    augment_dataset()
    print("Augmentasi data selesai!")
    print("Setiap gambar telah diaugmentasi dengan:")
    print("- Rotasi (90°, 180°, 270°)")
    print("- Pencerminan horizontal")
    print("- Pencerminan vertikal")
