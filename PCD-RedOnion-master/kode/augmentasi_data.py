import cv2
import numpy as np
import os
from pathlib import Path
import random
import shutil

def load_image(image_path):
    """
    Membaca citra dari file.
    
    Args:
        image_path (str/Path): Path ke file citra
        
    Returns:
        numpy.array: Array citra atau None jika gagal
    """
    return cv2.imread(str(image_path))

def save_image(image, save_path):
    """
    Menyimpan citra ke file.
    
    Args:
        image (numpy.array): Array citra untuk disimpan
        save_path (str/Path): Path tujuan penyimpanan
    """
    cv2.imwrite(str(save_path), image)

def rotate_image(image, angle):
    """
    Merotasi citra dengan sudut tertentu.
    
    Args:
        image (numpy.array): Array citra input
        angle (float): Sudut rotasi dalam derajat
        
    Returns:
        numpy.array: Citra hasil rotasi
    """
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
    """
    Mencerminkan citra secara horizontal atau vertikal.
    
    Args:
        image (numpy.array): Array citra input
        direction (str): Arah pencerminan ('horizontal' atau 'vertical')
        
    Returns:
        numpy.array: Citra hasil pencerminan
    """
    if direction == 'horizontal':
        return cv2.flip(image, 1)  # 1 for horizontal flip
    else:
        return cv2.flip(image, 0)  # 0 for vertical flip

def augment_dataset(test_split=0.2, seed=42):
    """
    Augment dataset and split into training and testing sets
    
    Args:
        test_split (float): Proportion of original images to use for testing
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Define paths
    base_path = Path("citra/asli")
    categories = ["kertas_samples", "organik_samples", "plastik_samples"]
    
    # Rotation angles
    angles = [90, 180, 270]
    
    # Process each category
    for category in categories:
        print(f"\nProcessing {category}...")
        source_dir = base_path / category
        category_name = category.replace("_samples", "")
        
        # Create training and testing directories
        train_dir = Path("citra/training") / category_name
        test_dir = Path("citra/testing") / category_name
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of all images
        image_files = [f for f in source_dir.glob("*.*") 
                      if f.suffix.lower() in ['.jpg', '.png']]
        
        # Split into training and testing
        num_test = int(len(image_files) * test_split)
        test_files = random.sample(image_files, num_test)
        train_files = [f for f in image_files if f not in test_files]
        
        print(f"Total images: {len(image_files)}")
        print(f"Training images: {len(train_files)}")
        print(f"Testing images: {len(test_files)}")
        
        # Process training images with augmentation
        print("Augmenting training images...")
        for img_path in train_files:
            image = load_image(img_path)
            if image is None:
                continue
            
            # Save original
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
        
        # Copy testing images without augmentation
        print("Copying testing images...")
        for img_path in test_files:
            image = load_image(img_path)
            if image is None:
                continue
            save_image(image, test_dir / f"{img_path.stem}{img_path.suffix}")

def cleanup_directories():
    """Clean up training and testing directories before augmentation"""
    for dir_type in ['training', 'testing']:
        for category in ['kertas', 'organik', 'plastik']:
            dir_path = Path(f"citra/{dir_type}/{category}")
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"Cleaned up {dir_path}")

if __name__ == "__main__":
    print("Starting data augmentation process...")
    
    # Clean up existing directories
    print("\nCleaning up existing directories...")
    cleanup_directories()
    
    # Perform augmentation and splitting
    print("\nPerforming data augmentation and splitting...")
    augment_dataset(test_split=0.2)
    
    # Print summary
    print("\nAugmentation complete!")
    print("For each training image:")
    print("- Original preserved")
    print("- Rotations: 90°, 180°, 270°")
    print("- Horizontal flip")
    print("- Vertical flip")
    print("\nTest set contains original images only (no augmentation)")
    
    # Print statistics
    for dir_type in ['training', 'testing']:
        print(f"\n{dir_type.capitalize()} set statistics:")
        for category in ['kertas', 'organik', 'plastik']:
            path = Path(f"citra/{dir_type}/{category}")
            if path.exists():
                num_images = len(list(path.glob("*.*")))
                print(f"- {category}: {num_images} images")
