"""Dataset loader for USC-SIPI and BOSSBase"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetLoader:
    """Load and manage benchmark datasets"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def load_usc_sipi(self, path):
        """
        Load USC-SIPI standard test images.
        Chapter 3 Section 3.3.1
        """
        standard_images = [
            'lena.tiff', 'barbara.png', 'boat.png', 
            'house.png', 'pepper.png', 'couple.tiff',
            'elaine.tiff', 'gray21.512.tiff', 'slope.tiff'
        ]
        
        images = []
        for img_name in standard_images:
            img_path = os.path.join(path, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return images
    
    def load_bossbase(self, path, num_images=100):
        """
        Load BOSSBase natural images.
        Chapter 3 Section 3.3.1
        """
        images = []
        if os.path.exists(path):
            all_files = [f for f in os.listdir(path) if f.endswith(('.pgm', '.png'))]
            selected = np.random.choice(all_files, min(num_images, len(all_files)), replace=False)
            
            for fname in selected:
                img_path = os.path.join(path, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Convert grayscale to RGB for consistency
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    images.append(img_rgb)
        
        return images
    
    def split_dataset(self, images, tuningset_size=200, valset_size=50, testset_size=50):
        """
        Split dataset into tuning, validation, and test sets.
        Chapter 3 Section 3.3.2 (Table 3.2)
        """
        # Ensure we have enough images
        total_needed = tuningset_size + valset_size + testset_size
        if len(images) < total_needed:
            print(f"Warning: Only {len(images)} images available, need {total_needed}")
        
        # Split: tuning set first
        remaining = images
        tuning_set = remaining[:tuningset_size]
        remaining = remaining[tuningset_size:]
        
        # Split remaining into validation and test
        val_set = remaining[:valset_size]
        test_set = remaining[valset_size:valset_size+testset_size]
        
        return {
            'tuning': tuning_set,
            'validation': val_set,
            'test': test_set
        }
    
    def get_sample_image(self):
        """Create a sample image for quick testing"""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img[i, :, 0] = int(128 + 127 * np.sin(i/50))
            img[i, :, 1] = int(128 + 127 * np.cos(i/40))
            img[:, i, 2] = int(128 + 127 * np.sin(i/30))
        return img
