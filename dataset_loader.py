"""Dataset loader for USC-SIPI and BOSSBase - Updated with better file handling"""

import os
import cv2
import numpy as np
from PIL import Image

class DatasetLoader:
    """
    Load and manage benchmark datasets for steganography experiments.
    Chapter 3 Section 3.3
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        print(f"📊 DatasetLoader initialized with seed={seed}")
    
    def load_usc_sipi(self, path):
        """
        Load USC-SIPI standard test images.
        Handles multiple formats: TIFF, PNG, JPEG, BMP, PGM
        """
        # Common USC-SIPI image names
        standard_images = [
            'lena.tiff', 'lena.tif', 'lena.png', 'lena.jpg',
            'barbara.png', 'barbara.tif', 'barbara.jpg',
            'boat.png', 'boat.tif', 'boat.jpg',
            'house.png', 'house.tif', 'house.jpg',
            'pepper.png', 'pepper.tif', 'pepper.jpg',
            'couple.tiff', 'couple.tif', 'couple.jpg',
            'elaine.tiff', 'elaine.tif', 'elaine.jpg',
            'gray21.512.tiff', 'gray21.512.png',
            'slope.tiff', 'slope.png',
            'mandrill.tiff', 'mandrill.png',
            'cameraman.tif', 'cameraman.png'
        ]
        
        images = []
        
        print(f"\n📁 Loading USC-SIPI images from: {path}")
        
        # Check if directory exists
        if not os.path.exists(path):
            print(f"❌ Directory not found: {path}")
            return images
        
        # Get all image files in the directory
        all_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
                    all_files.append(os.path.join(root, f))
        
        print(f"   Found {len(all_files)} total image files")
        
        # Try to load all images
        for img_path in all_files:
            try:
                # Try OpenCV first
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    print(f"  ✅ Loaded: {os.path.basename(img_path)} ({img_rgb.shape})")
                else:
                    # Try PIL as fallback
                    img_pil = Image.open(img_path)
                    img_np = np.array(img_pil)
                    if len(img_np.shape) == 2:  # Grayscale
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    print(f"  ✅ Loaded (PIL): {os.path.basename(img_path)} ({img_rgb.shape})")
            except Exception as e:
                print(f"  ⚠️ Error loading {os.path.basename(img_path)}: {e}")
        
        print(f"\n✅ Loaded {len(images)} USC-SIPI images")
        return images
    
    def load_bossbase(self, path, num_images=100):
        """
        Load BOSSBase natural images.
        BOSSBase images are typically in PGM format.
        """
        images = []
        
        print(f"\n📁 Loading BOSSBase images from: {path}")
        print(f"   Requested: {num_images} images")
        
        if not os.path.exists(path):
            print(f"❌ Directory not found: {path}")
            print("   Please download from: https://boss.physics.ualberta.ca/")
            return images
        
        # Get all image files
        all_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.pgm', '.png', '.ppm', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(root, f))
        
        print(f"   Found {len(all_files)} image files")
        
        if len(all_files) == 0:
            print("   ⚠️ No image files found in BOSSBase directory")
            print("   Make sure the images are in PGM or PNG format")
            return images
        
        # Randomly select images (with fixed seed for reproducibility)
        if len(all_files) > num_images:
            selected_files = np.random.choice(all_files, num_images, replace=False)
        else:
            selected_files = all_files
            print(f"   Using all {len(all_files)} images (less than requested {num_images})")
        
        for img_path in selected_files:
            try:
                # Read as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Try PIL as fallback
                    img_pil = Image.open(img_path)
                    img = np.array(img_pil)
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                if img is not None:
                    # Convert grayscale to RGB (3 channels) for consistency
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    images.append(img_rgb)
                    print(f"  ✅ Loaded: {os.path.basename(img_path)} ({img_rgb.shape})")
            except Exception as e:
                print(f"  ⚠️ Error loading {os.path.basename(img_path)}: {e}")
        
        print(f"\n✅ Loaded {len(images)} BOSSBase images")
        return images
    
    def load_from_folder(self, path, extensions=None):
        """Load images from any folder."""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.pgm', '.ppm']
        
        images = []
        
        if not os.path.exists(path):
            print(f"❌ Directory not found: {path}")
            return images
        
        print(f"\n📁 Loading images from: {path}")
        
        for fname in os.listdir(path):
            if any(fname.lower().endswith(ext) for ext in extensions):
                img_path = os.path.join(path, fname)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        # Try PIL
                        img_pil = Image.open(img_path)
                        img = np.array(img_pil)
                        if len(img.shape) == 2:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        else:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    images.append(img_rgb)
                    print(f"  ✅ Loaded: {fname} ({img_rgb.shape})")
                except Exception as e:
                    print(f"  ⚠️ Error loading {fname}: {e}")
        
        print(f"\n✅ Loaded {len(images)} images from folder")
        return images
    
    def split_dataset(self, images, 
                      tuningset_size=200, 
                      valset_size=50, 
                      testset_size=50):
        """Split dataset into tuning, validation, and test sets."""
        total_needed = tuningset_size + valset_size + testset_size
        
        print(f"\n📊 Splitting dataset (seed={self.seed})")
        print(f"   Total images available: {len(images)}")
        print(f"   Required: {total_needed} ({tuningset_size} tuning + {valset_size} validation + {testset_size} test)")
        
        if len(images) < total_needed:
            print(f"⚠️ Warning: Only {len(images)} images available, need {total_needed}")
            available = len(images)
            tuningset_size = min(tuningset_size, available // 2)
            valset_size = min(valset_size, (available - tuningset_size) // 2)
            testset_size = available - tuningset_size - valset_size
        
        # Shuffle with fixed seed for reproducibility
        shuffled = images.copy()
        np.random.seed(self.seed)
        np.random.shuffle(shuffled)
        
        # Split
        tuning_set = shuffled[:tuningset_size]
        remaining = shuffled[tuningset_size:]
        
        val_set = remaining[:valset_size]
        test_set = remaining[valset_size:valset_size + testset_size]
        
        print(f"\n✅ Split complete:")
        print(f"   Tuning set: {len(tuning_set)} images")
        print(f"   Validation set: {len(val_set)} images")
        print(f"   Test set: {len(test_set)} images")
        
        return {
            'tuning': tuning_set,
            'validation': val_set,
            'test': test_set
        }
    
    def get_sample_image(self, size=512):
        """Create a synthetic sample image for quick testing."""
        print(f"\n🎨 Creating synthetic sample image ({size}x{size})")
        
        img = np.zeros((size, size, 3), dtype=np.uint8)
        
        for i in range(size):
            img[i, :, 0] = int(128 + 127 * np.sin(i/50))
            img[i, :, 1] = int(128 + 127 * np.cos(i/40))
            img[:, i, 2] = int(128 + 127 * np.sin(i/30))
        
        for i in range(0, size, 64):
            cv2.rectangle(img, (i, i), (i+32, i+32), (200, 200, 200), -1)
            cv2.rectangle(img, (i+32, i), (i+64, i+32), (100, 100, 100), -1)
        
        return img
    
    def get_dataset_info(self, dataset):
        """Get statistics about a dataset."""
        if not dataset:
            return {'count': 0}
        
        sizes = [img.shape for img in dataset]
        unique_sizes = list(set(sizes))
        
        info = {
            'count': len(dataset),
            'unique_sizes': unique_sizes,
            'min_size': min(sizes, key=lambda x: x[0] * x[1]),
            'max_size': max(sizes, key=lambda x: x[0] * x[1]),
        }
        
        return info


def test_dataset_loader():
    """Quick test to verify dataset loader works"""
    print("\n" + "="*60)
    print("🧪 Testing DatasetLoader")
    print("="*60)
    
    loader = DatasetLoader(seed=42)
    
    # Test 1: Load USC-SIPI
    usc_images = loader.load_usc_sipi("dataset/usc_sipi/")
    print(f"USC-SIPI images loaded: {len(usc_images)}")
    
    # Test 2: Load BOSSBase
    boss_images = loader.load_bossbase("dataset/bossbase/", num_images=100)
    print(f"BOSSBase images loaded: {len(boss_images)}")
    
    # Test 3: Load sample images
    sample_images = loader.load_from_folder("dataset/sample_images/")
    print(f"Sample images loaded: {len(sample_images)}")
    
    # Combine all images
    all_images = usc_images + boss_images + sample_images
    print(f"\nTotal images loaded: {len(all_images)}")
    
    # Test 4: Split
    if len(all_images) > 10:
        split = loader.split_dataset(all_images, tuningset_size=200, valset_size=50, testset_size=50)
        print(f"Split: Tuning={len(split['tuning'])}, Validation={len(split['validation'])}, Test={len(split['test'])}")
    
    print("\n✅ DatasetLoader test complete!")


if __name__ == "__main__":
    test_dataset_loader()