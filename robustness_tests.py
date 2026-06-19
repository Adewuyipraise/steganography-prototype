"""Image transformation simulations for robustness testing"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image

class TransformationTester:
    """Apply various transformations to stego images"""
    
    def __init__(self, config):
        self.jpeg_qualities = config.JPEG_QUALITIES
        self.resize_factors = config.RESIZE_FACTORS
        self.crop_percents = config.CROP_PERCENTS
        self.noise_variances = config.NOISE_VARIANCES
        self.combined_sequences = config.COMBINED_SEQUENCES
    
    def apply_jpeg_compression(self, image, quality):
        """
        Apply JPEG compression.
        Chapter 3 Section 3.5.1
        
        Args:
            image: RGB numpy array
            quality: 0-100 (higher = better quality)
            
        Returns:
            Compressed image
        """
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = np.array(Image.open(buffer))
        return compressed
    
    def apply_resizing(self, image, scale_factor):
        """
        Resize image using bilinear interpolation.
        Chapter 3 Section 3.5.2
        
        Args:
            image: RGB numpy array
            scale_factor: e.g., 0.5 = 50% down, 1.5 = 150% up
            
        Returns:
            Resized image (resized back to original dimensions for extraction)
        """
        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Resize down/up
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back to original dimensions
        restored = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored
    
    def apply_cropping(self, image, crop_percent):
        """
        Center cropping to simulate platform auto-cropping.
        Chapter 3 Section 3.5.3 (Table 3.5)
        
        Args:
            image: RGB numpy array
            crop_percent: Proportion of image to remove (0.10 to 0.40)
            
        Returns:
            Cropped image (resized back to original dimensions with black borders)
        """
        h, w = image.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        # Center crop
        start_h = crop_h // 2
        start_w = crop_w // 2
        end_h = h - crop_h // 2
        end_w = w - crop_w // 2
        
        # Create black image
        cropped = np.zeros_like(image)
        
        # Copy center region
        cropped[start_h:end_h, start_w:end_w] = image[start_h:end_h, start_w:end_w]
        
        return cropped
    
    def apply_noise(self, image, variance):
        """
        Add Gaussian noise in normalized [0,1] range.
        Chapter 3 Section 3.5.4 (Table 3.6)
        
        Args:
            image: RGB numpy array (0-255)
            variance: Noise variance in [0,1] range (0.01 to 0.05)
            
        Returns:
            Noisy image
        """
        # Convert to float in range [0,1]
        img_float = image.astype(np.float32) / 255.0
        
        # Standard deviation = sqrt(variance)
        sigma = np.sqrt(variance)
        noise = np.random.normal(0, sigma, img_float.shape)
        
        # Add noise and clip to valid range
        noisy_float = np.clip(img_float + noise, 0.0, 1.0)
        
        # Convert back to uint8
        return (noisy_float * 255).astype(np.uint8)
    
    def apply_blur(self, image, kernel_size):
        """Apply Gaussian blur (for additional testing)"""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_format_conversion(self, image, target_format):
        """Convert between PNG and JPEG formats"""
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        
        if target_format.upper() == 'PNG':
            pil_img.save(buffer, format='PNG')
        else:  # JPEG
            pil_img.save(buffer, format='JPEG', quality=95)
        
        buffer.seek(0)
        converted = np.array(Image.open(buffer))
        return converted
    
    def apply_combined_sequence(self, image, sequence_id):
        """
        Apply combined transformation sequences (C1-C4).
        Chapter 3 Section 3.5.5 (Table 3.7)
        
        Args:
            image: RGB numpy array
            sequence_id: 'C1', 'C2', 'C3', or 'C4'
            
        Returns:
            Transformed image
        """
        if sequence_id not in self.combined_sequences:
            raise ValueError(f"Unknown sequence: {sequence_id}")
        
        seq = self.combined_sequences[sequence_id]
        result = image.copy()
        
        for op in seq['order']:
            params = seq['params']
            if op == 'resize':
                result = self.apply_resizing(result, params['resize'])
            elif op == 'jpeg':
                result = self.apply_jpeg_compression(result, params['jpeg'])
            elif op == 'crop':
                result = self.apply_cropping(result, params['crop'])
            elif op == 'noise':
                result = self.apply_noise(result, params['noise'])
        
        return result
    
    def apply_all_transformations(self, image):
        """
        Apply all transformations and return results for evaluation.
        Chapter 3 Section 3.5
        
        Returns:
            Dictionary with transformation name -> transformed image
        """
        results = {}
        
        # Single transformations
        # JPEG compression (Table 3.3)
        for q in self.jpeg_qualities:
            results[f'jpeg_{q}'] = self.apply_jpeg_compression(image, q)
        
        # Resizing (Table 3.4)
        for s in self.resize_factors:
            results[f'resize_{int(s*100)}'] = self.apply_resizing(image, s)
        
        # Cropping (Table 3.5)
        for c in self.crop_percents:
            results[f'crop_{int(c*100)}'] = self.apply_cropping(image, c)
        
        # Gaussian noise (Table 3.6)
        for v in self.noise_variances:
            results[f'noise_{v}'] = self.apply_noise(image, v)
        
        # Combined sequences (Table 3.7)
        for seq_id in ['C1', 'C2', 'C3', 'C4']:
            results[f'combined_{seq_id}'] = self.apply_combined_sequence(image, seq_id)
        
        return results
