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
        self.blur_kernels = config.BLUR_KERNELS
    
    def apply_jpeg_compression(self, image, quality):
        """
        Apply JPEG compression.
        """
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = np.array(Image.open(buffer))
        return compressed
    
    def apply_resizing(self, image, scale_factor):
        """
        Resize image and then restore to original dimensions.
        """
        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        restored = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored
    
    def apply_cropping(self, image, crop_percent):
        """
        Randomly crop and restore to original size (filling missing with zeros).
        """
        h, w = image.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        start_h = np.random.randint(0, crop_h)
        start_w = np.random.randint(0, crop_w)
        
        cropped = np.zeros_like(image)
        end_h = min(start_h + (h - crop_h), h)
        end_w = min(start_w + (w - crop_w), w)
        cropped[start_h:end_h, start_w:end_w] = image[start_h:end_h, start_w:end_w]
        
        return cropped
    
    def apply_noise(self, image, variance):
        """
        Add Gaussian noise.
        """
        noise = np.random.normal(0, np.sqrt(variance), image.shape) * 255
        noisy = np.clip(image + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def apply_blur(self, image, kernel_size):
        """
        Apply Gaussian blur.
        """
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_all_transformations(self, image):
        """
        Apply all transformations and return results.
        """
        results = {}
        for q in self.jpeg_qualities:
            results[f'jpeg_{q}'] = self.apply_jpeg_compression(image, q)
        for s in self.resize_factors:
            results[f'resize_{s}'] = self.apply_resizing(image, s)
        for c in self.crop_percents:
            results[f'crop_{int(c*100)}'] = self.apply_cropping(image, c)
        for v in self.noise_variances:
            results[f'noise_{v}'] = self.apply_noise(image, v)
        for k in self.blur_kernels:
            results[f'blur_{k[0]}'] = self.apply_blur(image, k)
        return results
