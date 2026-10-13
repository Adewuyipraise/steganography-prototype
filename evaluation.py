"""Evaluation metrics - FIXED CALCULATIONS"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class MetricsCalculator:
    @staticmethod
    def bit_error_rate(original_bits, extracted_bits):
        """Calculate Bit Error Rate (BER)"""
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len == 0:
            return 1.0
        errors = sum(original_bits[i] != extracted_bits[i] for i in range(min_len))
        return errors / min_len
    
    @staticmethod
    def text_recovery_accuracy(original_text, recovered_text):
        """
        Calculate text recovery accuracy percentage.
        FIXED: Explicit percentage calculation.
        
        Recovery Accuracy = (Correctly Recovered Characters / Total Characters) × 100
        """
        if not original_text:
            return 0.0
        
        # Handle case where recovered is None or empty
        if not recovered_text:
            return 0.0
        
        # Use character-by-character comparison
        min_len = min(len(original_text), len(recovered_text))
        matches = sum(original_text[i] == recovered_text[i] for i in range(min_len))
        
        # Calculate percentage
        accuracy = (matches / len(original_text)) * 100
        
        return accuracy
    
    @staticmethod
    def psnr(original, distorted):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(float) - distorted.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    @staticmethod
    def ssim_index(original, distorted):
        """Calculate Structural Similarity Index"""
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            dist_gray = cv2.cvtColor(distorted, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            dist_gray = distorted
        
        return ssim(orig_gray, dist_gray, data_range=255)
    
    @staticmethod
    def compute_all_metrics(original_image, transformed_image, 
                           original_bits, extracted_bits, 
                           original_text, recovered_text):
        """
        Compute all metrics at once - FIXED
        """
        ber = MetricsCalculator.bit_error_rate(original_bits, extracted_bits)
        recovery_acc = MetricsCalculator.text_recovery_accuracy(original_text, recovered_text)
        psnr_val = MetricsCalculator.psnr(original_image, transformed_image)
        ssim_val = MetricsCalculator.ssim_index(original_image, transformed_image)
        
        return {
            'ber': ber,
            'recovery_accuracy': recovery_acc,  # Now correctly returns percentage
            'psnr': psnr_val,
            'ssim': ssim_val,
            'robustness_score': MetricsCalculator.robustness_score(recovery_acc, ber)
        }
    
    @staticmethod
    def robustness_score(avg_recovery_accuracy, avg_ber):
        """Calculate composite robustness score"""
        norm_accuracy = avg_recovery_accuracy / 100.0
        return norm_accuracy * (1 - avg_ber)
