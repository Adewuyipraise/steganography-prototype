"""Steganalysis evaluation - Chapter 3 Section 3.6.5"""

import numpy as np
from scipy.stats import chisquare

class Steganalysis:
    """Chi-square steganalysis detection"""
    
    @staticmethod
    def chi_square_test(cover_image, stego_image):
        """
        Perform chi-square steganalysis test.
        Chapter 3 Section 3.6.5
        
        Args:
            cover_image: Original cover image
            stego_image: Stego image to test
            
        Returns:
            Dictionary with chi-square statistic, p-value, and detection status
        """
        # Convert to grayscale if RGB
        if len(cover_image.shape) == 3:
            cover_gray = np.mean(cover_image, axis=2).astype(np.uint8)
            stego_gray = np.mean(stego_image, axis=2).astype(np.uint8)
        else:
            cover_gray = cover_image
            stego_gray = stego_image
        
        # Compute histograms
        hist_cover = np.histogram(cover_gray.flatten(), bins=256, range=(0, 256))[0]
        hist_stego = np.histogram(stego_gray.flatten(), bins=256, range=(0, 256))[0]
        
        # Add small epsilon to avoid division by zero
        hist_cover = hist_cover + 1e-10
        
        # Chi-square test
        chi2_stat, p_value = chisquare(hist_stego, f_exp=hist_cover)
        
        # Determine if detectable (p_value < 0.05)
        is_detectable = p_value < 0.05
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'is_detectable': is_detectable,
            'security_score': 1.0 - float(p_value) if p_value < 0.05 else 1.0
        }
    
    @staticmethod
    def analyze_pairs(cover_images, stego_images):
        """
        Analyze multiple image pairs for steganalysis.
        
        Args:
            cover_images: List of cover images
            stego_images: List of stego images
            
        Returns:
            Aggregated statistics
        """
        results = []
        for cover, stego in zip(cover_images, stego_images):
            result = Steganalysis.chi_square_test(cover, stego)
            results.append(result)
        
        # Aggregate
        detection_rate = sum(1 for r in results if r['is_detectable']) / len(results)
        avg_p_value = np.mean([r['p_value'] for r in results])
        avg_security = np.mean([r['security_score'] for r in results])
        
        return {
            'detection_rate': detection_rate,
            'avg_p_value': avg_p_value,
            'avg_security_score': avg_security,
            'individual_results': results
        }
