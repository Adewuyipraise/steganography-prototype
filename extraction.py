"""DCT-based extraction module with differential encoding"""

import numpy as np
import cv2

class DCTExtractor:
    """Handles extraction of bits from DCT coefficients"""
    
    def __init__(self, config):
        self.block_size = config.BLOCK_SIZE
        self.positions = config.MID_FREQ_POSITIONS
    
    def _rgb_to_ycbcr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _apply_dct(self, block):
        """Apply 2D DCT using scipy"""
        from scipy.fftpack import dct
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _extract_bit_from_coefficient(self, coeff):
        """
        Extract a single bit from DCT coefficient.
        Chapter 3 Algorithm 3.6: If coefficient > 0 → bit = 1, else bit = 0
        """
        return 1 if coeff > 0 else 0
    
    def extract(self, stego_image, num_bits):
        """
        Extract bitstream from stego image.
        
        Args:
            stego_image: RGB image as numpy array
            num_bits: Number of bits to extract
            
        Returns:
            List of extracted bits
        """
        # Convert to YCbCr
        ycbcr = self._rgb_to_ycbcr(stego_image)
        luminance = ycbcr[:, :, 0].copy()
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        extracted_bits = []
        missing_blocks = 0
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                # Extract block
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                # Check if block is complete (for cropping handling)
                if block.shape != (self.block_size, self.block_size):
                    missing_blocks += 1
                    continue
                
                # Apply DCT
                dct_block = self._apply_dct(block)
                
                # Extract from mid-frequency positions (Algorithm 3.2)
                for u, v in self.positions:
                    if len(extracted_bits) >= num_bits:
                        break
                    if u < self.block_size and v < self.block_size:
                        coeff = dct_block[u, v]
                        bit = self._extract_bit_from_coefficient(coeff)
                        extracted_bits.append(bit)
                
                if len(extracted_bits) >= num_bits:
                    break
            if len(extracted_bits) >= num_bits:
                break
        
        if missing_blocks > 0:
            print(f"Warning: {missing_blocks} blocks missing due to cropping")
        
        return extracted_bits[:num_bits]
