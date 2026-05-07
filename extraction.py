"""DCT-based extraction module with differential encoding"""

import numpy as np
import cv2

class DCTExtractor:
    """Handles extraction of bits from DCT coefficients using differential encoding"""
    
    def __init__(self, config):
        self.block_size = config.BLOCK_SIZE
        self.coeff_pairs = config.COEFF_PAIRS
    
    def _rgb_to_ycbcr(self, image):
        """Convert RGB to YCbCr color space"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _apply_dct(self, block):
        """Apply 2D DCT to an 8x8 block"""
        return cv2.dct(np.float32(block))
    
    def extract(self, stego_image, num_bits):
        """
        Extract bitstream from stego image using differential DCT.
        """
        ycbcr = self._rgb_to_ycbcr(stego_image)
        luminance = ycbcr[:, :, 0].copy().astype(np.float32)
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        extracted_bits = []
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                if len(extracted_bits) >= num_bits:
                    break
                
                r_start, r_end = br * self.block_size, (br + 1) * self.block_size
                c_start, c_end = bc * self.block_size, (bc + 1) * self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                dct_block = self._apply_dct(block)
                
                for p1, p2 in self.coeff_pairs:
                    if len(extracted_bits) >= num_bits:
                        break
                    
                    u1, v1 = p1
                    u2, v2 = p2
                    
                    # Extract based on which coefficient is larger
                    if dct_block[u1, v1] > dct_block[u2, v2]:
                        extracted_bits.append(1)
                    else:
                        extracted_bits.append(0)
        
        return extracted_bits
