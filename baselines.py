"""Baseline Methods: LSB Spatial Domain and DCT without Error Correction - Updated for Chapter 3"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct

from text_utils import text_to_bits, bits_to_text

class LSBSteganography:
    """
    Simple LSB (Least Significant Bit) spatial domain steganography.
    This is the most basic baseline method for comparison.
    Chapter 3 Section 3.7.2
    """
    
    def __init__(self):
        self.name = "LSB Spatial Domain"
    
    def text_to_bits(self, text):
        """Convert text to binary bits"""
        return text_to_bits(text)
    
    def bits_to_text(self, bits):
        """Convert bits back to text"""
        return bits_to_text(bits)
    
    def embed(self, cover_image, secret_text):
        """
        Embed secret text using LSB substitution.
        Chapter 3 Section 3.7.2
        """
        bits = self.text_to_bits(secret_text)
        
        # Flatten the image
        flat = cover_image.flatten()
        
        # Check capacity
        max_bits = len(flat)
        if len(bits) > max_bits:
            print(f"Warning: Text too long. Truncating to {max_bits//8} characters.")
            bits = bits[:max_bits]
        
        # Embed bits into LSB
        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | bit
        
        # Reshape back
        stego_image = flat.reshape(cover_image.shape)
        
        return stego_image.astype(np.uint8)
    
    def extract(self, stego_image, num_bits):
        """Extract hidden bits from stego image"""
        flat = stego_image.flatten()
        extracted = []
        
        for i in range(min(num_bits, len(flat))):
            extracted.append(flat[i] & 1)
        
        return extracted
    
    def get_capacity(self, image_shape):
        """Maximum bits that can be hidden"""
        h, w = image_shape[:2]
        return h * w * 3  # RGB channels


class DCTWithoutEC:
    """
    DCT-based steganography WITHOUT Hamming error correction.
    Chapter 3 Section 3.7.3
    Same as proposed method but no error correction coding.
    Uses uniform quantization-based embedding.
    """
    
    def __init__(self, config):
        self.block_size = config.BLOCK_SIZE
        self.positions = config.MID_FREQ_POSITIONS
        self.embedding_strength = 0.3
        self.name = "DCT without Error Correction"
    
    def _rgb_to_ycbcr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _ycbcr_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    
    def _apply_dct(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _apply_idct(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _embed_bit_in_coefficient(self, coeff, bit, alpha):
        step = alpha * 10
        quantized = np.round(coeff / step) * step
        current_parity = int(np.abs(np.round(quantized / step))) % 2
        
        if current_parity != bit:
            if quantized >= 0:
                quantized += step if bit == 1 else -step
            else:
                quantized -= step if bit == 1 else -step
        
        return quantized
    
    def _extract_bit_from_coefficient(self, coeff, alpha):
        step = alpha * 10
        quantized_idx = np.round(coeff / step)
        return int(np.abs(quantized_idx)) % 2
    
    def text_to_bits(self, text):
        return text_to_bits(text)
    
    def bits_to_text(self, bits):
        return bits_to_text(bits)
    
    def embed(self, cover_image, secret_text):
        original_bits = self.text_to_bits(secret_text)
        
        ycbcr = self._rgb_to_ycbcr(cover_image)
        luminance = ycbcr[:, :, 0].copy()
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        max_bits = blocks_h * blocks_w * len(self.positions)
        if len(original_bits) > max_bits:
            print(f"Warning: Text too long. Truncating to {max_bits//8} characters.")
            original_bits = original_bits[:max_bits]
        
        bit_idx = 0
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                dct_block = self._apply_dct(block)
                
                for u, v in self.positions:
                    if bit_idx >= len(original_bits):
                        break
                    
                    if u < self.block_size and v < self.block_size:
                        bit = original_bits[bit_idx]
                        dct_block[u, v] = self._embed_bit_in_coefficient(
                            dct_block[u, v], bit, self.embedding_strength
                        )
                        bit_idx += 1
                
                reconstructed_block = self._apply_idct(dct_block)
                luminance[r_start:r_end, c_start:c_end] = reconstructed_block
                
                if bit_idx >= len(original_bits):
                    break
            if bit_idx >= len(original_bits):
                break
        
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        stego_image = self._ycbcr_to_rgb(ycbcr)
        
        return stego_image.astype(np.uint8), len(original_bits)
    
    def extract(self, stego_image, num_bits):
        ycbcr = self._rgb_to_ycbcr(stego_image)
        luminance = ycbcr[:, :, 0].copy()
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        extracted_bits = []
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                dct_block = self._apply_dct(block)
                
                for u, v in self.positions:
                    if len(extracted_bits) >= num_bits:
                        break
                    if u < self.block_size and v < self.block_size:
                        coeff = dct_block[u, v]
                        bit = self._extract_bit_from_coefficient(coeff, self.embedding_strength)
                        extracted_bits.append(bit)
                
                if len(extracted_bits) >= num_bits:
                    break
            if len(extracted_bits) >= num_bits:
                break
        
        return extracted_bits[:num_bits]
    
    def get_capacity(self, image_shape):
        h, w = image_shape[:2]
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        return blocks_h * blocks_w * len(self.positions)
