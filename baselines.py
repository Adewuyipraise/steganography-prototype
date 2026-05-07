"""Baseline Methods: LSB Spatial Domain and DCT without Error Correction""" 

import numpy as np 
import cv2 

class LSBSteganography: 
    """ 
    Simple LSB (Least Significant Bit) spatial domain steganography. 
    This is the most basic baseline method for comparison. 
    """ 
    
    def __init__(self): 
        self.name = "LSB Spatial Domain" 
    
    def text_to_bits(self, text): 
        """Convert text to binary bits""" 
        bits = [] 
        for char in text: 
            ascii_val = ord(char) 
            for i in range(7, -1, -1):  # 8 bits per char
                bits.append((ascii_val >> i) & 1) 
        return bits 
    
    def bits_to_text(self, bits): 
        """Convert bits back to text""" 
        if len(bits) % 8 != 0: 
            bits = bits[:-(len(bits) % 8)] 
        
        chars = [] 
        for i in range(0, len(bits), 8): 
            byte = bits[i:i+8] 
            ascii_val = sum(bit << (7 - j) for j, bit in enumerate(byte)) 
            chars.append(chr(ascii_val)) 
        return ''.join(chars) 
    
    def embed(self, cover_image, secret_text): 
        """ 
        Embed secret text using LSB substitution. 
        Replaces LSB of each pixel with secret bits. 
        """ 
        bits = self.text_to_bits(secret_text) 
        flat = cover_image.copy().flatten() 
        max_bits = len(flat) 
        if len(bits) > max_bits: 
            bits = bits[:max_bits] 
        for i, bit in enumerate(bits): 
            flat[i] = (flat[i] & 0xFE) | bit 
        stego_image = flat.reshape(cover_image.shape) 
        return stego_image.astype(np.uint8) 
    
    def extract(self, stego_image, num_bits): 
        flat = stego_image.flatten() 
        extracted = [] 
        for i in range(min(num_bits, len(flat))): 
            extracted.append(flat[i] & 1) 
        return extracted 
    
    def get_capacity(self, image_shape): 
        h, w = image_shape[:2] 
        return h * w * 3 


class DCTWithoutEC: 
    """ 
    DCT-based steganography WITHOUT Hamming error correction. 
    """ 
    
    def __init__(self, config): 
        self.block_size = config.BLOCK_SIZE 
        self.positions = config.MID_FREQ_POSITIONS 
        self.base_strength = config.EMBEDDING_STRENGTH_BASE 
        self.edge_factor = config.EDGE_STRENGTH_FACTOR 
        self.name = "DCT without Error Correction" 
    
    def _rgb_to_ycbcr(self, image): 
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) 
    
    def _ycbcr_to_rgb(self, image): 
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB) 
    
    def _compute_edge_map(self, luminance): 
        sobel_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3) 
        sobel_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3) 
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2) 
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8) 
        return edge_magnitude 
    
    def _get_block_edge_intensity(self, edge_map, block_row, block_col): 
        r_start = block_row * self.block_size 
        r_end = r_start + self.block_size 
        c_start = block_col * self.block_size 
        c_end = c_start + self.block_size 
        block_edges = edge_map[r_start:r_end, c_start:c_end] 
        return np.mean(block_edges) 
    
    def _apply_dct(self, block): 
        return cv2.dct(np.float32(block)) 
    
    def _apply_idct(self, block): 
        return cv2.idct(block) 
    
    def _embed_bit_in_coefficient(self, coeff, bit, strength): 
        step = strength * 10 
        quantized = np.round(coeff / step) * step 
        current_parity = int(np.abs(quantized / step)) % 2 
        if current_parity != bit: 
            if quantized >= 0: 
                quantized += step if bit == 1 else -step 
            else: 
                quantized -= step if bit == 1 else -step 
        return quantized 
    
    def text_to_bits(self, text): 
        bits = [] 
        for char in text: 
            ascii_val = ord(char) 
            for i in range(7, -1, -1): 
                bits.append((ascii_val >> i) & 1) 
        return bits 
    
    def bits_to_text(self, bits): 
        if len(bits) % 8 != 0: 
            bits = bits[:-(len(bits) % 8)] 
        chars = [] 
        for i in range(0, len(bits), 8): 
            byte = bits[i:i+8] 
            ascii_val = sum(bit << (7 - j) for j, bit in enumerate(byte)) 
            chars.append(chr(ascii_val)) 
        return ''.join(chars) 
    
    def embed(self, cover_image, secret_text): 
        original_bits = self.text_to_bits(secret_text) 
        ycbcr = self._rgb_to_ycbcr(cover_image) 
        luminance = ycbcr[:, :, 0].copy().astype(np.float32) 
        edge_map = self._compute_edge_map(luminance) 
        h, w = luminance.shape 
        blocks_h, blocks_w = h // self.block_size, w // self.block_size 
        bit_idx = 0 
        for br in range(blocks_h): 
            for bc in range(blocks_w): 
                if bit_idx >= len(original_bits): break 
                r, c = br * self.block_size, bc * self.block_size 
                block = luminance[r:r+self.block_size, c:c+self.block_size] 
                dct_block = self._apply_dct(block) 
                strength = self.base_strength * (1 + self.edge_factor * self._get_block_edge_intensity(edge_map, br, bc)) 
                for u, v in self.positions: 
                    if bit_idx >= len(original_bits): break 
                    dct_block[u, v] = self._embed_bit_in_coefficient(dct_block[u, v], original_bits[bit_idx], strength) 
                    bit_idx += 1 
                luminance[r:r+self.block_size, c:c+self.block_size] = self._apply_idct(dct_block) 
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8) 
        return self._ycbcr_to_rgb(ycbcr), len(original_bits) 
    
    def extract(self, stego_image, num_bits): 
        ycbcr = self._rgb_to_ycbcr(stego_image) 
        luminance = ycbcr[:, :, 0].copy().astype(np.float32) 
        h, w = luminance.shape 
        blocks_h, blocks_w = h // self.block_size, w // self.block_size 
        extracted_bits = [] 
        for br in range(blocks_h): 
            for bc in range(blocks_w): 
                if len(extracted_bits) >= num_bits: break 
                r, c = br * self.block_size, bc * self.block_size 
                dct_block = self._apply_dct(luminance[r:r+self.block_size, c:c+self.block_size]) 
                for u, v in self.positions: 
                    if len(extracted_bits) >= num_bits: break 
                    coeff = dct_block[u, v] 
                    step = self.base_strength * 10 
                    bit = int(np.round(coeff / step)) % 2 
                    extracted_bits.append(bit) 
        return extracted_bits 
    
    def get_capacity(self, image_shape): 
        h, w = image_shape[:2] 
        return (h // self.block_size) * (w // self.block_size) * len(self.positions)
