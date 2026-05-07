"""DCT-based embedding with adaptive edge detection and differential encoding"""

import numpy as np
import cv2

class DCTEmbedder:
    """Handles DCT-based embedding with Sobel edge adaptation and differential encoding"""
    
    def __init__(self, config):
        self.block_size = config.BLOCK_SIZE
        self.coeff_pairs = config.COEFF_PAIRS
        self.base_strength = config.EMBEDDING_STRENGTH_BASE
        self.edge_factor = config.EDGE_STRENGTH_FACTOR
    
    def _rgb_to_ycbcr(self, image):
        """Convert RGB to YCbCr color space"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _ycbcr_to_rgb(self, image):
        """Convert YCbCr back to RGB"""
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    
    def _compute_edge_map(self, luminance):
        """Compute Sobel edge magnitude map for adaptive embedding"""
        sobel_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        return edge_magnitude
    
    def _get_block_edge_intensity(self, edge_map, block_row, block_col):
        """Get average edge intensity for an 8x8 block"""
        r_start = block_row * self.block_size
        r_end = r_start + self.block_size
        c_start = block_col * self.block_size
        c_end = c_start + self.block_size
        block_edges = edge_map[r_start:r_end, c_start:c_end]
        return np.mean(block_edges)
    
    def _apply_dct(self, block):
        """Apply 2D DCT to an 8x8 block"""
        return cv2.dct(np.float32(block))
    
    def _apply_idct(self, block):
        """Apply inverse 2D DCT to an 8x8 block"""
        return cv2.idct(block)
    
    def _embed_bit_differential(self, dct_block, bit, p1, p2, strength):
        """
        Embed a bit using the difference between two coefficients.
        If bit=1, ensure dct_block[p1] > dct_block[p2] + strength
        If bit=0, ensure dct_block[p1] < dct_block[p2] - strength
        """
        u1, v1 = p1
        u2, v2 = p2
        
        # Calculate the average and desired difference
        avg = (dct_block[u1, v1] + dct_block[u2, v2]) / 2
        
        if bit == 1:
            if dct_block[u1, v1] <= dct_block[u2, v2] + strength:
                dct_block[u1, v1] = avg + strength / 2 + 0.1
                dct_block[u2, v2] = avg - strength / 2 - 0.1
        else:
            if dct_block[u1, v1] >= dct_block[u2, v2] - strength:
                dct_block[u1, v1] = avg - strength / 2 - 0.1
                dct_block[u2, v2] = avg + strength / 2 + 0.1
        
        return dct_block

    def embed(self, cover_image, bitstream):
        """
        Embed bitstream into cover image using differential DCT.
        """
        ycbcr = self._rgb_to_ycbcr(cover_image)
        luminance = ycbcr[:, :, 0].copy().astype(np.float32)
        
        edge_map = self._compute_edge_map(luminance)
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        bit_idx = 0
        embedding_map = []
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                if bit_idx >= len(bitstream):
                    break
                
                r_start, r_end = br * self.block_size, (br + 1) * self.block_size
                c_start, c_end = bc * self.block_size, (bc + 1) * self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                dct_block = self._apply_dct(block)
                
                edge_intensity = self._get_block_edge_intensity(edge_map, br, bc)
                adaptive_strength = self.base_strength * (1 + self.edge_factor * edge_intensity)
                
                for p1, p2 in self.coeff_pairs:
                    if bit_idx >= len(bitstream):
                        break
                    
                    bit = bitstream[bit_idx]
                    dct_block = self._embed_bit_differential(dct_block, bit, p1, p2, adaptive_strength)
                    embedding_map.append((br, bc, p1, p2, bit))
                    bit_idx += 1
                
                reconstructed_block = self._apply_idct(dct_block)
                luminance[r_start:r_end, c_start:c_end] = reconstructed_block
        
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
        stego_image = self._ycbcr_to_rgb(ycbcr)
        
        return stego_image, embedding_map
    
    def get_capacity(self, image_shape):
        h, w = image_shape[:2]
        return (h // self.block_size) * (w // self.block_size) * len(self.coeff_pairs)
