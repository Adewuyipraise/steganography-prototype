"""DCT-based embedding with adaptive edge detection and differential encoding"""

import numpy as np
import cv2

class DCTEmbedder:
    """Handles DCT-based embedding with binary edge classification"""
    
    def __init__(self, config):
        self.block_size = config.BLOCK_SIZE
        self.positions = config.MID_FREQ_POSITIONS
        self.alpha_edge = config.EMBEDDING_STRENGTH_EDGE
        self.alpha_smooth = config.EMBEDDING_STRENGTH_SMOOTH
        self.edge_threshold = config.EDGE_DENSITY_THRESHOLD
    
    def _rgb_to_ycbcr(self, image):
        """Convert RGB to YCbCr color space"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _ycbcr_to_rgb(self, image):
        """Convert YCbCr back to RGB"""
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    
    def _compute_edge_map(self, luminance):
        """
        Compute Sobel edge magnitude map.
        Chapter 3 Algorithm 3.1: Sobel Edge Detection
        """
        sobel_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        # Normalize to [0, 1]
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)
        return edge_magnitude
    
    def _get_block_edge_density(self, edge_map, block_row, block_col):
        """
        Get edge density (proportion of edge pixels) for an 8x8 block.
        Chapter 3 Algorithm 3.1: Edge density calculation
        """
        r_start = block_row * self.block_size
        r_end = r_start + self.block_size
        c_start = block_col * self.block_size
        c_end = c_start + self.block_size
        block_edges = edge_map[r_start:r_end, c_start:c_end]
        
        # Apply threshold to get binary edge map (T_edge = 0.15 as per Chapter 3)
        binary_edges = (block_edges >= 0.15).astype(np.float32)
        return np.mean(binary_edges)  # Proportion of edge pixels
    
    def _classify_block(self, edge_map, block_row, block_col):
        """
        Classify block as 'edge' or 'smooth' based on 30% threshold.
        Chapter 3 Algorithm 3.1: Block classification
        """
        edge_density = self._get_block_edge_density(edge_map, block_row, block_col)
        if edge_density >= self.edge_threshold:
            return 'edge', self.alpha_edge
        else:
            return 'smooth', self.alpha_smooth
    
    def _apply_dct(self, block):
        """Apply 2D DCT to an 8x8 block using scipy for compatibility"""
        from scipy.fftpack import dct
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _apply_idct(self, block):
        """Apply inverse 2D DCT to an 8x8 block"""
        from scipy.fftpack import idct
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _embed_bit_in_coefficient(self, coeff, bit, alpha):
        """
        Embed a single bit by modifying DCT coefficient.
        Chapter 3 Algorithm 3.4: Coefficient modification
        """
        if bit == 1:
            return coeff + alpha
        else:
            return coeff - alpha
    
    def embed(self, cover_image, bitstream):
        """
        Embed bitstream into cover image using binary edge classification.
        
        Args:
            cover_image: RGB image as numpy array (H, W, 3)
            bitstream: List of bits to embed
            
        Returns:
            stego_image, embedding_map (positions where bits were placed)
        """
        # Convert to YCbCr
        ycbcr = self._rgb_to_ycbcr(cover_image)
        luminance = ycbcr[:, :, 0].copy()
        
        # Compute edge map (Algorithm 3.1)
        edge_map = self._compute_edge_map(luminance)
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        max_bits = blocks_h * blocks_w * len(self.positions)
        if len(bitstream) > max_bits:
            print(f"Warning: Bitstream exceeds capacity. Truncating.")
            bitstream = bitstream[:max_bits]
        
        bit_idx = 0
        embedding_map = []  # Track (block_row, block_col, pos_idx, bit, alpha)
        classification_map = []  # Track edge/smooth classification
        
        # Process each block
        for br in range(blocks_h):
            for bc in range(blocks_w):
                # Extract block
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                # Apply DCT
                dct_block = self._apply_dct(block)
                
                # Classify block as edge or smooth (Algorithm 3.1)
                block_type, alpha = self._classify_block(edge_map, br, bc)
                classification_map.append((br, bc, block_type, alpha))
                
                # Embed bits into selected mid-frequency positions (Algorithm 3.2)
                for pos_idx, (u, v) in enumerate(self.positions):
                    if bit_idx >= len(bitstream):
                        break
                    
                    if u < self.block_size and v < self.block_size:
                        bit = bitstream[bit_idx]
                        original_coeff = dct_block[u, v]
                        modified_coeff = self._embed_bit_in_coefficient(
                            original_coeff, bit, alpha
                        )
                        dct_block[u, v] = modified_coeff
                        embedding_map.append((br, bc, u, v, bit, alpha))
                        bit_idx += 1
                
                # Apply IDCT
                reconstructed_block = self._apply_idct(dct_block)
                luminance[r_start:r_end, c_start:c_end] = reconstructed_block
        
        # Reconstruct image
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        stego_image = self._ycbcr_to_rgb(ycbcr)
        
        print(f"Block classification: {len([c for c in classification_map if c[2]=='edge'])} edge blocks, "
              f"{len([c for c in classification_map if c[2]=='smooth'])} smooth blocks")
        
        return stego_image.astype(np.uint8), embedding_map
    
    def get_capacity(self, image_shape):
        """Calculate maximum embedding capacity in bits"""
        h, w = image_shape[:2]
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        return blocks_h * blocks_w * len(self.positions)
    
    def get_classification_map(self, image):
        """
        Generate classification map for visualization.
        Returns: 2D array with 1 for edge blocks, 0 for smooth blocks
        """
        ycbcr = self._rgb_to_ycbcr(image)
        luminance = ycbcr[:, :, 0].copy()
        edge_map = self._compute_edge_map(luminance)
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        class_map = np.zeros((blocks_h, blocks_w))
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                block_type, _ = self._classify_block(edge_map, br, bc)
                class_map[br, bc] = 1 if block_type == 'edge' else 0
        
        return class_map
