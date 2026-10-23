"""DCT-based embedding - FINAL WORKING VERSION"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct


class DCTEmbedder:
    def __init__(self, config):
        self.config = config
        self.block_size = config.BLOCK_SIZE
        self.positions = config.MID_FREQ_POSITIONS
        self.positions_robust = list(config.ROBUST_POSITIONS)
        self.alpha_edge = getattr(config, 'EMBEDDING_STRENGTH_EDGE', 0.26)
        self.alpha_smooth = getattr(config, 'EMBEDDING_STRENGTH_SMOOTH', 0.26)
        self.alpha = self.alpha_smooth  # Use conservative default for smooth content
        self.SYNC_MARKER = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
        self.FLAT_THRESHOLD = getattr(config, 'EDGE_DENSITY_THRESHOLD', 5.0)
        self.edge_threshold = 15.0  # Threshold for edge detection
        self._rs_robust = None
    
    def _rgb_to_ycbcr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def _ycbcr_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    
    def _apply_dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _apply_idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _is_edge_block(self, block):
        """Determine if block is edge-rich using Sobel edge detection."""
        grad_x = cv2.Sobel(block.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(block.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        return edge_strength > self.edge_threshold
    
    def _is_flat_block(self, block, br, bc):
        """Skip genuinely flat blocks (content-aware)."""
        block_range = block.max() - block.min()
        if block_range < 5.0:
            return True
        # Also skip very smooth blocks with low variance
        if np.var(block) < 2.0:
            return True
        return False
    
    def _embed_bit(self, coeff, bit):
        step = self.alpha * 10
        quantized = np.round(coeff / step) * step
        current_idx = int(np.round(quantized / step))
        current_parity = abs(current_idx) % 2
        
        if current_parity != bit:
            if quantized >= 0:
                quantized += step if bit == 1 else -step
            else:
                quantized -= step if bit == 1 else -step
        
        return quantized
    
    def embed(self, cover_image, bitstream):
        full_bitstream = self.SYNC_MARKER + list(bitstream)
        
        ycbcr = self._rgb_to_ycbcr(cover_image).astype(np.float32)
        luminance = ycbcr[:, :, 0].copy()
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        bit_idx = 0
        skipped_blocks = 0
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                if bit_idx >= len(full_bitstream):
                    break
                
                r_start = br * self.block_size
                c_start = bc * self.block_size
                block = luminance[r_start:r_start+self.block_size,
                                  c_start:c_start+self.block_size]
                
                # Skip flat blocks (content-aware)
                if self._is_flat_block(block, br, bc):
                    skipped_blocks += 1
                    continue
                
                dct_block = self._apply_dct2(block)
                
                for u, v in self.positions:
                    if bit_idx >= len(full_bitstream):
                        break
                    bit = full_bitstream[bit_idx]
                    dct_block[u, v] = self._embed_bit(dct_block[u, v], bit)
                    bit_idx += 1
                
                reconstructed = self._apply_idct2(dct_block)
                luminance[r_start:r_start+self.block_size,
                          c_start:c_start+self.block_size] = reconstructed
            
            if bit_idx >= len(full_bitstream):
                break
        
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        stego_image = self._ycbcr_to_rgb(ycbcr.astype(np.uint8))
        
        print(f"Embedded {bit_idx} bits (skipped {skipped_blocks} flat blocks)")
        
        return stego_image.astype(np.uint8), bit_idx
    
    def get_capacity(self, image_shape):
        h, w = image_shape[:2]
        return (h // self.block_size) * (w // self.block_size) * len(self.positions)
    
    # ------------------------------------------------------------------
    # Robust pipeline: JPEG-guard grid embedding + repetition (see robust.py)
    # ------------------------------------------------------------------
    def _rs(self):
        if self._rs_robust is None:
            from reed_solomon import TurboReedSolomon
            self._rs_robust = TurboReedSolomon(
                self.config.RS_DATA_SYMBOLS,
                self.config.RS_ECC_SYMBOLS,
                self.config.RS_INTERLEAVE_DEPTH,
                verbose=False,
            )
        return self._rs_robust

    def max_message_bits(self, image_shape):
        """Max message bits the robust pipeline can embed in an image."""
        from robust import max_message_bits
        h, w = image_shape[:2]
        return max_message_bits(h, w, self.block_size, self.config)

    def build_robust_stream(self, message_bits):
        """Build the full repeated stream: K x [sync | RS-encoded payload]."""
        from robust import SYNC_MARKER, payload_bits
        payload = payload_bits(message_bits, self.config)
        enc_bits, _ = self._rs().encode_bitstream(payload)
        stream = []
        for _ in range(self.config.ROBUST_K_COPIES):
            stream.extend(SYNC_MARKER)
            stream.extend(enc_bits)
        return stream

    def embed_robust(self, cover_image, message_bits):
        """
        Embed a message with JPEG-guard grid steps in the crop-safe window.

        Steps per coefficient come from the libjpeg luminance quantisation
        table at the guard quality, so parity survives JPEG requantisation.
        The stream is repeated ROBUST_K_COPIES times for majority-vote
        recovery, and confined to a centered window that survives cropping.
        """
        from robust import band_cols, band_rows, quant_table

        stream = self.build_robust_stream(message_bits)
        guard = quant_table(self.config.ROBUST_JPEG_GUARD_QUALITY)
        step_table = { (u, v): int(guard[u, v]) for (u, v) in self.positions_robust }

        ycbcr = self._rgb_to_ycbcr(cover_image).astype(np.float32)
        luminance = ycbcr[:, :, 0]
        h, w = luminance.shape
        r0, r1 = band_rows(h, self.block_size, self.config)
        c0, c1 = band_cols(w, self.block_size, self.config)
        skip_mod = self.config.ROBUST_SKIP_MOD

        bit_idx = 0
        for br in range(r0, r1):
            for bc in range(c0, c1):
                if bit_idx >= len(stream):
                    break
                if (br + bc) % skip_mod == 0:
                    continue

                r_start = br * self.block_size
                c_start = bc * self.block_size
                block = luminance[r_start:r_start + self.block_size,
                                  c_start:c_start + self.block_size].copy()
                dct_block = self._apply_dct2(block)

                for (u, v) in self.positions_robust:
                    if bit_idx >= len(stream):
                        break
                    step = step_table[(u, v)]
                    bit = stream[bit_idx]
                    q = np.round(dct_block[u, v] / step) * step
                    idx = int(round(q / step))
                    if abs(idx) % 2 != bit:
                        q += step if bit == 1 else -step
                    dct_block[u, v] = q
                    bit_idx += 1

                reconstructed = self._apply_idct2(dct_block)
                luminance[r_start:r_start + self.block_size,
                          c_start:c_start + self.block_size] = reconstructed

        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        stego_image = self._ycbcr_to_rgb(ycbcr.astype(np.uint8))

        print(f"   Embedded {bit_idx} bits (robust, guard Q{self.config.ROBUST_JPEG_GUARD_QUALITY}, "
              f"{self.config.ROBUST_K_COPIES} copies)")

        return stego_image.astype(np.uint8), bit_idx
    
    def get_classification_map(self, image):
        """Generate a real Sobel-based edge classification map for visualization."""
        ycbcr = self._rgb_to_ycbcr(image).astype(np.float32)
        luminance = ycbcr[:, :, 0]
        h, w = luminance.shape
        
        # 1. Apply Sobel
        grad_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()
            
        # 2. Threshold
        binary_edges = (edge_magnitude >= 0.15).astype(np.float32)
        
        # 3. Classify blocks
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        class_map = np.zeros((blocks_h, blocks_w))
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                
                block_edges = binary_edges[r_start:r_end, c_start:c_end]
                edge_density = np.mean(block_edges)
                
                # If 30% or more pixels are edges, mark as 1 (Edge Block)
                if edge_density >= 0.30:
                    class_map[br, bc] = 1
                    
        return class_map