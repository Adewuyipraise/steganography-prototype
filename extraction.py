"""DCT-based extraction - FINAL WORKING VERSION"""

import numpy as np
import cv2
from scipy.fftpack import dct


class DCTExtractor:
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
        self.edge_threshold = 15.0
        self._rs_robust = None
    
    def _apply_dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _is_flat_block(self, block, br, bc):
        """Skip genuinely flat blocks (content-aware)."""
        block_range = block.max() - block.min()
        if block_range < 5.0:
            return True
        if np.var(block) < 2.0:
            return True
        return False
    
    def _extract_bit(self, coeff):
        step = self.alpha * 10
        quantized_idx = np.round(coeff / step)
        return int(np.abs(quantized_idx)) % 2
    
    def _extract_raw_bits(self, stego_image, num_bits):
        ycbcr = cv2.cvtColor(stego_image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        luminance = ycbcr[:, :, 0].copy()
        
        h, w = luminance.shape
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        extracted_bits = []
        
        for br in range(blocks_h):
            for bc in range(blocks_w):
                if len(extracted_bits) >= num_bits:
                    break
                
                r_start = br * self.block_size
                r_end = r_start + self.block_size
                c_start = bc * self.block_size
                c_end = c_start + self.block_size
                block = luminance[r_start:r_end, c_start:c_end]
                
                if block.shape != (self.block_size, self.block_size):
                    continue
                
                # Skip flat blocks (SAME rule as embedder)
                if self._is_flat_block(block, br, bc):
                    continue
                
                dct_block = self._apply_dct2(block)
                
                for u, v in self.positions:
                    if len(extracted_bits) >= num_bits:
                        break
                    extracted_bits.append(self._extract_bit(dct_block[u, v]))
            
            if len(extracted_bits) >= num_bits:
                break
        
        return extracted_bits
    
    def _find_sync_marker(self, bits):
        marker_len = len(self.SYNC_MARKER)
        best_match = -1
        best_score = 0
        
        for i in range(len(bits) - marker_len + 1):
            segment = bits[i:i + marker_len]
            score = sum(1 for a, b in zip(segment, self.SYNC_MARKER) if a == b)
            if score > best_score:
                best_score = score
                best_match = i
        
        if best_score >= 12:  # Lower threshold for better resilience
            return best_match + marker_len, best_score
        return -1, best_score
    
    def extract(self, stego_image, num_bits):
        extra = 200
        raw_bits = self._extract_raw_bits(stego_image, num_bits + len(self.SYNC_MARKER) + extra)
        
        sync_end, score = self._find_sync_marker(raw_bits)
        
        if sync_end == -1:
            print(f"   ⚠️ Sync marker NOT found (best score: {score}/16)")
            # Try to recover by scanning for best partial match even if below threshold
            # and validate by extracting what we can
            data_bits = raw_bits[:num_bits]
        else:
            print(f"   ✅ Sync marker found (score: {score}/16)")
            data_bits = raw_bits[sync_end:sync_end + num_bits]
        
        while len(data_bits) < num_bits:
            data_bits.append(0)
        
        return data_bits[:num_bits]
    
    # ------------------------------------------------------------------
    # Robust pipeline: multi-quality-hypothesis extraction (see robust.py)
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

    def _extract_raw_bits_robust(self, stego_image, num_bits, table):
        """Extract raw parity bits from the crop-safe window using a quant grid."""
        from robust import band_cols, band_rows
        ycbcr = cv2.cvtColor(stego_image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        luminance = ycbcr[:, :, 0]
        h, w = luminance.shape
        r0, r1 = band_rows(h, self.block_size, self.config)
        c0, c1 = band_cols(w, self.block_size, self.config)
        skip_mod = self.config.ROBUST_SKIP_MOD

        extracted_bits = []
        for br in range(r0, r1):
            for bc in range(c0, c1):
                if len(extracted_bits) >= num_bits:
                    break
                if (br + bc) % skip_mod == 0:
                    continue
                r_start = br * self.block_size
                c_start = bc * self.block_size
                block = luminance[r_start:r_start + self.block_size,
                                  c_start:c_start + self.block_size]
                dct_block = self._apply_dct2(block)
                for (u, v) in self.positions_robust:
                    if len(extracted_bits) >= num_bits:
                        break
                    idx = int(round(dct_block[u, v] / table[u, v]))
                    extracted_bits.append(abs(idx) % 2)
        return extracted_bits[:num_bits]

    def _find_frames(self, raw, unit, enc_len):
        """Unit-spaced sync chains: the true frame is (almost always) the only
        long chain because the false 16-bit sync matches scatter randomly."""
        from robust import SYNC_MARKER
        marker_len = len(SYNC_MARKER)
        cands = []
        for i in range(len(raw) - marker_len):
            s = sum(1 for a, b in zip(raw[i:i + marker_len], SYNC_MARKER) if a == b)
            if s >= 12:
                cands.append((s, i))
        cands.sort(key=lambda c: (-c[0], c[1]))

        frames = []
        for sc, anchor in cands:
            chain = [anchor]
            pos = anchor + unit
            while len(chain) < self.config.ROBUST_K_COPIES:
                nxt = [p for (_, p) in cands if abs(p - pos) <= 3 and p not in chain]
                if not nxt:
                    break
                p = nxt[0]
                chain.append(p)
                pos = p + unit
            # Keep unit-spaced chains, plus lone very-strong candidates
            if len(chain) >= 2 or (len(chain) == 1 and sc >= 14):
                frames.append(chain)

        frames.sort(key=lambda ch: (-len(ch), -sum(s for (s, p) in cands if p in ch)))
        return frames

    def _majority_vote(self, copies):
        n = len(copies[0])
        voted = []
        for i in range(n):
            ones = sum(c[i] for c in copies)
            voted.append(1 if ones >= (len(copies) + 1) // 2 else 0)
        return voted

    def extract_robust(self, stego_image):
        """
        Recover the hidden message after arbitrary transforms.

        Tries every quantisation-grid hypothesis (the one matching the attack
        quality yields low BER), finds unit-spaced sync chains, majority-votes
        the repeated copies, RS-decodes and CRC-validates. Returns the message
        bit list, or None if no copy decodes.
        """
        from robust import (SYNC_MARKER, encoded_length_bits, parse_payload,
                            quant_table, score_text_bits)

        cfg = self.config
        enc_bits = encoded_length_bits(cfg)
        unit = enc_bits + len(SYNC_MARKER)
        nbits = cfg.ROBUST_K_COPIES * unit + 400
        payload_len = cfg.ROBUST_PAYLOAD_BYTES * 8
        padding_info = (payload_len, (payload_len, cfg.ROBUST_PAYLOAD_BYTES),
                        cfg.RS_INTERLEAVE_DEPTH)

        best_bits, best_score = None, -1
        for q in cfg.ROBUST_HYPOTHESIS_QUALITIES:
            table = quant_table(q)
            raw = self._extract_raw_bits_robust(stego_image, nbits, table)

            frames = self._find_frames(raw, unit, enc_bits)
            seen = set()
            for chain in frames[:16]:
                key = tuple(chain)
                if key in seen:
                    continue
                seen.add(key)

                copies = []
                for st in chain:
                    data = raw[st + len(SYNC_MARKER):st + len(SYNC_MARKER) + enc_bits]
                    if len(data) < enc_bits:
                        data = data + [0] * (enc_bits - len(data))
                    copies.append(list(data[:enc_bits]))

                voted = copies[0] if len(copies) == 1 else self._majority_vote(copies)
                try:
                    decoded_bits, _ = self._rs().decode_bitstream(voted, padding_info)
                except Exception:
                    continue

                payload = parse_payload(decoded_bits, cfg)
                if payload is None:
                    continue
                sc = score_text_bits(payload)
                if sc > best_score:
                    best_score, best_bits = sc, payload

        if best_bits is None:
            print("   ⚠️ Robust extraction failed: no CRC-valid decode found")
        return best_bits