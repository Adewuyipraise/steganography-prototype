"""Hamming (7,4) Error Correction - WITH INTERLEAVING FOR BURST ERRORS"""

import numpy as np

class Hamming74:
    """Hamming (7,4) with interleaving to combat burst errors"""
    
    HAMMING_DATA_BITS = 4
    HAMMING_CODE_BITS = 7
    INTERLEAVING_DEPTH = 12  # Interleave across more blocks for better burst error protection
    
    @classmethod
    def encode(cls, data_bits):
        assert len(data_bits) == 4
        b0, b1, b2, b3 = data_bits
        p0 = b0 ^ b1 ^ b2
        p1 = b0 ^ b1 ^ b3
        p2 = b0 ^ b2 ^ b3
        return [b0, b1, b2, b3, p0, p1, p2]
    
    @classmethod
    def decode(cls, received_bits):
        assert len(received_bits) == 7
        c0, c1, c2, c3, c4, c5, c6 = received_bits
        d0, d1, d2, d3 = c0, c1, c2, c3
        
        p0_prime = d0 ^ d1 ^ d2
        p1_prime = d0 ^ d1 ^ d3
        p2_prime = d0 ^ d2 ^ d3
        
        s0 = p0_prime ^ c4
        s1 = p1_prime ^ c5
        s2 = p2_prime ^ c6
        
        syndrome_map = {
            (0, 0, 0): -1,
            (1, 0, 0): 4,
            (0, 1, 0): 5,
            (0, 0, 1): 6,
            (1, 1, 0): 1,
            (1, 0, 1): 2,
            (0, 1, 1): 3,
            (1, 1, 1): 0
        }
        
        error_pos = syndrome_map.get((s0, s1, s2), -1)
        error_corrected = False
        corrected = received_bits.copy()
        
        if error_pos != -1:
            corrected[error_pos] ^= 1
            error_corrected = True
            decoded = [corrected[0], corrected[1], corrected[2], corrected[3]]
        else:
            decoded = [d0, d1, d2, d3]
        
        return decoded, error_corrected, error_pos, corrected
    
    @classmethod
    def encode_bitstream(cls, bitstream):
        """Encode with bit interleaving to fight burst errors"""
        original_length = len(bitstream)
        padding = (4 - len(bitstream) % 4) % 4
        bitstream_padded = bitstream + [0] * padding
        
        # Encode each 4-bit block to 7-bit codeword
        encoded = []
        for i in range(0, len(bitstream_padded), 4):
            data_block = bitstream_padded[i:i+4]
            code_block = cls.encode(data_block)
            encoded.extend(code_block)
        
        # Apply interleaving (spread bits across blocks)
        encoded = cls._interleave(encoded, depth=cls.INTERLEAVING_DEPTH)
        
        return encoded, original_length
    
    @classmethod
    def decode_bitstream(cls, encoded_bitstream, original_length):
        """Decode with de-interleaving"""
        # De-interleave first
        deinterleaved = cls._deinterleave(encoded_bitstream, depth=cls.INTERLEAVING_DEPTH)
        
        decoded = []
        error_count = 0
        
        for i in range(0, len(deinterleaved), 7):
            if i + 7 > len(deinterleaved):
                break
            code_block = deinterleaved[i:i+7]
            data_block, corrected, _, _ = cls.decode(code_block)
            decoded.extend(data_block)
            if corrected:
                error_count += 1
        
        # Remove padding
        if len(decoded) > original_length:
            decoded = decoded[:original_length]
        elif len(decoded) < original_length:
            decoded = decoded + [0] * (original_length - len(decoded))
        
        print(f"   Errors corrected (with interleaving): {error_count}")
        
        return decoded, error_count
    
    @classmethod
    def _interleave(cls, bits, depth=8):
        """Block interleaving to spread burst errors"""
        if len(bits) < depth:
            return bits
        
        n = len(bits)
        padded = bits + [0] * ((depth - n % depth) % depth)
        rows = len(padded) // depth
        matrix = [padded[i*depth:(i+1)*depth] for i in range(rows)]
        
        interleaved = []
        for col in range(depth):
            for row in range(rows):
                interleaved.append(matrix[row][col])
        
        return interleaved
    
    @classmethod
    def _deinterleave(cls, bits, depth=8):
        """De-interleave bits"""
        if len(bits) < depth:
            return bits
        
        n = len(bits)
        cols = depth
        rows = n // depth
        matrix = [[0] * cols for _ in range(rows)]
        
        idx = 0
        for col in range(cols):
            for row in range(rows):
                if idx < n:
                    matrix[row][col] = bits[idx]
                    idx += 1
        
        deinterleaved = []
        for row in range(rows):
            deinterleaved.extend(matrix[row])
        
        return deinterleaved
