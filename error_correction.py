"""Standard Systematic Hamming (7,4) Error Correction Code"""

import numpy as np

class Hamming74:
    """
    Hamming (7,4) linear error-correcting code.
    Chapter 3 Algorithm 3.3 and 3.5
    Encodes 4 data bits into 7 code bits, corrects single-bit errors.
    """
    
    HAMMING_DATA_BITS = 4
    HAMMING_CODE_BITS = 7
    
    @classmethod
    def encode(cls, data_bits):
        """
        Encode 4 data bits into 7 code bits.
        Chapter 3 Algorithm 3.3
        
        Data bits: [b0, b1, b2, b3]
        Parity bits:
            p0 = b0 ⊕ b1 ⊕ b2
            p1 = b0 ⊕ b1 ⊕ b3
            p2 = b0 ⊕ b2 ⊕ b3
        Codeword: [b0, b1, b2, b3, p0, p1, p2]
        """
        assert len(data_bits) == 4, f"Need 4 bits, got {len(data_bits)}"
        b0, b1, b2, b3 = data_bits
        
        # Compute parity bits (Algorithm 3.3)
        p0 = b0 ^ b1 ^ b2
        p1 = b0 ^ b1 ^ b3
        p2 = b0 ^ b2 ^ b3
        
        # Form 7-bit codeword
        codeword = [b0, b1, b2, b3, p0, p1, p2]
        return codeword
    
    @classmethod
    def decode(cls, received_bits):
        """
        Decode 7 received bits, correct single-bit error.
        Chapter 3 Algorithm 3.5
        
        Args:
            received_bits: List of 7 bits [c0,c1,c2,c3,c4,c5,c6]
            
        Returns:
            Tuple: (decoded_4_bits, error_corrected_flag, error_position)
        """
        assert len(received_bits) == 7, f"Need 7 bits, got {len(received_bits)}"
        c0, c1, c2, c3, c4, c5, c6 = received_bits
        
        # Extract data bits
        d0, d1, d2, d3 = c0, c1, c2, c3
        
        # Recompute parity bits (Algorithm 3.5)
        p0_prime = d0 ^ d1 ^ d2
        p1_prime = d0 ^ d1 ^ d3
        p2_prime = d0 ^ d2 ^ d3
        
        # Compute syndrome
        s0 = p0_prime ^ c4
        s1 = p1_prime ^ c5
        s2 = p2_prime ^ c6
        syndrome = (s0, s1, s2)
        
        # Map syndrome to error position (Table 3.5 in thesis)
        # Syndrome (s0,s1,s2) → bit position to flip
        syndrome_map = {
            (0, 0, 0): -1,   # No error
            (1, 0, 0): 4,    # c4 (p0)
            (0, 1, 0): 5,    # c5 (p1)
            (0, 0, 1): 6,    # c6 (p2)
            (1, 1, 0): 1,    # c1 (b1)
            (1, 0, 1): 2,    # c2 (b2)
            (0, 1, 1): 3,    # c3 (b3)
            (1, 1, 1): 0     # c0 (b0)
        }
        
        error_pos = syndrome_map.get(syndrome, -1)
        error_corrected = False
        
        if error_pos != -1:
            # Flip the erroneous bit
            corrected = received_bits.copy()
            corrected[error_pos] ^= 1
            error_corrected = True
            # Extract corrected data bits
            decoded = [corrected[0], corrected[1], corrected[2], corrected[3]]
        else:
            # No error
            decoded = [d0, d1, d2, d3]
        
        return decoded, error_corrected, error_pos
    
    @classmethod
    def encode_bitstream(cls, bitstream):
        """
        Encode a long bitstream using Hamming (7,4).
        Chapter 3 Algorithm 3.3 with interleaving
        
        Args:
            bitstream: List of bits
            
        Returns:
            Encoded bitstream, padding_info
        """
        # Pad to multiple of 4
        padding = (4 - len(bitstream) % 4) % 4
        bitstream_padded = bitstream + [0] * padding
        padding_info = (padding, bitstream[-1] if bitstream else 0)
        
        encoded = []
        for i in range(0, len(bitstream_padded), 4):
            data_block = bitstream_padded[i:i+4]
            code_block = cls.encode(data_block)
            encoded.extend(code_block)
        
        # Apply interleaving (depth = 5 as per Chapter 3)
        encoded = cls._interleave(encoded, depth=5)
        
        return encoded, padding_info
    
    @classmethod
    def decode_bitstream(cls, encoded_bitstream, padding_info):
        """
        Decode a Hamming-encoded bitstream.
        Chapter 3 Algorithm 3.5 with de-interleaving
        
        Args:
            encoded_bitstream: List of encoded bits
            padding_info: (padding_bits, last_bit_value) from encoding
            
        Returns:
            Decoded bitstream, error_count
        """
        # De-interleave first
        deinterleaved = cls._deinterleave(encoded_bitstream, depth=5)
        
        decoded = []
        error_count = 0
        
        for i in range(0, len(deinterleaved), 7):
            if i + 7 > len(deinterleaved):
                break
            code_block = deinterleaved[i:i+7]
            data_block, corrected, _ = cls.decode(code_block)
            decoded.extend(data_block)
            if corrected:
                error_count += 1
        
        # Remove padding
        padding, _ = padding_info
        if padding > 0:
            decoded = decoded[:-padding]
        
        return decoded, error_count
    
    @classmethod
    def _interleave(cls, bits, depth=5):
        """Interleave bits to distribute burst errors (Chapter 3)"""
        # Simple block interleaving
        if len(bits) < depth:
            return bits
        
        # Reshape into rows of depth
        n = len(bits)
        padded = bits + [0] * ((depth - n % depth) % depth)
        rows = len(padded) // depth
        matrix = [padded[i*depth:(i+1)*depth] for i in range(rows)]
        
        # Read column-wise
        interleaved = []
        for col in range(depth):
            for row in range(rows):
                interleaved.append(matrix[row][col])
        
        return interleaved
    
    @classmethod
    def _deinterleave(cls, bits, depth=5):
        """De-interleave bits (reverse of interleaving)"""
        if len(bits) < depth:
            return bits
        
        n = len(bits)
        cols = depth
        rows = n // depth
        matrix = [[0] * cols for _ in range(rows)]
        
        # Fill column-wise
        idx = 0
        for col in range(cols):
            for row in range(rows):
                if idx < n:
                    matrix[row][col] = bits[idx]
                    idx += 1
        
        # Read row-wise
        deinterleaved = []
        for row in range(rows):
            deinterleaved.extend(matrix[row])
        
        return deinterleaved
