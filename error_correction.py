"""Standard Systematic Hamming (7,4) Error Correction Code"""

import numpy as np

class Hamming74:
    """
    Standard systematic Hamming (7,4) code.
    Data bits are at indices 0, 1, 2, 3.
    Parity bits are at indices 4, 5, 6.
    """
    
    # Generator matrix G = [I | P]
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)
    
    # Parity check matrix H = [P^T | I]
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ], dtype=int)
    
    HAMMING_DATA_BITS = 4
    HAMMING_CODE_BITS = 7

    @classmethod
    def encode(cls, data_bits):
        data = np.array(data_bits, dtype=int)
        code = np.mod(np.dot(data, cls.G), 2)
        return code.tolist()
    
    @classmethod
    def decode(cls, received_bits):
        received = np.array(received_bits, dtype=int)
        syndrome = np.mod(np.dot(cls.H, received), 2)
        syndrome_int = sum(syndrome[i] * (1 << (2 - i)) for i in range(3))
        
        # Syndrome mapping to error position (1-indexed based on H columns)
        # H columns:
        # col 0: [1,1,0] -> 6
        # col 1: [1,0,1] -> 5
        # col 2: [0,1,1] -> 3
        # col 3: [1,1,1] -> 7
        # col 4: [1,0,0] -> 4
        # col 5: [0,1,0] -> 2
        # col 6: [0,0,1] -> 1
        
        # Mapping syndrome_int to 0-indexed position in codeword
        syndrome_to_pos = {
            6: 0, 5: 1, 3: 2, 7: 3, 4: 4, 2: 5, 1: 6
        }
        
        corrected = False
        error_pos = -1
        if syndrome_int != 0:
            error_pos = syndrome_to_pos.get(syndrome_int, -1)
            if error_pos != -1:
                received[error_pos] ^= 1
                corrected = True
        
        # Data bits are the first 4 bits in systematic form
        return received[:4].tolist(), corrected, error_pos
    
    @classmethod
    def encode_bitstream(cls, bitstream):
        encoded = []
        padding = (4 - len(bitstream) % 4) % 4
        bitstream_padded = bitstream + [0] * padding
        padding_info = (padding, 0)
        
        for i in range(0, len(bitstream_padded), 4):
            encoded.extend(cls.encode(bitstream_padded[i:i+4]))
        return encoded, padding_info
    
    @classmethod
    def decode_bitstream(cls, encoded_bitstream, padding_info):
        decoded = []
        error_count = 0
        for i in range(0, len(encoded_bitstream), 7):
            if i + 7 > len(encoded_bitstream): break
            data, corrected, _ = cls.decode(encoded_bitstream[i:i+7])
            decoded.extend(data)
            if corrected: error_count += 1
        
        padding, _ = padding_info
        if padding > 0: decoded = decoded[:-padding]
        return decoded, error_count
