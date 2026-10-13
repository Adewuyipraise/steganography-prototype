"""
Reed-Solomon Error Correction for Robust Steganography
FIXED VERSION - Handles decode errors properly
"""

from reedsolo import RSCodec, ReedSolomonError


class ReedSolomonCodec:
    """Reed-Solomon error correction codec - FIXED"""

    def __init__(self, data_symbols=191, ecc_symbols=64, symbol_size=8, verbose=True):
        self.data_symbols = data_symbols
        self.ecc_symbols = ecc_symbols
        self.symbol_size = symbol_size
        self.verbose = verbose
        self.rs = RSCodec(ecc_symbols)
        self.total_symbols = data_symbols + ecc_symbols
        self.max_errors = ecc_symbols // 2

        if self.verbose:
            print("🔐 Reed-Solomon initialized:")
            print(f"   Data symbols: {data_symbols}")
            print(f"   ECC symbols: {ecc_symbols}")
            print(f"   Max correctable errors per block: {self.max_errors}")

    def _bits_to_bytes(self, bits):
        """Convert bit list to bytes."""
        if len(bits) % 8 != 0:
            bits = bits + [0] * (8 - len(bits) % 8)

        bytes_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_list.append(byte)

        return bytes(bytes_list)

    def _bytes_to_bits(self, byte_data):
        """Convert bytes to bit list."""
        bits = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def encode_bitstream(self, bitstream):
        """Encode a bitstream using Reed-Solomon."""
        original_length = len(bitstream)

        data_bytes = self._bits_to_bytes(bitstream)
        encoded_bytes = self.rs.encode(data_bytes)
        encoded_bits = self._bytes_to_bits(encoded_bytes)

        padding_info = (original_length, len(data_bytes))

        if self.verbose:
            print(f"   RS Encode: {len(bitstream)} bits → {len(encoded_bits)} bits (+{len(encoded_bits)-len(bitstream)} bits)")

        return encoded_bits, padding_info

    def decode_bitstream(self, encoded_bitstream, padding_info):
        """Decode a Reed-Solomon encoded bitstream - FIXED."""
        original_length, data_bytes_count = padding_info

        encoded_bytes = self._bits_to_bytes(encoded_bitstream)

        try:
            # FIX: decode returns (decoded_bytes, decoded_bytes+ecc, errata_positions)
            # depending on the reedsolo version. Error count = len(errata_positions).
            result = self.rs.decode(encoded_bytes)

            if isinstance(result, tuple):
                decoded_bytes = result[0]
                if len(result) >= 3:
                    errata = result[2]
                    error_count = len(errata) if errata else 0
                elif len(result) >= 2:
                    # Older formats: second element is either positions or a count
                    second = result[1]
                    if isinstance(second, (list, bytearray)):
                        error_count = len(second) if second else 0
                    else:
                        error_count = 0
                else:
                    error_count = 0
            else:
                decoded_bytes = result
                error_count = 0

            if self.verbose:
                print(f"   RS Decode: {len(encoded_bytes)} bytes → {len(decoded_bytes)} bytes, errors corrected: {error_count}")

            decoded_bits = self._bytes_to_bits(decoded_bytes)

            if len(decoded_bits) > original_length:
                decoded_bits = decoded_bits[:original_length]
            elif len(decoded_bits) < original_length:
                decoded_bits = decoded_bits + [0] * (original_length - len(decoded_bits))

            return decoded_bits, error_count

        except ReedSolomonError as e:
            if self.verbose:
                print(f"   ❌ RS decode failed: {e}")
            # Return best effort - just trim to original length
            return encoded_bitstream[:original_length], -1
        except Exception as e:
            if self.verbose:
                print(f"   ❌ RS decode error: {e}")
            return encoded_bitstream[:original_length], -1


class TurboReedSolomon:
    """Turbo Reed-Solomon with interleaving - FIXED"""

    def __init__(self, data_symbols=127, ecc_symbols=128, interleave_depth=8, verbose=True):
        self.codec = ReedSolomonCodec(data_symbols, ecc_symbols, verbose=verbose)
        self.interleave_depth = interleave_depth

    def _interleave(self, bits, depth=None):
        if depth is None:
            depth = self.interleave_depth

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

    def _deinterleave(self, bits, depth=None):
        if depth is None:
            depth = self.interleave_depth

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

    def encode_bitstream(self, bitstream):
        """Encode with interleaving."""
        original_length = len(bitstream)

        encoded_bits, base_padding = self.codec.encode_bitstream(bitstream)
        interleaved_bits = self._interleave(encoded_bits, self.interleave_depth)

        padding_info = (original_length, base_padding, self.interleave_depth)

        return interleaved_bits, padding_info

    def decode_bitstream(self, encoded_bitstream, padding_info):
        """Decode with de-interleaving - FIXED."""
        original_length, base_padding, interleave_depth = padding_info

        deinterleaved_bits = self._deinterleave(encoded_bitstream, interleave_depth)
        decoded_bits, error_count = self.codec.decode_bitstream(deinterleaved_bits, base_padding)

        return decoded_bits, error_count


# Quick test
if __name__ == "__main__":
    print("Testing Reed-Solomon...")

    from text_utils import text_to_bits, bits_to_text

    test = "Hello World"
    bits = text_to_bits(test)
    print(f"Original: {test} ({len(bits)} bits)")

    rs = TurboReedSolomon(data_symbols=191, ecc_symbols=64, interleave_depth=6)
    encoded, padding = rs.encode_bitstream(bits)
    print(f"Encoded: {len(encoded)} bits")

    # Test with no errors
    decoded, errors = rs.decode_bitstream(encoded, padding)
    recovered = bits_to_text(decoded)
    print(f"Recovered: {recovered}, errors: {errors}")
    print(f"Match: {test == recovered}")
