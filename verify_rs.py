"""Verify Reed-Solomon works on top of working DCT pipeline"""

import numpy as np
import cv2
from embedding import DCTEmbedder
from extraction import DCTExtractor
from reed_solomon import TurboReedSolomon
import config


def bits_from_text(text):
    bits = []
    for char in text:
        for i in range(7, -1, -1):
            bits.append((ord(char) >> i) & 1)
    return bits


def text_from_bits(bits):
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i+8]
        val = 0
        for b in byte:
            val = (val << 1) | b
        if 32 <= val <= 126:
            chars.append(chr(val))
    return ''.join(chars)


def test_rs():
    print("=" * 60)
    print("REED-SOLOMON VERIFICATION")
    print("=" * 60)
    
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    message = "Hello World, this is a test!"
    
    original_bits = bits_from_text(message)
    print(f"\n📝 Message: '{message}'")
    print(f"📊 Original bits: {len(original_bits)}")
    
    # Encode with Reed-Solomon
    rs = TurboReedSolomon(data_symbols=191, ecc_symbols=64, interleave_depth=6)
    encoded_bits, padding = rs.encode_bitstream(original_bits)
    print(f"📊 RS-encoded bits: {len(encoded_bits)}")
    
    # Embed
    embedder = DCTEmbedder(config)
    stego, _ = embedder.embed(img, encoded_bits)
    cv2.imwrite("verify_rs_stego.png", cv2.cvtColor(stego, cv2.COLOR_RGB2BGR))
    
    # Extract
    extractor = DCTExtractor(config)
    extracted = extractor.extract(stego, len(encoded_bits))
    
    # Decode with Reed-Solomon
    decoded, errors = rs.decode_bitstream(extracted, padding)
    recovered = text_from_bits(decoded)
    
    print(f"\n📊 Errors corrected: {errors}")
    print(f"📝 Recovered: '{recovered}'")
    
    if message == recovered:
        print("\n✅ FULL PIPELINE WITH RS WORKS!")
        return True
    else:
        print("\n⚠️ Partial recovery. Check output.")
        return False


if __name__ == "__main__":
    test_rs()