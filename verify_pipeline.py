"""Verify the pipeline works step by step"""

import numpy as np
import cv2
from embedding import DCTEmbedder
from extraction import DCTExtractor
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


def test_pipeline():
    print("=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Create test image
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # Test message
    message = "Hello World"
    bits = bits_from_text(message)
    
    print(f"\n📝 Message: '{message}'")
    print(f"📊 Bits: {len(bits)}")
    
    # Embed
    embedder = DCTEmbedder(config)
    stego, _ = embedder.embed(img, bits)
    
    # Save stego for inspection
    cv2.imwrite("verify_stego.png", cv2.cvtColor(stego, cv2.COLOR_RGB2BGR))
    
    # Extract
    extractor = DCTExtractor(config)
    extracted = extractor.extract(stego, len(bits))
    
    # Compare
    errors = sum(1 for i in range(min(len(bits), len(extracted))) if bits[i] != extracted[i])
    ber = errors / len(bits) if bits else 1.0
    
    print(f"\n📊 ORIGINAL:  {bits[:20]}...")
    print(f"📊 EXTRACTED: {extracted[:20]}...")
    print(f"📊 Errors: {errors}/{len(bits)}")
    print(f"📊 BER: {ber:.4f}")
    
    recovered = text_from_bits(extracted)
    print(f"\n📝 Recovered: '{recovered}'")
    
    if ber < 0.01:
        print("\n✅ PASS! Pipeline works.")
        return True
    else:
        print(f"\n❌ FAIL! BER = {ber:.4f}")
        return False


if __name__ == "__main__":
    test_pipeline()