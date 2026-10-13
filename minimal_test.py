"""Minimal test to debug recovery failure"""
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import config
from text_utils import text_to_bits, bits_to_text
from error_correction import Hamming74
from embedding import DCTEmbedder
from extraction import DCTExtractor

# Create a dummy cover image (512x512 RGB gradient)
cover_img = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(512):
    cover_img[i, :, 0] = int(128 + 127 * np.sin(i / 50))
    cover_img[i, :, 1] = int(128 + 127 * np.cos(i / 40))
    cover_img[:, i, 2] = int(128 + 127 * np.sin(i / 30))

# Simple test message
test_msg = "Hello world, this is a test!"
print(f"[1] Original text: {test_msg}")

original_bits = text_to_bits(test_msg)
print(f"[2] Original bits length: {len(original_bits)}")

# Encode with Hamming
encoded_bits, original_length = Hamming74.encode_bitstream(original_bits)
print(f"[3] Encoded bits length: {len(encoded_bits)}")

# Initialize embedder
embedder = DCTEmbedder(config)
stego_img, embedding_map = embedder.embed(cover_img, encoded_bits)

# Save stego image for inspection
cv2.imwrite("test_stego.png", cv2.cvtColor(stego_img, cv2.COLOR_RGB2BGR))

print("\n[4] Now extracting...")
# Extract
extractor = DCTExtractor(config)
extracted_encoded = extractor.extract(stego_img, len(encoded_bits))
print(f"[5] Extracted encoded bits length: {len(extracted_encoded)}")

# Let's compare the first 50 bits of encoded vs extracted
print("\n[6] Comparing first 50 bits of encoded vs extracted:")
for i in range(50):
    print(f"  bit {i}: encoded={encoded_bits[i]}, extracted={extracted_encoded[i]} {'✅' if encoded_bits[i]==extracted_encoded[i] else '❌'}")

# Decode
decoded_bits, error_count = Hamming74.decode_bitstream(extracted_encoded, original_length)
print(f"\n[7] Decoded bits length: {len(decoded_bits)}, errors corrected: {error_count}")

# Convert back to text
recovered_text = bits_to_text(decoded_bits)
print(f"\n[8] Recovered text: {recovered_text}")
print(f"[9] Text matches original: {test_msg == recovered_text}")

print("\nDone!")
