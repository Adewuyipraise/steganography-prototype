# test_bypass.py
import numpy as np
import cv2
from scipy.fftpack import dct, idct

# Load cover image
img = cv2.imread("dataset/sample_images/cover.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to YCbCr with float32
ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
lum = ycbcr[:, :, 0].copy()

print(f"Input luminance dtype: {lum.dtype}")
print(f"Input luminance range: [{lum.min():.2f}, {lum.max():.2f}]")

# Take first block
block = lum[0:8, 0:8].copy()
print(f"\nOriginal block[0:8, 0:8]:\n{block}")

# DCT
dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
print(f"\nDCT block:\n{dct_block}")

# Modify one coefficient
alpha = 2.0
step = alpha * 10
u, v = 1, 2
original_coeff = dct_block[u, v]
print(f"\nOriginal coefficient [{u},{v}]: {original_coeff:.4f}")

# Embed bit=1
quantized = np.round(original_coeff / step) * step
idx = int(np.round(quantized / step))
if abs(idx) % 2 != 1:
    if quantized >= 0:
        quantized += step
    else:
        quantized -= step
dct_block[u, v] = quantized
print(f"Modified coefficient [{u},{v}]: {quantized:.4f}")

# IDCT
reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
print(f"\nReconstructed block:\n{reconstructed}")
print(f"Reconstructed dtype: {reconstructed.dtype}")
print(f"Reconstructed range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")

# Round to uint8
rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
print(f"\nAfter rounding to uint8:\n{rounded}")

# Re-DCT
back = rounded.astype(np.float32)
dct_back = dct(dct(back.T, norm='ortho').T, norm='ortho')
print(f"\nRe-DCT coefficient [{u},{v}]: {dct_back[u, v]:.4f}")
print(f"Expected: {quantized:.4f}")
print(f"Difference: {abs(dct_back[u, v] - quantized):.4f}")

# Extract bit
idx2 = int(np.round(dct_back[u, v] / step))
extracted = abs(idx2) % 2
print(f"\nExtracted bit: {extracted} (expected: 1)")
print(f"Match: {extracted == 1}")