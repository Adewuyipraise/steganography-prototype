"""RAW test - no classes, no abstraction, just math"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct


def test_raw():
    print("=" * 60)
    print("RAW DCT TEST ON cover.jpg")
    print("=" * 60)
    
    # Load image
    img = cv2.imread("dataset/sample_images/cover.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {img.shape}")
    
    # Convert to YCbCr float
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lum = ycbcr[:, :, 0].copy()
    print(f"Luminance shape: {lum.shape}")
    
    # Test on block (30, 30) — safe within 512x512
    br, bc = 30, 30
    r, c = br * 8, bc * 8
    print(f"\nBlock position: ({br}, {bc}), pixel offset ({r}, {c})")
    
    block = lum[r:r+8, c:c+8].copy()
    
    print(f"Block shape: {block.shape}")
    print(f"Block range: [{block.min():.2f}, {block.max():.2f}]")
    print(f"Block mean: {block.mean():.2f}")
    
    # DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    print(f"\nDCT[0,0] (DC): {dct_block[0,0]:.4f}")
    print(f"DCT[1,2]: {dct_block[1,2]:.4f}")
    print(f"DCT[2,3]: {dct_block[2,3]:.4f}")
    
    # Save originals
    orig_coeff_12 = dct_block[1, 2]
    orig_coeff_23 = dct_block[2, 3]
    
    # Embed: bit 1 into [1,2], bit 0 into [2,3]
    alpha = 2.0
    step = alpha * 10  # = 20
    
    # Embed bit 1 into [1,2]
    target_bit_12 = 1
    quantized_12 = np.round(orig_coeff_12 / step) * step
    idx_12 = int(np.round(quantized_12 / step))
    if abs(idx_12) % 2 != target_bit_12:
        if quantized_12 >= 0:
            quantized_12 += step
        else:
            quantized_12 -= step
    dct_block[1, 2] = quantized_12
    
    # Embed bit 0 into [2,3]
    target_bit_23 = 0
    quantized_23 = np.round(orig_coeff_23 / step) * step
    idx_23 = int(np.round(quantized_23 / step))
    if abs(idx_23) % 2 != target_bit_23:
        if quantized_23 >= 0:
            quantized_23 -= step
        else:
            quantized_23 += step
    dct_block[2, 3] = quantized_23
    
    print(f"\nAfter embedding:")
    print(f"  Coeff[1,2] changed from {orig_coeff_12:.4f} to {quantized_12:.4f}")
    print(f"  Coeff[2,3] changed from {orig_coeff_23:.4f} to {quantized_23:.4f}")
    
    # IDCT
    reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    print(f"\nReconstructed block range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
    
    # Clip to 0-255 (what save to PNG does)
    clipped = np.clip(reconstructed, 0, 255)
    rounded = np.round(clipped).astype(np.uint8)
    back_to_float = rounded.astype(np.float32)
    
    print(f"Rounded block dtype: {rounded.dtype}")
    print(f"Rounded block range: [{rounded.min()}, {rounded.max()}]")
    
    # Re-DCT from the reconstructed
    re_dct = dct(dct(back_to_float.T, norm='ortho').T, norm='ortho')
    
    recovered_12 = re_dct[1, 2]
    recovered_23 = re_dct[2, 3]
    
    print(f"\nRe-DCT coefficients:")
    print(f"  Coeff[1,2]: {recovered_12:.4f} (expected: {quantized_12:.4f})")
    print(f"  Coeff[2,3]: {recovered_23:.4f} (expected: {quantized_23:.4f})")
    
    print(f"\n  Difference [1,2]: {abs(recovered_12 - quantized_12):.4f}")
    print(f"  Difference [2,3]: {abs(recovered_23 - quantized_23):.4f}")
    
    # Extract bits
    idx_extracted_12 = int(np.round(recovered_12 / step))
    bit_extracted_12 = abs(idx_extracted_12) % 2
    
    idx_extracted_23 = int(np.round(recovered_23 / step))
    bit_extracted_23 = abs(idx_extracted_23) % 2
    
    print(f"\nExtracted bits:")
    print(f"  Bit [1,2]: {bit_extracted_12} (expected: {target_bit_12}) {'✅' if bit_extracted_12 == target_bit_12 else '❌'}")
    print(f"  Bit [2,3]: {bit_extracted_23} (expected: {target_bit_23}) {'✅' if bit_extracted_23 == target_bit_23 else '❌'}")
    
    # Also test on a FLAT region (top-left corner is usually sky)
    print("\n" + "=" * 60)
    print("FLAT REGION TEST (block 0,0)")
    print("=" * 60)
    
    flat_block = lum[0:8, 0:8].copy()
    print(f"Flat block range: [{flat_block.min():.2f}, {flat_block.max():.2f}]")
    print(f"Flat block mean: {flat_block.mean():.2f}")
    
    flat_dct = dct(dct(flat_block.T, norm='ortho').T, norm='ortho')
    print(f"Flat DCT[0,0]: {flat_dct[0,0]:.4f}")
    print(f"Flat DCT[1,2]: {flat_dct[1,2]:.4f}")


if __name__ == "__main__":
    test_raw()