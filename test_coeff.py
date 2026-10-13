"""Diagnostic - check coefficient survival for every position"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct


def test_coefficients():
    print("=" * 70)
    print("COEFFICIENT SURVIVAL TEST")
    print("=" * 70)
    
    # Load image
    img = cv2.imread("dataset/sample_images/cover.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lum = ycbcr[:, :, 0].copy()
    
    alpha = 2.0
    step = alpha * 10  # = 20.0
    
    positions = [
        (1, 2), (2, 1), (3, 0), (0, 3), (2, 2),
        (1, 3), (3, 1), (4, 0), (0, 4), (2, 3),
        (3, 2), (1, 4), (4, 1), (5, 0), (0, 5),
        (2, 4), (3, 3), (4, 2), (1, 5), (5, 1),
        (6, 0), (0, 6), (2, 5), (3, 4), (4, 3),
        (5, 2), (1, 6)
    ]
    
    # Track survival per position
    pos_survival = {pos: {'total': 0, 'survived': 0} for pos in positions}
    
    # Test on 100 random blocks
    np.random.seed(42)
    blocks_tested = 0
    
    for _ in range(200):
        br = np.random.randint(0, 60)
        bc = np.random.randint(0, 60)
        r, c = br * 8, bc * 8
        
        block = lum[r:r+8, c:c+8].copy()
        
        # Skip flat blocks
        if block.max() - block.min() < 15:
            continue
        
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Embed random bits and check survival
        for (u, v) in positions:
            original_coeff = dct_block[u, v]
            bit = np.random.randint(0, 2)
            
            # Embed
            quantized = np.round(original_coeff / step) * step
            idx = int(np.round(quantized / step))
            if abs(idx) % 2 != bit:
                if quantized >= 0:
                    quantized += step if bit == 1 else -step
                else:
                    quantized -= step if bit == 1 else -step
            dct_block[u, v] = quantized
            
            pos_survival[(u, v)]['total'] += 1
        
        # IDCT → uint8 → DCT
        reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        print(f"Reconstructed range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
        rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8).astype(np.float32)
        re_dct = dct(dct(rounded.T, norm='ortho').T, norm='ortho')
        
        # Check survival
        for (u, v) in positions:
            original_coeff = dct(dct(block.T, norm='ortho').T, norm='ortho')[u, v]
            bit = np.random.randint(0, 2)
            
            # Re-embed with same bit to compare (deterministic)
            quantized = np.round(original_coeff / step) * step
            idx = int(np.round(quantized / step))
            if abs(idx) % 2 != bit:
                if quantized >= 0:
                    quantized += step if bit == 1 else -step
                else:
                    quantized -= step if bit == 1 else -step
            
            # Extract from re_dct
            recovered_idx = int(np.round(re_dct[u, v] / step))
            recovered_bit = abs(recovered_idx) % 2
            
            if recovered_bit == bit:
                pos_survival[(u, v)]['survived'] += 1
        
        blocks_tested += 1
    
    print(f"\nTested {blocks_tested} blocks × 26 positions")
    print(f"\n{'Position':<12} {'Survival':<12} {'Rate':<10}")
    print("-" * 40)
    
    for pos in positions:
        stats = pos_survival[pos]
        if stats['total'] > 0:
            rate = stats['survived'] / stats['total'] * 100
            status = "✅" if rate > 95 else "⚠️" if rate > 70 else "❌"
            print(f"{str(pos):<12} {stats['survived']}/{stats['total']:<8} {rate:<9.1f}% {status}")


if __name__ == "__main__":
    test_coefficients()