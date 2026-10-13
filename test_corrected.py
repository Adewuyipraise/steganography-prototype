"""Corrected coefficient survival test"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct


def test_corrected():
    print("=" * 70)
    print("CORRECTED COEFFICIENT SURVIVAL TEST")
    print("=" * 70)
    
    img = cv2.imread("dataset/sample_images/cover.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lum = ycbcr[:, :, 0].copy()
    
    alpha = 2.0
    step = alpha * 10
    
    positions = [
        (1, 2), (2, 1), (3, 0), (0, 3), (2, 2),
        (1, 3), (3, 1), (4, 0), (0, 4), (2, 3),
        (3, 2), (1, 4), (4, 1), (5, 0), (0, 5),
        (2, 4), (3, 3), (4, 2), (1, 5), (5, 1),
        (6, 0), (0, 6), (2, 5), (3, 4), (4, 3),
        (5, 2), (1, 6)
    ]
    
    np.random.seed(42)
    pos_survival = {pos: {'total': 0, 'survived': 0} for pos in positions}
    blocks_tested = 0
    
    for _ in range(200):
        br = np.random.randint(0, 60)
        bc = np.random.randint(0, 60)
        r, c = br*8, bc*8
        block = lum[r:r+8, c:c+8].copy()
        
        if block.max() - block.min() < 15:
            continue
        
        # ⚠️ CRITICAL: Store bits so embed and check use SAME bit
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Generate bits ONCE
        bits = {pos: np.random.randint(0, 2) for pos in positions}
        
        # Embed using stored bits
        for (u, v) in positions:
            coeff = dct_block[u, v]
            bit = bits[(u, v)]
            
            quantized = np.round(coeff / step) * step
            idx = int(np.round(quantized / step))
            if abs(idx) % 2 != bit:
                if quantized >= 0:
                    quantized += step if bit == 1 else -step
                else:
                    quantized -= step if bit == 1 else -step
            dct_block[u, v] = quantized
        
        # Round-trip
        reconstructed = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8).astype(np.float32)
        re_dct = dct(dct(rounded.T, norm='ortho').T, norm='ortho')
        
        # Check using SAME bits
        for (u, v) in positions:
            bit = bits[(u, v)]
            extracted_idx = int(np.round(re_dct[u, v] / step))
            extracted_bit = abs(extracted_idx) % 2
            
            if extracted_bit == bit:
                pos_survival[(u, v)]['survived'] += 1
            pos_survival[(u, v)]['total'] += 1
        
        blocks_tested += 1
    
    print(f"\nTested {blocks_tested} blocks × 26 positions")
    print(f"\n{'Position':<12} {'Survival':<14} {'Rate':<10}")
    print("-" * 40)
    
    total_survived = 0
    total_tested = 0
    
    for pos in positions:
        stats = pos_survival[pos]
        if stats['total'] > 0:
            rate = stats['survived'] / stats['total'] * 100
            status = "✅" if rate > 95 else "⚠️" if rate > 70 else "❌"
            print(f"{str(pos):<12} {stats['survived']}/{stats['total']:<10} {rate:<9.1f}% {status}")
            total_survived += stats['survived']
            total_tested += stats['total']
    
    print(f"\n{'TOTAL':<12} {total_survived}/{total_tested:<10} {total_survived/total_tested*100:.1f}%")
    
    # Show worst blocks
    print(f"\n🎯 Recommendation:")
    good_positions = [pos for pos in positions 
                     if pos_survival[pos]['total'] > 0 
                     and pos_survival[pos]['survived']/pos_survival[pos]['total'] > 0.9]
    print(f"   Positions with >90% survival: {len(good_positions)}/26")
    print(f"   {good_positions}")


if __name__ == "__main__":
    test_corrected()