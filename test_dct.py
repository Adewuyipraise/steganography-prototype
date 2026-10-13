"""Test which alpha value survives DCT→IDCT→DCT roundtrip"""

import numpy as np
from scipy.fftpack import dct, idct


def test_alpha(alpha, num_coeffs=26, num_trials=100):
    """Test how many bits survive at this alpha."""
    step = alpha * 10
    np.random.seed(42)
    
    total_bits = 0
    correct_bits = 0
    
    for trial in range(num_trials):
        # Random block
        block = np.random.randint(0, 255, (8, 8)).astype(np.float32)
        
        # DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Embed random bits in first num_coeffs positions
        positions = [(1,2), (2,1), (3,0), (0,3), (2,2),
                     (1,3), (3,1), (4,0), (0,4), (2,3),
                     (3,2), (1,4), (4,1), (5,0), (0,5),
                     (2,4), (3,3), (4,2), (1,5), (5,1),
                     (6,0), (0,6), (2,5), (3,4), (4,3),
                     (5,2)]
        
        bits_embedded = []
        for i, (u, v) in enumerate(positions[:num_coeffs]):
            bit = np.random.randint(0, 2)
            bits_embedded.append(bit)
            
            coeff = dct_block[u, v]
            quantized = np.round(coeff / step) * step
            idx = int(np.round(quantized / step))
            
            if abs(idx) % 2 != bit:
                if quantized >= 0:
                    quantized += step if bit == 1 else -step
                else:
                    quantized -= step if bit == 1 else -step
            
            dct_block[u, v] = quantized
        
        # IDCT → uint8 → float (simulates image storage)
        modified_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        modified_block = np.clip(np.round(modified_block), 0, 255).astype(np.uint8)
        modified_block = modified_block.astype(np.float32)
        
        # Re-DCT
        dct_recovered = dct(dct(modified_block.T, norm='ortho').T, norm='ortho')
        
        # Extract bits
        for i, (u, v) in enumerate(positions[:num_coeffs]):
            idx = int(np.round(dct_recovered[u, v] / step))
            extracted = abs(idx) % 2
            
            total_bits += 1
            if extracted == bits_embedded[i]:
                correct_bits += 1
    
    return correct_bits / total_bits


print("=" * 60)
print("ALPHA TEST: Which α survives DCT roundtrip?")
print("=" * 60)
print(f"{'Alpha':<10} {'Step':<10} {'Survival Rate':<15}")
print("-" * 40)

for alpha in [0.08, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    rate = test_alpha(alpha)
    status = "✅" if rate > 0.95 else "⚠️" if rate > 0.80 else "❌"
    print(f"{alpha:<10} {alpha*10:<10.1f} {rate*100:<14.1f}% {status}")

print("=" * 60)