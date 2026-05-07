"""Configuration parameters for the steganography system"""

# DCT Block Size
BLOCK_SIZE = 8

# Mid-frequency DCT coefficient pairs for differential embedding
# Each bit uses two coefficients (u1, v1) and (u2, v2)
COEFF_PAIRS = [
    ((1, 2), (2, 1)),
    ((3, 1), (1, 3)),
    ((2, 2), (3, 2)),
    ((2, 3), (4, 1))
]

# Mid-frequency DCT coefficient positions (for baseline comparisons)
MID_FREQ_POSITIONS = [
    (2, 1), (1, 2), (3, 1), (2, 2), (1, 3),
    (3, 2), (2, 3), (4, 1), (1, 4), (3, 3)
]

MAX_EMBEDDING_BITS_PER_BLOCK = len(COEFF_PAIRS)

# Embedding Parameters
EMBEDDING_STRENGTH_BASE = 15.0      # Margin for differential embedding
EDGE_STRENGTH_FACTOR = 0.5          # λ in methodology (0-1)

# Hamming (7,4) Parameters
HAMMING_DATA_BITS = 4
HAMMING_CODE_BITS = 7

# Transformation Test Parameters
JPEG_QUALITIES = [10, 30, 50, 70, 90]
RESIZE_FACTORS = [0.5, 0.75, 1.25, 1.5]
CROP_PERCENTS = [0.1, 0.2, 0.3, 0.4]
NOISE_VARIANCES = [0.001, 0.005, 0.01, 0.02, 0.05]
BLUR_KERNELS = [(3, 3), (5, 5)]

# RC6 Encryption (Optional - Simplified for simulation)
USE_ENCRYPTION = False
RC6_ROUNDS = 20
