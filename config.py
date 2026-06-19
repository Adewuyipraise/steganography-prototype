"""Configuration parameters for the steganography system"""

# DCT Block Size
BLOCK_SIZE = 8

# Embedding Parameters (Chapter 3 Section 3.4.3)
EMBEDDING_STRENGTH_EDGE = 0.08      # α_edge - for edge blocks
EMBEDDING_STRENGTH_SMOOTH = 0.03    # α_smooth - for smooth blocks
EDGE_DENSITY_THRESHOLD = 0.30       # 30% edge pixels = edge block

# Mid-frequency DCT coefficient positions (zigzag indices 10-35)
# As specified in Chapter 3 Section 3.4.3 Algorithm 3.2
MID_FREQ_POSITIONS = [
    (1, 2), (2, 1), (3, 0), (0, 3), (2, 2),  # Zigzag positions 10-14
    (1, 3), (3, 1), (4, 0), (0, 4), (2, 3),  # Positions 15-19
    (3, 2), (1, 4), (4, 1), (5, 0), (0, 5),  # Positions 20-24
    (2, 4), (3, 3), (4, 2), (1, 5), (5, 1),  # Positions 25-29
    (6, 0), (0, 6), (2, 5), (3, 4), (4, 3),  # Positions 30-34
    (5, 2), (1, 6)                           # Position 35 (total 26 coefficients)
]

# Hamming (7,4) Parameters
HAMMING_DATA_BITS = 4
HAMMING_CODE_BITS = 7

# Transformation Test Parameters (Chapter 3 Section 3.5)
JPEG_QUALITIES = [10, 30, 50, 70, 90]  # Table 3.3
RESIZE_FACTORS = [0.5, 0.75, 1.25, 1.5]  # Table 3.4
CROP_PERCENTS = [0.10, 0.20, 0.30, 0.40]  # Table 3.5
NOISE_VARIANCES = [0.01, 0.02, 0.03, 0.04, 0.05]  # Table 3.6

# Combined Transformation Sequences (Chapter 3 Section 3.5.5 - Table 3.7)
COMBINED_SEQUENCES = {
    'C1': {'order': ['resize', 'jpeg'], 'params': {'resize': 0.75, 'jpeg': 50}},
    'C2': {'order': ['jpeg', 'resize'], 'params': {'jpeg': 50, 'resize': 0.75}},
    'C3': {'order': ['resize', 'crop', 'jpeg'], 'params': {'resize': 0.75, 'crop': 0.20, 'jpeg': 50}},
    'C4': {'order': ['jpeg', 'crop', 'noise'], 'params': {'jpeg': 50, 'crop': 0.20, 'noise': 0.02}}
}

# Dataset Settings
RANDOM_SEED = 42  # For reproducibility
TUNING_SET_SIZE = 200
VALIDATION_SET_SIZE = 50
TEST_SET_SIZE = 50
