"""Configuration parameters - FINAL WORKING VERSION"""

BLOCK_SIZE = 8

# CRITICAL: Use conservative alpha for better round-trip reliability on smooth images
# Tuned for both textured and smooth content
EMBEDDING_STRENGTH_EDGE = 0.26
EMBEDDING_STRENGTH_SMOOTH = 0.26
EDGE_DENSITY_THRESHOLD = 0.30

# Mid-frequency DCT coefficient positions (zigzag indices 10-35)
MID_FREQ_POSITIONS = [
    (1, 2), (2, 1), (3, 0), (0, 3), (2, 2),
    (1, 3), (3, 1), (4, 0), (0, 4), (2, 3),
    (3, 2), (1, 4), (4, 1), (5, 0), (0, 5),
    (2, 4), (3, 3), (4, 2), (1, 5), (5, 1),
    (6, 0), (0, 6), (2, 5), (3, 4), (4, 3),
    (5, 2), (1, 6)
]

# Reed-Solomon Parameters (high-ECC multi-codeword: reedsolo auto-chunks data)
# Corrects up to 192/2 = 96 byte errors per 255-byte codeword -> survives ~5-7% per-bit noise
RS_DATA_SYMBOLS = 63
RS_ECC_SYMBOLS = 192
RS_INTERLEAVE_DEPTH = 8

# =====================================================================
# Robust pipeline: JPEG-guard grid embedding + repetition + validated sync
# (fixes text recovery after JPEG compression, resize and cropping)
# =====================================================================

# Parity magnitude per coefficient = the libjpeg luminance quantisation step
# at this "guard" quality, so embedded values sit exactly on the grid JPEG
# requantises to. Same-quality roundtrip BER drops from ~40-50% to ~5-10%.
ROBUST_JPEG_GUARD_QUALITY = 50

# Quantisation-grid hypotheses tried during extraction (must include the guard
# quality; the hypothesis matching the attack quality yields low BER).
ROBUST_HYPOTHESIS_QUALITIES = [90, 70, 60, 55, 50, 45, 40, 30]

# The encoded stream is repeated K times; extraction majority-votes copies.
# K=4: survives ~6-11% per-copy JPEG error (voted BER ~1-2%).
ROBUST_K_COPIES = 4

# Fixed RS payload size in bytes: [16-bit length | 16-bit crc16 | message | pad].
# Deterministic framing lets the decoder know exactly how many bits to expect.
ROBUST_PAYLOAD_BYTES = 112

# The embedding window is computed adaptively as the centered 60% of the
# image (block rows/cols from ceil(0.2*B) to floor(0.8*B)) — the region that
# survives a 40% center-crop-with-black-borders at any resolution.

# Deterministic block skipping (embed + extract MUST agree; content-based
# skipping is unstable because embedding itself changes block statistics).
ROBUST_SKIP_MOD = 5        # skip blocks where (br + bc) % 5 == 0

# Low-mid frequency positions whose Q50 guard steps are <= 16: robust to JPEG
# with moderate distortion.
ROBUST_POSITIONS = [
    (1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1), (1, 2),
    (3, 0), (0, 3), (2, 2), (1, 3), (3, 1), (4, 0), (0, 4),
]

# Transformations
JPEG_QUALITIES = [10, 30, 50, 70, 90]
RESIZE_FACTORS = [0.5, 0.75, 1.25, 1.5]
CROP_PERCENTS = [0.10, 0.20, 0.30, 0.40]
NOISE_VARIANCES = [0.01, 0.02, 0.03, 0.04, 0.05]

COMBINED_SEQUENCES = {
    'C1': {'order': ['resize', 'jpeg'], 'params': {'resize': 0.75, 'jpeg': 50}},
    'C2': {'order': ['jpeg', 'resize'], 'params': {'jpeg': 50, 'resize': 0.75}},
    'C3': {'order': ['resize', 'crop', 'jpeg'], 'params': {'resize': 0.75, 'crop': 0.20, 'jpeg': 50}},
    'C4': {'order': ['jpeg', 'crop', 'noise'], 'params': {'jpeg': 50, 'crop': 0.20, 'noise': 0.02}}
}

RANDOM_SEED = 42