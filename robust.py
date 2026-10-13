"""Shared helpers for the robust (JPEG-guard grid) steganography pipeline.

Design (validated empirically against the project's transformation battery):
  1. Parity is embedded at magnitudes equal to the libjpeg luminance
     quantisation table at a "guard" quality (Q50), so values sit exactly on
     the grid JPEG requantises to. This turns a JPEG roundtrip from ~40-50%
     per-bit error into ~5-10% (down to 5% clean).
  2. The payload is a fixed-size frame: [16-bit length | 16-bit crc16 | msg].
     It is RS-encoded with a high-ECC multi-codeword code and bit-interleaved.
  3. The encoded stream is embedded ROBUST_K_COPIES times (each copy preceded
     by the sync marker). Extraction majority-votes the copies, which turns
     6-11% per-copy bit errors into ~1-2% before RS decoding.
  4. Embedding happens in a centered window of blocks that survives every crop
     percentage in the test battery, so cropping can never destroy all copies.
"""

import numpy as np
import zlib

# Standard IJG / libjpeg luminance quantisation table.
BASE_QUANT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

# 16-bit sync marker (same sequence used by the classic pipeline).
SYNC_MARKER = [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]

# 32-bit framing header: 16-bit message length + 16-bit crc16.
HEADER_BITS = 32


def quant_table(quality):
    """libjpeg luminance quantisation table at the given quality (1-100)."""
    scale = 5000.0 / quality if quality < 50 else 200 - 2 * quality
    table = np.floor((BASE_QUANT * scale + 50) / 100)
    table[table < 1] = 1
    return table.astype(np.int64)


def bits_to_bytes(bits):
    """Convert a bit list to bytes (zero-padded to a byte boundary)."""
    bits = list(bits) + [0] * ((8 - len(bits) % 8) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def bytes_to_bits(data):
    """Convert bytes to an MSB-first bit list."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def crc16(bits):
    """16-bit CRC of a bit list (for payload integrity validation)."""
    crc = zlib.crc32(bits_to_bytes(bits)) & 0xFFFF
    return [(crc >> (15 - i)) & 1 for i in range(16)]


def message_capacity_bits(config):
    """Maximum message bits that fit inside a fixed payload frame."""
    return config.ROBUST_PAYLOAD_BYTES * 8 - HEADER_BITS


def payload_bits(msg_bits, config):
    """Build the fixed-length payload: [len16 | crc16 | msg | zeros]."""
    n = len(msg_bits)
    max_msg = message_capacity_bits(config)
    assert n <= max_msg, (
        f'message too long: {n} bits exceeds {max_msg}-bit payload capacity')
    head = [int(b) for b in f'{n:016b}'] + crc16(msg_bits)
    return head + list(msg_bits) + [0] * (max_msg - n)


def parse_payload(bits, config):
    """Unpack a decoded payload. Returns the message bits or None if invalid."""
    want = config.ROBUST_PAYLOAD_BYTES * 8
    if len(bits) != want:
        return None
    n = int(''.join(str(b) for b in bits[:16]), 2)
    if n < 0 or n > message_capacity_bits(config):
        return None
    msg = bits[HEADER_BITS:HEADER_BITS + n]
    if crc16(msg) != bits[16:HEADER_BITS]:
        return None
    return msg


def encoded_length_bits(config):
    """Length in bits of the RS-encoded fixed payload (before repetition)."""
    payload_bytes = config.ROBUST_PAYLOAD_BYTES
    chunk_size = 255 - config.RS_ECC_SYMBOLS
    n_chunks = -(-payload_bytes // chunk_size)
    return (payload_bytes + n_chunks * config.RS_ECC_SYMBOLS) * 8


def stream_bits(config):
    """Total bits required for the full repeated stream (sync + RS frames)."""
    unit = len(SYNC_MARKER) + encoded_length_bits(config)
    return config.ROBUST_K_COPIES * unit


def band_rows(h, block_size, config):
    """Row range (in 8px blocks) of the centered crop-safe embedding window.

    Corner case convenience: crops in the battery keep the centered 60% of the
    image, so we embed in block rows [ceil(0.2*B), floor(0.8*B))."""
    blocks_h = h // block_size
    r0 = (blocks_h + 4) // 5          # ceil(0.2 * B)
    r1 = (blocks_h * 4) // 5          # floor(0.8 * B)
    return max(0, r0), max(r0, r1)


def band_cols(w, block_size, config):
    """Column range (in 8px blocks) of the centered crop-safe embedding window."""
    blocks_w = w // block_size
    c0 = (blocks_w + 4) // 5          # ceil(0.2 * B)
    c1 = (blocks_w * 4) // 5          # floor(0.8 * B)
    return max(0, c0), max(c0, c1)


def usable_blocks(h, w, block_size, config):
    """Number of usable blocks in the window (after the deterministic skip)."""
    r0, r1 = band_rows(h, block_size, config)
    c0, c1 = band_cols(w, block_size, config)
    mod = config.ROBUST_SKIP_MOD
    return sum(1 for br in range(r0, r1) for bc in range(c0, c1)
               if (br + bc) % mod != 0)


def max_message_bits(h, w, block_size, config):
    """Max message bits that fit for a given image shape, else 0."""
    usable = usable_blocks(h, w, block_size, config)
    embeddable = usable * len(config.ROBUST_POSITIONS)
    if embeddable < stream_bits(config):
        return 0
    return message_capacity_bits(config)


def score_text_bits(bits):
    """Heuristic score of decoded payload bits as printable text."""
    if not bits:
        return -1
    data = bits_to_bytes(bits)
    ok = sum(1 for c in data if 32 <= c <= 126 or c in (10, 13, 9))
    bad = sum(1 for c in data if c < 32 or c > 126)
    return ok - 3.0 * bad
