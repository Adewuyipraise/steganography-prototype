# 🔐 Robust Image Steganography Prototype

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready steganography system that hides secret messages inside images using **DCT-based frequency domain embedding** with **Reed-Solomon error correction** and **adaptive edge detection**. The robust pipeline embeds parity on JPEG-quantisation-aligned grids so hidden text survives JPEG compression, resizing and cropping. Includes web interface and comparison with LSB and DCT-only methods.

## 🌟 Features

- **Three Steganography Methods**:
  - ✅ **Proposed Method**: DCT + JPEG-guard grid embedding + Reed-Solomon(63,192) + repetition & majority voting (most robust)
  - 📊 **DCT without EC**: DCT embedding without error correction
  - 💾 **LSB Spatial Domain**: Basic least significant bit substitution

- **Robustness Testing**:
  - JPEG compression (10% to 90% quality)
  - Gaussian noise injection
  - Image resizing (50% to 150%)
  - Random cropping (10% to 40%)
  - Gaussian blur (3×3, 5×5)
  - Format conversion (PNG ↔ JPEG)

- **Evaluation Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - BER (Bit Error Rate)
  - Text Recovery Accuracy (%)
  - Composite Robustness Score

- **Web Interface**:
  - Drag-and-drop image upload
  - Real-time embedding/extraction
  - Side-by-side method comparison
  - Download stego images
  - Mobile responsive design

## 📊 Performance Results

| Method | Avg Recovery (Under Attack) | PSNR | BER | Robustness Score |
|--------|----------------------------|------|-----|------------------|
| **Proposed (DCT+Hamming)** | **47.6%** | 47.7 dB | 0.224 | **0.395** |
| DCT without EC | 10.7% | 36.1 dB | 0.349 | 0.070 |
| LSB Spatial | 3.0% | 81.8 dB | 0.447 | 0.016 |

**Key Finding**: The proposed method is **4.5× more robust** than DCT alone and **16× more robust** than LSB under image transformations.

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
```

# Installation
# Clone the repository
git clone https://github.com/Adewuyipraise/steganography-prototype.git
cd steganography-prototype

# Install dependencies
pip install -r requirements.txt

# Run the web interface
python web_app.py

Open your browser to: http://localhost:5000


## 🖥️ Usage
Web Interface

run 
```bash
python web_app.py
```
Then:

Upload an image (PNG, JPG, JPEG)
Enter your secret message
Select embedding method
Click "Hide Message"
Download the stego image

To extract:
Upload the stego image
Select the same method used for embedding
Click "Extract Message"


 ## Methodology
### Embedding Process
- Convert image to YCbCr color space
- Extract luminance (Y) channel
- Apply Sobel edge detection for adaptive strength
- Divide into 8×8 blocks
- Apply DCT to each block
- Embed bits into mid-frequency coefficients
- Apply Inverse DCT
- Reconstruct stego image

### Robust Pipeline (survives JPEG / resize / crop)
Why naive parity embedding fails under transformations and what fixes it:

1. **JPEG-guard grid alignment** — the earlier scheme added a tiny absolute value
   (`alpha ≈ 0.26`) to DCT coefficients, while JPEG quantisation replaces every
   coefficient with a multiple of its quantisation step (≈ 8–30 for Q50–Q70).
   The embedded parity was destroyed. The robust pipeline instead snaps each
   coefficient to a multiple of the *libjpeg luminance quantisation step at the
   guard quality* (Q50), so parity sits exactly on the grid a JPEG encoder
   requantises to — same-quality JPEG round-trips drop from ~40–50% to ~5–10%
   per-bit error.
2. **Fixed-size deterministic framing** — the message is framed as
   `[16-bit length | 16-bit crc16 | payload]` and zero-padded to a fixed frame
   so the decoder always knows exactly how many bits to expect. The CRC header
   rejects Reed-Solomon miscorrections.
3. **High-ECC multi-codeword Reed-Solomon** — `RS(63, 192)` (96 correctable
   bytes per 255-byte codeword) with depth-8 interleaving. Small messages span
   multiple codewords, so a few bad bytes no longer blow the whole budget.
4. **Repetition + majority voting** — the encoded stream is embedded 4 times,
   each copy guarded by the 16-bit sync marker. Extraction finds unit-spaced
   sync chains (spurious 16-bit matches scatter randomly and are filtered out),
   majority-votes the copies (turning 6–11% per-copy JPEG error into ~1–2%),
   then RS-decodes and CRC-validates.
5. **Crop-safe centered window** — embedding is confined to the centered 60% of
   the image, the exact region a 40% center-crop-with-black-borders preserves,
   so cropping can never destroy all copies.
6. **Multi-quality hypothesis search** — extraction re-reads parity under several
   quantisation-grid hypotheses and keeps the frame whose payload passes CRC.

### Error Correction
Reed-Solomon(63, 192) with depth-8 interleaving:
Encodes the fixed payload into multiple 255-byte codewords
Corrects up to 96 byte errors per codeword
Combined with 4-copy repetition and majority voting for very low residual BER
A 16-bit CRC in the payload header guarantees decoded-text integrity



---

<div align="center">
  
  <sub>
    <br/>
    ═══════════════════════════════════════════════════
  </sub>
  
  <br/>
  
### 🚀 Crafted with Code & Precision by **Pesedie Rasheed** 🚀
  
  ### 🔐 "Quantum Computing Enthusiats || Network Administrator || Cybersecurity Analyst || Computer Scientist || IT Support Personnel || Frontend Developer" 🔐
  
  <sub>
    ═══════════════════════════════════════════════════
  </sub>
  
  <br/>
  <br/>
  
  [![GitHub](https://img.shields.io/badge/GitHub-Pesedie_Rasheed-181717?style=flat-square&logo=github)](https://github.com/Adewuyipraise)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Pesedie_Rasheed-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/pesedierasheed)
  [![Email](https://img.shields.io/badge/Email-rasheedpesedie@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:rasheedpesedie@gmail.com)
  
</div>

---