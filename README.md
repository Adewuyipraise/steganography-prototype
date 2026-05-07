# 🔐 Robust Image Steganography Prototype

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready steganography system that hides secret messages inside images using **DCT-based frequency domain embedding** with **Hamming (7,4) error correction** and **adaptive edge detection**. Includes web interface and comparison with LSB and DCT-only methods.

## 🌟 Features

- **Three Steganography Methods**:
  - ✅ **Proposed Method**: DCT + Hamming(7,4) + Sobel edge adaptation (most robust)
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
Embedding Process
Convert image to YCbCr color space
Extract luminance (Y) channel
Apply Sobel edge detection for adaptive strength
Divide into 8×8 blocks
Apply DCT to each block
Embed bits into mid-frequency coefficients
Apply Inverse DCT
Reconstruct stego image

### Error Correction
Hamming (7,4) code
Encodes 4 data bits into 7 code bits
Corrects single-bit errors per codeword
Improves robustness under JPEG compression

