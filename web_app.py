"""Web Interface for Steganography Prototype"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import json

import config
from embedding import DCTEmbedder
from extraction import DCTExtractor
from error_correction import Hamming74
from baselines import LSBSteganography, DCTWithoutEC
from robustness_tests import TransformationTester
from evaluation import MetricsCalculator

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize components
embedder = DCTEmbedder(config)
extractor = DCTExtractor(config)
lsb = LSBSteganography()
dct_no_ec = DCTWithoutEC(config)
metrics = MetricsCalculator()

def text_to_bits(text):
    """Convert text to bits - FIXED"""
    bits = []
    for char in text:
        ascii_val = ord(char)
        for i in range(7, -1, -1):
            bits.append((ascii_val >> i) & 1)
    return bits

def bits_to_text(bits):
    """Convert bits to text - FIXED"""
    # Remove padding zeros at end
    while bits and bits[-1] == 0:
        bits = bits[:-1]
    
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - len(bits) % 8)
    
    chars = []
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        byte = bits[i:i+8]
        ascii_val = 0
        for j, bit in enumerate(byte):
            ascii_val = (ascii_val << 1) | bit
        if 32 <= ascii_val <= 126 or ascii_val in [10, 13]:
            chars.append(chr(ascii_val))
    
    return ''.join(chars)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_message():
    """Handle image upload and message embedding - FIXED"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        message = request.form.get('message', '')
        method = request.form.get('method', 'proposed')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not message:
            return jsonify({'error': 'No message to hide'}), 400
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if cover_image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Embed based on selected method
        if method == 'proposed':
            original_bits = text_to_bits(message)
            encoded_bits, padding_info = Hamming74.encode_bitstream(original_bits)
            
            capacity = embedder.get_capacity(cover_image.shape)
            if len(encoded_bits) > capacity:
                return jsonify({'error': f'Message too long. Max {capacity//8} characters'}), 400
            
            stego_image, embedding_map = embedder.embed(cover_image, encoded_bits)
            
            # Get classification map
            class_map = embedder.get_classification_map(cover_image)
            edge_blocks = int(np.sum(class_map))
            smooth_blocks = class_map.size - edge_blocks
            
            # Extract to verify
            extracted_encoded = extractor.extract(stego_image, len(encoded_bits))
            decoded_bits, error_count = Hamming74.decode_bitstream(extracted_encoded, padding_info)
            recovered_text = bits_to_text(decoded_bits)
            
            method_name = f"Proposed EDC-HC (Edge: {edge_blocks}, Smooth: {smooth_blocks})"
            
        elif method == 'dct_no_ec':
            stego_image, num_bits = dct_no_ec.embed(cover_image, message)
            extracted_bits = dct_no_ec.extract(stego_image, num_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"
            
        else:  # LSB
            stego_image = lsb.embed(cover_image, message)
            original_bits = lsb.text_to_bits(message)
            extracted_bits = lsb.extract(stego_image, len(original_bits))
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"
        
        # Save stego image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_{timestamp}.png')
        cv2.imwrite(output_path, cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate metrics
        original_bits = text_to_bits(message)
        
        if method == 'proposed':
            extracted_bits = decoded_bits
        else:
            extracted_bits = text_to_bits(recovered_text)
        
        ber = metrics.bit_error_rate(original_bits, extracted_bits)
        recovery_acc = metrics.text_recovery_accuracy(message, recovered_text)
        psnr = metrics.psnr(cover_image, stego_image)
        ssim = metrics.ssim_index(cover_image, stego_image)
        
        # Check if recovered text matches
        message_matches = recovered_text == message
        
        return jsonify({
            'success': True,
            'stego_image': f'data:image/png;base64,{img_base64}',
            'method': method_name,
            'metrics': {
                'psnr': round(psnr, 2),
                'ssim': round(ssim, 4),
                'ber': round(ber, 4),
                'recovery_accuracy': round(recovery_acc, 1),
                'message_recovered': recovered_text[:100] + ('...' if len(recovered_text) > 100 else ''),
                'message_matches': message_matches
            },
            'output_path': output_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_message():
    """Extract hidden message - FIXED"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        method = request.form.get('method', 'proposed')
        message_length = int(request.form.get('message_length', 100))
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        stego_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if stego_image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        recovered_text = ""
        
        if method == 'proposed':
            capacity = embedder.get_capacity(stego_image.shape)
            estimated_bits = min(capacity, message_length * 8 * 7 // 4 + 10)
            extracted_encoded = extractor.extract(stego_image, estimated_bits)
            
            # Try different paddings
            best_text = ""
            best_score = 0
            
            for padding in range(8):
                try:
                    decoded_bits, _ = Hamming74.decode_bitstream(extracted_encoded, (padding, 0))
                    text = bits_to_text(decoded_bits)
                    score = sum(1 for c in text if 32 <= ord(c) <= 126)
                    if score > best_score:
                        best_score = score
                        best_text = text
                except:
                    continue
            
            recovered_text = best_text
            method_name = "Proposed EDC-HC"
            
        elif method == 'dct_no_ec':
            capacity = dct_no_ec.get_capacity(stego_image.shape)
            estimated_bits = min(capacity, message_length * 8 + 10)
            extracted_bits = dct_no_ec.extract(stego_image, estimated_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"
            
        else:
            capacity = stego_image.shape[0] * stego_image.shape[1] * 3
            estimated_bits = min(capacity, message_length * 8 + 10)
            extracted_bits = lsb.extract(stego_image, estimated_bits)
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"
        
        # Clean extracted text
        cleaned = ''.join(c for c in recovered_text if 32 <= ord(c) <= 126 or c in '\n\r\t')
        
        return jsonify({
            'success': True,
            'extracted_message': cleaned[:500] if cleaned else "No readable message found",
            'method': method_name,
            'message_length': len(cleaned)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_image():
    """Show edge/smooth block classification - FIXED (no matplotlib)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get classification map
        class_map = embedder.get_classification_map(image)
        
        # Create visualization using OpenCV (no matplotlib)
        h, w = image.shape[:2]
        block_h = h // 8
        block_w = w // 8
        
        overlay = image.copy()
        
        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w
                
                if class_map[br, bc] == 1:  # Edge block - RED
                    color = (0, 0, 255)
                    alpha = 0.3
                else:  # Smooth block - GREEN
                    color = (0, 255, 0)
                    alpha = 0.2
                
                overlay[y1:y2, x1:x2] = cv2.addWeighted(
                    overlay[y1:y2, x1:x2], 1 - alpha,
                    np.full((block_h, block_w, 3), color, dtype=np.uint8), alpha, 0
                )
        
        # Add borders
        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w
                
                if class_map[br, bc] == 1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        edge_count = int(np.sum(class_map))
        smooth_count = class_map.size - edge_count
        
        return jsonify({
            'success': True,
            'classification_image': f'data:image/png;base64,{img_base64}',
            'edge_blocks': edge_count,
            'smooth_blocks': smooth_count,
            'total_blocks': class_map.size,
            'alpha_edge': config.EMBEDDING_STRENGTH_EDGE,
            'alpha_smooth': config.EMBEDDING_STRENGTH_SMOOTH
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_methods():
    """Compare all three methods"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        message = request.form.get('message', '')
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        results = {}
        methods = {
            'proposed': ('Proposed EDC-HC', embedder, extractor, True),
            'dct_no_ec': ('DCT without EC', dct_no_ec, dct_no_ec, False),
            'lsb': ('LSB', lsb, lsb, False)
        }
        
        for method_key, (method_name, embed, extract, use_hamming) in methods.items():
            try:
                if use_hamming:
                    original_bits = text_to_bits(message)
                    encoded_bits, padding_info = Hamming74.encode_bitstream(original_bits)
                    capacity = embed.get_capacity(cover_image.shape)
                    if len(encoded_bits) <= capacity:
                        stego, _ = embed.embed(cover_image, encoded_bits)
                        extracted_encoded = extract.extract(stego, len(encoded_bits))
                        decoded_bits, _ = Hamming74.decode_bitstream(extracted_encoded, padding_info)
                        recovered = bits_to_text(decoded_bits)
                        class_map = embed.get_classification_map(cover_image)
                        extra = f"Edge: {int(np.sum(class_map))}, Smooth: {class_map.size - int(np.sum(class_map))}"
                    else:
                        recovered = "Message too long"
                        extra = "N/A"
                else:
                    if method_key == 'lsb':
                        stego = embed.embed(cover_image, message)
                        original_bits = embed.text_to_bits(message)
                        extracted = embed.extract(stego, len(original_bits))
                        recovered = embed.bits_to_text(extracted)
                        extra = "N/A"
                    else:
                        stego, num = embed.embed(cover_image, message)
                        extracted = embed.extract(stego, num)
                        recovered = bits_to_text(extracted)
                        extra = "N/A"
                
                psnr = metrics.psnr(cover_image, stego)
                ssim = metrics.ssim_index(cover_image, stego)
                
                matches = sum(message[i] == recovered[i] for i in range(min(len(message), len(recovered))))
                recovery_acc = (matches / len(message)) * 100 if message else 0
                
                results[method_key] = {
                    'name': method_name,
                    'psnr': round(psnr, 2),
                    'ssim': round(ssim, 4),
                    'recovery_accuracy': round(recovery_acc, 1),
                    'message_recovered': recovered[:100],
                    'extra': extra
                }
            except Exception as e:
                results[method_key] = {'error': str(e)}
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download stego image"""
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 404
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 Steganography Web Interface - EDC-HC")
    print("="*50)
    print(f"α_edge = {config.EMBEDDING_STRENGTH_EDGE}")
    print(f"α_smooth = {config.EMBEDDING_STRENGTH_SMOOTH}")
    print(f"Edge Threshold = {config.EDGE_DENSITY_THRESHOLD}")
    print("="*50)
    print("Starting server at: http://localhost:5000")
    print("Press CTRL+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
