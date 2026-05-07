"""Web Interface for Steganography Prototype"""
# Add production fix
'1iimport os\nimport sys\n'
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import json

# Import your existing modules
import config
from embedding import DCTEmbedder
from extraction import DCTExtractor
from error_correction import Hamming74
from baselines import LSBSteganography, DCTWithoutEC

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize components
embedder = DCTEmbedder(config)
extractor = DCTExtractor(config)
lsb = LSBSteganography()
dct_no_ec = DCTWithoutEC(config)

def text_to_bits(text):
    bits = []
    for char in text:
        ascii_val = ord(char)
        for i in range(7, -1, -1):
            bits.append((ascii_val >> i) & 1)
    return bits

def bits_to_text(bits):
    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        ascii_val = sum(bit << (7 - j) for j, bit in enumerate(byte))
        chars.append(chr(ascii_val))
    return ''.join(chars)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed_message():
    """Handle image upload and message embedding"""
    try:
        # Get uploaded file
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
            
            stego_image, _ = embedder.embed(cover_image, encoded_bits)
            
            # Verify extraction
            extracted_encoded = extractor.extract(stego_image, len(encoded_bits))
            decoded_bits, _ = Hamming74.decode_bitstream(extracted_encoded, padding_info)
            recovered_text = bits_to_text(decoded_bits)
            
            method_name = "Proposed (DCT + Hamming)"
            
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
        
        # Convert to base64 for display
        _, buffer = cv2.imencode('.png', cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate metrics
        from evaluation import MetricsCalculator
        metrics = MetricsCalculator()
        
        original_bits = text_to_bits(message)
        extracted_bits = text_to_bits(recovered_text) if method != 'proposed' else decoded_bits
        
        min_len = min(len(original_bits), len(extracted_bits))
        errors = sum(original_bits[i] != extracted_bits[i] for i in range(min_len))
        ber = errors / min_len if min_len > 0 else 1.0
        
        matches = sum(message[i] == recovered_text[i] for i in range(min(len(message), len(recovered_text))))
        recovery_acc = (matches / len(message)) * 100 if message else 0
        
        psnr = metrics.psnr(cover_image, stego_image)
        
        return jsonify({
            'success': True,
            'stego_image': f'data:image/png;base64,{img_base64}',
            'method': method_name,
            'metrics': {
                'psnr': round(psnr, 2),
                'ber': round(ber, 4),
                'recovery_accuracy': round(recovery_acc, 1),
                'message_recovered': recovered_text[:100] + ('...' if len(recovered_text) > 100 else ''),
                'message_matches': recovered_text == message
            },
            'output_path': output_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_message():
    """Extract hidden message from stego image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        method = request.form.get('method', 'proposed')
        message_length = int(request.form.get('message_length', 100))  # Approximate length
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        stego_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        if stego_image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        if method == 'proposed':
            # Need to know encoded length - estimate based on image capacity
            capacity = embedder.get_capacity(stego_image.shape)
            estimated_bits = min(capacity, message_length * 8 * 7 // 4)  # Rough estimate
            extracted_encoded = extractor.extract(stego_image, estimated_bits)
            # Attempt decoding with different paddings
            for padding in range(8):
                decoded_bits, _ = Hamming74.decode_bitstream(extracted_encoded, (padding, 0))
                recovered_text = bits_to_text(decoded_bits)
                if len(recovered_text) > 5 and all(ord(c) < 128 for c in recovered_text):
                    break
            method_name = "Proposed (DCT + Hamming)"
            
        elif method == 'dct_no_ec':
            capacity = dct_no_ec.get_capacity(stego_image.shape)
            estimated_bits = min(capacity, message_length * 8)
            extracted_bits = dct_no_ec.extract(stego_image, estimated_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"
            
        else:  # LSB
            capacity = stego_image.shape[0] * stego_image.shape[1] * 3
            estimated_bits = min(capacity, message_length * 8)
            extracted_bits = lsb.extract(stego_image, estimated_bits)
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"
        
        return jsonify({
            'success': True,
            'extracted_message': recovered_text[:500],
            'method': method_name,
            'message_length': len(recovered_text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_methods():
    """Compare all three methods on same image and message"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        message = request.form.get('message', '')
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
        
        results = {}
        
        # Test all three methods
        methods = {
            'proposed': ('Proposed (DCT+Hamming)', embedder, extractor, True),
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
                    else:
                        recovered = "Message too long"
                else:
                    if method_key == 'lsb':
                        stego = embed.embed(cover_image, message)
                        original_bits = embed.text_to_bits(message)
                        extracted = embed.extract(stego, len(original_bits))
                        recovered = embed.bits_to_text(extracted)
                    else:
                        stego, num = embed.embed(cover_image, message)
                        extracted = embed.extract(stego, num)
                        recovered = bits_to_text(extracted)
                
                from evaluation import MetricsCalculator
                metrics = MetricsCalculator()
                psnr = metrics.psnr(cover_image, stego)
                
                # Calculate recovery accuracy
                matches = sum(message[i] == recovered[i] for i in range(min(len(message), len(recovered))))
                recovery_acc = (matches / len(message)) * 100 if message else 0
                
                results[method_key] = {
                    'name': method_name,
                    'psnr': round(psnr, 2),
                    'recovery_accuracy': round(recovery_acc, 1),
                    'message_recovered': recovered[:100]
                }
            except Exception as e:
                results[method_key] = {'error': str(e)}
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download generated stego images"""
    from flask import send_from_directory
    return send_from_directory('uploads', filename, as_attachment=True)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 Steganography Web Interface")
    print("="*50)
    print("Starting server at: http://localhost:5000")
    print("Press CTRL+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)