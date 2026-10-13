"""Web Interface for Steganography Prototype - EDC-HC"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from text_utils import text_to_bits, bits_to_text, verify_text_encoding
from embedding import DCTEmbedder
from extraction import DCTExtractor
from error_correction import Hamming74
from baselines import LSBSteganography, DCTWithoutEC
from evaluation import MetricsCalculator
from robustness_tests import TransformationTester

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize components
embedder = DCTEmbedder(config)
extractor = DCTExtractor(config)
lsb = LSBSteganography()
dct_no_ec = DCTWithoutEC(config)
metrics = MetricsCalculator()
tester = TransformationTester(config)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/embed', methods=['POST'])
def embed_message():
    """
    Handle image upload and message embedding.
    Uses the updated text_utils for proper encoding/decoding.
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        message = request.form.get('message', '')
        method = request.form.get('method', 'proposed')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not message:
            return jsonify({'error': 'No message to hide'}), 400

        # Read and validate image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cover_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
        h, w = cover_image.shape[:2]

        if h < 32 or w < 32:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 32x32 pixels required'}), 400

        print(f"\n📝 Embedding: '{message[:50]}...' ({len(message)} chars)")
        print(f"📸 Image: {w}x{h}")

        # Embed based on selected method
        if method == 'proposed':
            # Convert message to bits
            original_bits = text_to_bits(message)
            print(f"📊 Original bits: {len(original_bits)}")

            # Encode with Hamming
            encoded_bits, padding_info = Hamming74.encode_bitstream(original_bits)
            print(f"🔐 Encoded bits: {len(encoded_bits)}")

            # Check capacity
            capacity = embedder.get_capacity(cover_image.shape)
            if len(encoded_bits) > capacity:
                return jsonify({
                    'error': f'Message too long. Max {capacity//8} characters, got {len(message)}'
                }), 400

            # Embed
            stego_image, embedding_map = embedder.embed(cover_image, encoded_bits)

            # Get classification map for display
            class_map = embedder.get_classification_map(cover_image)
            edge_blocks = int(np.sum(class_map))
            smooth_blocks = class_map.size - edge_blocks

            # Verify extraction
            extracted_encoded = extractor.extract(stego_image, len(encoded_bits))
            decoded_bits, error_count = Hamming74.decode_bitstream(extracted_encoded, padding_info)

            # Recover text
            recovered_text = bits_to_text(decoded_bits)
            print(f"📝 Recovered: '{recovered_text[:50]}...'")
            print(f"✅ Hamming errors corrected: {error_count}")

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

        # Convert to base64 for display
        _, buffer = cv2.imencode('.png', cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate metrics
        original_bits = text_to_bits(message)

        if method == 'proposed':
            extracted_bits = decoded_bits
            ber = metrics.bit_error_rate(original_bits, extracted_bits)
        else:
            # For other methods, compare recovered text bits
            recovered_bits = text_to_bits(recovered_text) if recovered_text else []
            min_len = min(len(original_bits), len(recovered_bits))
            if min_len > 0:
                errors = sum(original_bits[i] != recovered_bits[i] for i in range(min_len))
                ber = errors / min_len
            else:
                ber = 1.0

        recovery_acc = metrics.text_recovery_accuracy(message, recovered_text)
        psnr = metrics.psnr(cover_image, stego_image)
        ssim = metrics.ssim_index(cover_image, stego_image)

        # Check if recovered text matches
        message_matches = (recovered_text == message)
        print(f"✅ Message matches: {message_matches}")
        print(f"📊 Recovery accuracy: {recovery_acc:.1f}%")

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
                'message_matches': message_matches,
                'message_length': len(recovered_text)
            },
            'output_path': output_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/extract', methods=['POST'])
def extract_message():
    """
    Extract hidden message from stego image.
    Uses updated text_utils for proper decoding.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        method = request.form.get('method', 'proposed')
        message_length = int(request.form.get('message_length', 100))

        # Read and validate image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        stego_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if stego_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        stego_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)
        h, w = stego_image.shape[:2]

        if h < 32 or w < 32:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 32x32 pixels required'}), 400

        recovered_text = ""
        method_name = ""

        if method == 'proposed':
            # Estimate number of encoded bits
            capacity = embedder.get_capacity(stego_image.shape)
            # Each char = 8 bits, Hamming adds 75% overhead
            estimated_bits = min(capacity, int(message_length * 8 * 1.75) + 20)

            extracted_encoded = extractor.extract(stego_image, estimated_bits)

            # Try different padding values to find the correct one
            best_text = ""
            best_score = 0

            for padding in range(8):
                try:
                    padding_info = (padding, 0)
                    decoded_bits, _ = Hamming74.decode_bitstream(extracted_encoded, padding_info)
                    text = bits_to_text(decoded_bits)

                    # Score: prefer text with more printable characters
                    printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
                    alnum = sum(1 for c in text if c.isalnum() or c.isspace())
                    score = printable + alnum * 2

                    if score > best_score:
                        best_score = score
                        best_text = text
                except Exception:
                    continue

            recovered_text = best_text
            method_name = "Proposed EDC-HC"

        elif method == 'dct_no_ec':
            capacity = dct_no_ec.get_capacity(stego_image.shape)
            estimated_bits = min(capacity, int(message_length * 8) + 20)
            extracted_bits = dct_no_ec.extract(stego_image, estimated_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"

        else:  # LSB
            capacity = stego_image.shape[0] * stego_image.shape[1] * 3
            estimated_bits = min(capacity, int(message_length * 8) + 20)
            extracted_bits = lsb.extract(stego_image, estimated_bits)
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"

        # Clean extracted text (keep only printable characters)
        cleaned = ''.join(c for c in recovered_text if 32 <= ord(c) <= 126 or c in '\n\r\t')

        # If we got nothing, try fallback
        if not cleaned and recovered_text:
            # Sometimes text has null characters - remove them
            cleaned = recovered_text.replace('\x00', '')
            if cleaned:
                cleaned = ''.join(c for c in cleaned if 32 <= ord(c) <= 126 or c in '\n\r\t')

        return jsonify({
            'success': True,
            'extracted_message': cleaned[:500] if cleaned else "No readable message found",
            'method': method_name,
            'message_length': len(cleaned),
            'raw_length': len(recovered_text)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/classify', methods=['POST'])
def classify_image():
    """
    Show edge/smooth block classification visualization.
    Uses OpenCV for rendering (no matplotlib dependency).
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if h < 32 or w < 32:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 32x32 pixels required'}), 400

        # Get classification map
        class_map = embedder.get_classification_map(image)

        # Create visualization
        h, w = image.shape[:2]
        block_h = h // 8
        block_w = w // 8

        # Create overlay
        overlay = image.astype(np.float32) / 255.0
        color_overlay = np.zeros_like(overlay, dtype=np.float32)

        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w

                if class_map[br, bc] == 1:  # Edge block - RED
                    color_overlay[y1:y2, x1:x2] = [1.0, 0.0, 0.0]
                else:  # Smooth block - GREEN
                    color_overlay[y1:y2, x1:x2] = [0.0, 1.0, 0.0]

        # Blend
        alpha = 0.3
        blended = (1 - alpha) * overlay + alpha * color_overlay
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        # Draw borders
        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w

                if class_map[br, bc] == 1:
                    cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 1)
                else:
                    cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        edge_count = int(np.sum(class_map))
        smooth_count = class_map.size - edge_count

        return jsonify({
            'success': True,
            'classification_image': f'data:image/png;base64,{img_base64}',
            'edge_blocks': edge_count,
            'smooth_blocks': smooth_count,
            'total_blocks': class_map.size,
            'edge_percentage': round((edge_count / class_map.size) * 100, 2),
            'alpha_edge': config.EMBEDDING_STRENGTH_EDGE,
            'alpha_smooth': config.EMBEDDING_STRENGTH_SMOOTH
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare_methods():
    """
    Compare all three methods on the same image and message.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        message = request.form.get('message', '')

        if not message:
            return jsonify({'error': 'No message provided for comparison'}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cover_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
        h, w = cover_image.shape[:2]

        if h < 32 or w < 32:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 32x32 pixels required'}), 400

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

                        # Get classification info
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

                # Calculate recovery accuracy
                matches = sum(message[i] == recovered[i] for i in range(min(len(message), len(recovered))))
                recovery_acc = (matches / len(message)) * 100 if message else 0

                results[method_key] = {
                    'name': method_name,
                    'psnr': round(psnr, 2),
                    'ssim': round(ssim, 4),
                    'recovery_accuracy': round(recovery_acc, 1),
                    'message_recovered': recovered[:100] + ('...' if len(recovered) > 100 else ''),
                    'message_matches': recovered == message,
                    'extra': extra
                }

            except Exception as e:
                results[method_key] = {'error': str(e)}

        return jsonify({
            'success': True,
            'results': results,
            'original_message': message
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download stego image file."""
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 404

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Steganography Web Interface - EDC-HC")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   α_edge = {config.EMBEDDING_STRENGTH_EDGE}")
    print(f"   α_smooth = {config.EMBEDDING_STRENGTH_SMOOTH}")
    print(f"   Edge Threshold = {config.EDGE_DENSITY_THRESHOLD}")
    print(f"   Block Size = {config.BLOCK_SIZE}")
    print("="*60)
    print("🚀 Starting server at: http://localhost:5000")
    print("📁 Upload folder: uploads/")
    print("📊 Results folder: results/")
    print("🛑 Press CTRL+C to stop")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)