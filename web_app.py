"""Web Interface for Steganography Prototype - EDC-HC (Reed-Solomon)"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from text_utils import text_to_bits, bits_to_text
from embedding import DCTEmbedder
from extraction import DCTExtractor
from reed_solomon import TurboReedSolomon
from baselines import LSBSteganography, DCTWithoutEC
from evaluation import MetricsCalculator
from robustness_tests import TransformationTester

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize components
embedder = DCTEmbedder(config)
extractor = DCTExtractor(config)
lsb = LSBSteganography()
dct_no_ec = DCTWithoutEC(config)
metrics = MetricsCalculator()
tester = TransformationTester(config)

# Reed-Solomon settings
RS_DATA_SYMBOLS = 127
RS_ECC_SYMBOLS = 128
RS_INTERLEAVE_DEPTH = 4


def estimate_capacity_bits(image_shape):
    """Estimate usable embedding capacity in bits."""
    return embedder.get_capacity(image_shape)


def calculate_required_bits(message):
    """
    Calculate required bits for a message with RS encoding.
    RS(127,128): 127 data bytes + 128 ECC = 255 total, so 2x expansion.
    """
    data_bits = len(message) * 8
    # RS roughly doubles (255/127 ≈ 2.0)
    rs_bits = int(data_bits * 255 / 127) + 512
    return rs_bits


def encode_with_rs(text, copies=2):
    """Encode text using Reed-Solomon with configurable repetition."""
    bits = text_to_bits(text)
    rs = TurboReedSolomon(
        data_symbols=RS_DATA_SYMBOLS,
        ecc_symbols=RS_ECC_SYMBOLS,
        interleave_depth=RS_INTERLEAVE_DEPTH
    )
    encoded_bits, padding_info = rs.encode_bitstream(bits)
    repeated = encoded_bits * copies
    return repeated, padding_info, len(encoded_bits)


def decode_with_rs(extracted_bits, padding_info, single_copy_length, copies=2):
    """Decode using Reed-Solomon, trying each copy."""
    rs = TurboReedSolomon(
        data_symbols=RS_DATA_SYMBOLS,
        ecc_symbols=RS_ECC_SYMBOLS,
        interleave_depth=RS_INTERLEAVE_DEPTH
    )
    
    best_text = ""
    best_score = 0
    
    for copy_idx in range(copies):
        start = copy_idx * single_copy_length
        end = start + single_copy_length
        if end > len(extracted_bits):
            continue
        
        single_copy = extracted_bits[start:end]
        
        try:
            decoded_bits, _ = rs.decode_bitstream(single_copy, padding_info)
            text = bits_to_text(decoded_bits)
            
            if text:
                printable = sum(1 for c in text if 32 <= ord(c) <= 126)
                score = printable / max(len(text), 1) * len(text)
                if score > best_score:
                    best_score = score
                    best_text = text
        except Exception:
            continue
    
    return best_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/embed', methods=['POST'])
def embed_message():
    """Handle image upload and message embedding with adaptive RS."""
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

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cover_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)
        h, w = cover_image.shape[:2]

        if h < 128 or w < 128:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 128x128 pixels required'}), 400

        print(f"\n📝 Embedding: '{message[:50]}...' ({len(message)} chars)")
        print(f"📸 Image: {w}x{h}")

        # =========================================================
        # PROPOSED: DCT + Reed-Solomon + ADAPTIVE REPETITION
        # =========================================================
        if method == 'proposed':
            capacity = estimate_capacity_bits(cover_image.shape)
            required_single = calculate_required_bits(message)
            
            # Adaptive copies: reduce copies if image is small
            if required_single * 2 <= capacity:
                copies = 2
            elif required_single <= capacity:
                copies = 1
            else:
                # Try shrinking RS overhead by reducing ECC
                # Calculate minimum required image size
                min_pixels = int(np.ceil(np.sqrt(required_single * 64 / 26)))
                return jsonify({
                    'error': f'Message too long for this image ({len(message)} chars). '
                             f'Required ~{required_single} bits, image has {capacity} bits. '
                             f'Try an image at least {min_pixels}x{min_pixels} pixels, '
                             f'or use a shorter message (max ~{capacity // 30} chars).'
                }), 400
            
            print(f"📊 Capacity: {capacity} bits, required: {required_single * copies} bits, copies: {copies}")
            
            encoded_bits, padding_info, single_copy_length = encode_with_rs(message, copies=copies)
            print(f"📊 Encoded bits: {len(encoded_bits)} ({single_copy_length} × {copies} copies)")

            if len(encoded_bits) > capacity:
                return jsonify({
                    'error': f'Encoded message too long. Required: {len(encoded_bits)} bits, '
                             f'capacity: {capacity} bits. Use a shorter message or larger image.'
                }), 400

            stego_image, _ = embedder.embed(cover_image, encoded_bits)
            extracted = extractor.extract(stego_image, len(encoded_bits))
            recovered_text = decode_with_rs(extracted, padding_info, single_copy_length, copies=copies)

            method_name = f"Proposed EDC-HC (DCT + RS, {copies} copy)"

        elif method == 'dct_no_ec':
            stego_image, num_bits = dct_no_ec.embed(cover_image, message)
            extracted_bits = dct_no_ec.extract(stego_image, num_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"

        else:
            stego_image = lsb.embed(cover_image, message)
            original_bits = lsb.text_to_bits(message)
            extracted_bits = lsb.extract(stego_image, len(original_bits))
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'stego_{timestamp}.png')
        cv2.imwrite(output_path, cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))

        _, buffer = cv2.imencode('.png', cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        original_bits = text_to_bits(message)
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

        message_matches = (recovered_text.strip() == message.strip())

        print(f"📝 Recovered: '{recovered_text[:50]}...'")
        print(f"✅ Match: {message_matches}")
        print(f"📊 PSNR: {psnr:.2f} dB, BER: {ber:.4f}, Recovery: {recovery_acc:.1f}%")

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
    """Extract hidden message using Reed-Solomon."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        method = request.form.get('method', 'proposed')
        message_length = int(request.form.get('message_length', 100))

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        stego_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if stego_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        stego_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)
        h, w = stego_image.shape[:2]

        if h < 128 or w < 128:
            return jsonify({'error': f'Image too small ({h}x{w}). Minimum 128x128 pixels required'}), 400

        recovered_text = ""
        method_name = ""

        if method == 'proposed':
            capacity = estimate_capacity_bits(stego_image.shape)
            
            # Try multiple copies settings and bit lengths
            best_text = ""
            best_score = 0
            
            for copies_guess in [2, 1]:
                # Estimate single copy length
                required_single = calculate_required_bits("X" * message_length)
                single_guess = min(required_single, capacity // copies_guess)
                
                if single_guess <= 0:
                    continue
                
                total_bits = single_guess * copies_guess
                extracted = extractor.extract(stego_image, total_bits)
                
                text = decode_with_rs(extracted, (len(extracted), message_length), single_guess, copies=copies_guess)
                if text:
                    printable = sum(1 for c in text if 32 <= ord(c) <= 126)
                    score = printable
                    if score > best_score:
                        best_score = score
                        best_text = text
                        method_name = f"Proposed EDC-HC ({copies_guess} copy)"
            
            recovered_text = best_text
            if not method_name:
                method_name = "Proposed EDC-HC"

        elif method == 'dct_no_ec':
            estimated_bits = min(
                dct_no_ec.get_capacity(stego_image.shape),
                int(message_length * 8) + 20
            )
            extracted_bits = dct_no_ec.extract(stego_image, estimated_bits)
            recovered_text = bits_to_text(extracted_bits)
            method_name = "DCT without Error Correction"

        else:
            estimated_bits = min(
                stego_image.shape[0] * stego_image.shape[1] * 3,
                int(message_length * 8) + 20
            )
            extracted_bits = lsb.extract(stego_image, estimated_bits)
            recovered_text = lsb.bits_to_text(extracted_bits)
            method_name = "LSB Spatial Domain"

        cleaned = ''.join(c for c in recovered_text if 32 <= ord(c) <= 126 or c in '\n\r\t')

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
    """Show edge/smooth block classification visualization."""
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
        class_map = embedder.get_classification_map(image)

        h, w = image.shape[:2]
        block_h = h // 8
        block_w = w // 8

        overlay = image.astype(np.float32) / 255.0
        color_overlay = np.zeros_like(overlay, dtype=np.float32)

        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w

                if class_map[br, bc] == 1:
                    color_overlay[y1:y2, x1:x2] = [1.0, 0.0, 0.0]
                else:
                    color_overlay[y1:y2, x1:x2] = [0.0, 1.0, 0.0]

        alpha = 0.3
        blended = (1 - alpha) * overlay + alpha * color_overlay
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        for br in range(class_map.shape[0]):
            for bc in range(class_map.shape[1]):
                y1 = br * block_h
                y2 = (br + 1) * block_h
                x1 = bc * block_w
                x2 = (bc + 1) * block_w
                color = (255, 0, 0) if class_map[br, bc] == 1 else (0, 255, 0)
                cv2.rectangle(blended, (x1, y1), (x2, y2), color, 1)

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
    """Compare all three methods."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        message = request.form.get('message', '')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        cover_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cover_image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2RGB)

        results = {}

        # --- Proposed (DCT + RS) ---
        try:
            capacity = estimate_capacity_bits(cover_image.shape)
            required_single = calculate_required_bits(message)
            
            if required_single * 2 <= capacity:
                copies = 2
            elif required_single <= capacity:
                copies = 1
            else:
                results['proposed'] = {'error': 'Message too long for this image'}
                copies = 1
            
            if copies >= 1:
                encoded_bits, padding_info, single_copy_length = encode_with_rs(message, copies=copies)

                if len(encoded_bits) <= capacity:
                    stego, _ = embedder.embed(cover_image, encoded_bits)
                    extracted = extractor.extract(stego, len(encoded_bits))
                    recovered = decode_with_rs(extracted, padding_info, single_copy_length, copies=copies)

                    psnr = metrics.psnr(cover_image, stego)
                    ssim = metrics.ssim_index(cover_image, stego)
                    matches = sum(1 for i in range(min(len(message), len(recovered))) 
                                 if message[i] == recovered[i])
                    recovery_acc = (matches / len(message)) * 100 if message else 0

                    results['proposed'] = {
                        'name': 'Proposed EDC-HC',
                        'psnr': round(psnr, 2),
                        'ssim': round(ssim, 4),
                        'recovery_accuracy': round(recovery_acc, 1),
                        'message_recovered': recovered[:100],
                        'message_matches': recovered.strip() == message.strip(),
                        'extra': f'{copies} copies, RS({RS_DATA_SYMBOLS},{RS_ECC_SYMBOLS})'
                    }
        except Exception as e:
            results['proposed'] = {'error': str(e)}

        # --- DCT no EC ---
        try:
            stego, num = dct_no_ec.embed(cover_image, message)
            extracted = dct_no_ec.extract(stego, num)
            recovered = bits_to_text(extracted)

            psnr = metrics.psnr(cover_image, stego)
            ssim = metrics.ssim_index(cover_image, stego)
            matches = sum(1 for i in range(min(len(message), len(recovered))) 
                         if message[i] == recovered[i])
            recovery_acc = (matches / len(message)) * 100 if message else 0

            results['dct_no_ec'] = {
                'name': 'DCT without EC',
                'psnr': round(psnr, 2),
                'ssim': round(ssim, 4),
                'recovery_accuracy': round(recovery_acc, 1),
                'message_recovered': recovered[:100],
                'message_matches': recovered.strip() == message.strip(),
                'extra': 'No error correction'
            }
        except Exception as e:
            results['dct_no_ec'] = {'error': str(e)}

        # --- LSB ---
        try:
            stego = lsb.embed(cover_image, message)
            original_bits = lsb.text_to_bits(message)
            extracted = lsb.extract(stego, len(original_bits))
            recovered = lsb.bits_to_text(extracted)

            psnr = metrics.psnr(cover_image, stego)
            ssim = metrics.ssim_index(cover_image, stego)
            matches = sum(1 for i in range(min(len(message), len(recovered))) 
                         if message[i] == recovered[i])
            recovery_acc = (matches / len(message)) * 100 if message else 0

            results['lsb'] = {
                'name': 'LSB Spatial',
                'psnr': round(psnr, 2),
                'ssim': round(ssim, 4),
                'recovery_accuracy': round(recovery_acc, 1),
                'message_recovered': recovered[:100],
                'message_matches': recovered.strip() == message.strip(),
                'extra': 'Spatial domain'
            }
        except Exception as e:
            results['lsb'] = {'error': str(e)}

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
    """Download stego image."""
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': 'Invalid filename'}), 404

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1.0',
        'rs_params': {
            'data_symbols': RS_DATA_SYMBOLS,
            'ecc_symbols': RS_ECC_SYMBOLS,
            'adaptive_copies': '1 or 2 based on image size'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Steganography Web Interface — EDC-HC (Reed-Solomon)")
    print("="*60)
    print(f"📊 Configuration:")
    print(f"   α_edge = {config.EMBEDDING_STRENGTH_EDGE}")
    print(f"   α_smooth = {config.EMBEDDING_STRENGTH_SMOOTH}")
    print(f"   RS({RS_DATA_SYMBOLS}, {RS_ECC_SYMBOLS}) with adaptive copies")
    print("="*60)
    print("🚀 Starting server at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)