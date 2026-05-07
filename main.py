"""Main simulation script with full baseline comparison""" 

import numpy as np 
import cv2 
import os 
import json
from datetime import datetime 

import config 
from error_correction import Hamming74 
from embedding import DCTEmbedder 
from extraction import DCTExtractor 
from robustness_tests import TransformationTester 
from evaluation import MetricsCalculator 
from baselines import LSBSteganography, DCTWithoutEC 

class SteganographySimulation: 
    """Main simulation controller with baseline comparisons""" 
    
    def __init__(self): 
        self.config = config 
        self.embedder = DCTEmbedder(config) 
        self.extractor = DCTExtractor(config) 
        self.tester = TransformationTester(config) 
        self.metrics = MetricsCalculator() 
        
        # Initialize baselines 
        self.lsb = LSBSteganography() 
        self.dct_no_ec = DCTWithoutEC(config) 
        
        self.results = {} 
    
    def text_to_bits(self, text): 
        bits = [] 
        for char in text: 
            ascii_val = ord(char) 
            for i in range(7, -1, -1): 
                bits.append((ascii_val >> i) & 1) 
        return bits 
    
    def bits_to_text(self, bits): 
        if len(bits) % 8 != 0: 
            bits = bits[:-(len(bits) % 8)] 
        chars = [] 
        for i in range(0, len(bits), 8): 
            byte = bits[i:i+8] 
            ascii_val = sum(bit << (7 - j) for j, bit in enumerate(byte)) 
            chars.append(chr(ascii_val)) 
        return ''.join(chars) 
    
    def load_or_create_image(self, image_path): 
        if os.path.exists(image_path): 
            img = cv2.imread(image_path) 
            if img is not None: 
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        print(f"Creating synthetic image...") 
        img = np.zeros((512, 512, 3), dtype=np.uint8) 
        for i in range(512): 
            img[i, :, 0] = int(128 + 127 * np.sin(i/50)) 
            img[i, :, 1] = int(128 + 127 * np.cos(i/40)) 
            img[:, i, 2] = int(128 + 127 * np.sin(i/30)) 
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True) 
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
        return img 
    
    def run_full_comparison(self, cover_image_path, secret_text): 
        """ 
        Run all three methods and compare results 
        """ 
        print("\n" + "="*70) 
        print("FULL COMPARATIVE STUDY: Proposed vs LSB vs DCT without EC") 
        print("="*70) 
        
        cover_image = self.load_or_create_image(cover_image_path) 
        original_bits = self.text_to_bits(secret_text)
        
        methods_results = {}

        # 1. Proposed Method 
        print("\n" + "="*50) 
        print("METHOD 1: Proposed (DCT + Hamming + Edge Adaptation)") 
        print("="*50) 
        
        encoded_bits, padding_info = Hamming74.encode_bitstream(original_bits) 
        capacity = self.embedder.get_capacity(cover_image.shape) 
        if len(encoded_bits) > capacity: 
            encoded_bits = encoded_bits[:capacity] 
        
        stego_proposed, _ = self.embedder.embed(cover_image, encoded_bits) 
        cv2.imwrite("stego_proposed.png", cv2.cvtColor(stego_proposed, cv2.COLOR_RGB2BGR)) 
        
        ext_proposed = self.extractor.extract(stego_proposed, len(encoded_bits)) 
        dec_proposed, _ = Hamming74.decode_bitstream(ext_proposed, padding_info) 
        rec_text_proposed = self.bits_to_text(dec_proposed) 
        
        proposed_baseline = self.metrics.compute_all_metrics( 
            cover_image, stego_proposed, original_bits, dec_proposed, secret_text, rec_text_proposed 
        )
        
        print(f"Baseline PSNR: {proposed_baseline['psnr']:.2f} dB, BER: {proposed_baseline['ber']:.4f}")
        
        proposed_trans = {}
        transformed_images = self.tester.apply_all_transformations(stego_proposed)
        for name, img in transformed_images.items():
            ext = self.extractor.extract(img, len(encoded_bits))
            dec, _ = Hamming74.decode_bitstream(ext, padding_info)
            proposed_trans[name] = self.metrics.compute_all_metrics(
                cover_image, img, original_bits, dec, secret_text, self.bits_to_text(dec)
            )
        
        methods_results['proposed'] = {'baseline': proposed_baseline, 'transformations': proposed_trans}

        # 2. DCT Without EC
        print("\n" + "="*50) 
        print("METHOD 2: DCT Without Error Correction") 
        print("="*50) 
        
        stego_dct, num_bits_dct = self.dct_no_ec.embed(cover_image, secret_text)
        cv2.imwrite("stego_dct_no_ec.png", cv2.cvtColor(stego_dct, cv2.COLOR_RGB2BGR))
        
        ext_dct = self.dct_no_ec.extract(stego_dct, num_bits_dct)
        rec_text_dct = self.dct_no_ec.bits_to_text(ext_dct)
        
        dct_baseline = self.metrics.compute_all_metrics(
            cover_image, stego_dct, original_bits[:num_bits_dct], ext_dct, secret_text, rec_text_dct
        )
        print(f"Baseline PSNR: {dct_baseline['psnr']:.2f} dB, BER: {dct_baseline['ber']:.4f}")
        
        dct_trans = {}
        transformed_images_dct = self.tester.apply_all_transformations(stego_dct)
        for name, img in transformed_images_dct.items():
            ext = self.dct_no_ec.extract(img, num_bits_dct)
            dct_trans[name] = self.metrics.compute_all_metrics(
                cover_image, img, original_bits[:num_bits_dct], ext, secret_text, self.dct_no_ec.bits_to_text(ext)
            )
        
        methods_results['dct_no_ec'] = {'baseline': dct_baseline, 'transformations': dct_trans}

        # 3. LSB
        print("\n" + "="*50) 
        print("METHOD 3: LSB Spatial Domain") 
        print("="*50) 
        
        stego_lsb = self.lsb.embed(cover_image, secret_text)
        cv2.imwrite("stego_lsb.png", cv2.cvtColor(stego_lsb, cv2.COLOR_RGB2BGR))
        
        num_bits_lsb = len(original_bits)
        ext_lsb = self.lsb.extract(stego_lsb, num_bits_lsb)
        rec_text_lsb = self.lsb.bits_to_text(ext_lsb)
        
        lsb_baseline = self.metrics.compute_all_metrics(
            cover_image, stego_lsb, original_bits, ext_lsb, secret_text, rec_text_lsb
        )
        print(f"Baseline PSNR: {lsb_baseline['psnr']:.2f} dB, BER: {lsb_baseline['ber']:.4f}")
        
        lsb_trans = {}
        transformed_images_lsb = self.tester.apply_all_transformations(stego_lsb)
        for name, img in transformed_images_lsb.items():
            ext = self.lsb.extract(img, num_bits_lsb)
            lsb_trans[name] = self.metrics.compute_all_metrics(
                cover_image, img, original_bits, ext, secret_text, self.lsb.bits_to_text(ext)
            )
        
        methods_results['lsb'] = {'baseline': lsb_baseline, 'transformations': lsb_trans}

        self.print_comparison_summary(methods_results)
        return methods_results

    def print_comparison_summary(self, results):
        print("\n" + "="*70)
        print(f"{'Method':<30} | {'Avg Recovery':<15} | {'Avg BER':<10} | {'PSNR':<8}")
        print("-" * 70)
        
        for name, res in results.items():
            avg_rec = np.mean([t['recovery_accuracy'] for t in res['transformations'].values()])
            avg_ber = np.mean([t['ber'] for t in res['transformations'].values()])
            psnr = res['baseline']['psnr']
            print(f"{name:<30} | {avg_rec:<15.1f}% | {avg_ber:<10.4f} | {psnr:<8.2f}")
        print("="*70)

    def save_comparison_results(self, results, output_dir="results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/comparison_{timestamp}.json"
        
        # Helper to make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(filename, 'w') as f:
            json.dump(make_serializable(results), f, indent=4)
        print(f"\n✓ Comparison results saved to {filename}")

def main(): 
    print("\n" + "="*60) 
    print("ROBUST STEGANOGRAPHY COMPARATIVE STUDY") 
    print("="*60) 
    
    sim = SteganographySimulation() 
    secret_text = "Hello, this is a secret message for robustness testing! Comparing different methods." 
    image_path = "dataset/sample_images/cover.jpg" 
    
    results = sim.run_full_comparison(image_path, secret_text) 
    sim.save_comparison_results(results)

if __name__ == "__main__": 
    main()
