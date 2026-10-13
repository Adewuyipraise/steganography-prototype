"""Main simulation script - FULL DATASET VERSION"""

import numpy as np
import cv2
import os
import json
from datetime import datetime

import config
from text_utils import text_to_bits, bits_to_text
from reed_solomon import TurboReedSolomon
from embedding import DCTEmbedder
from extraction import DCTExtractor
from robustness_tests import TransformationTester
from evaluation import MetricsCalculator
from baselines import LSBSteganography, DCTWithoutEC
from steganalysis import Steganalysis
from dataset_loader import DatasetLoader


class SteganographySimulation:
    def __init__(self):
        self.embedder = DCTEmbedder(config)
        self.extractor = DCTExtractor(config)
        self.tester = TransformationTester(config)
        self.metrics = MetricsCalculator()
        self.lsb = LSBSteganography()
        self.dct_no_ec = DCTWithoutEC(config)
        self.loader = DatasetLoader(seed=42)
        
        # Initialize Reed-Solomon error correction
        # Can correct up to 16 errors per 255-byte block
        self.rs = TurboReedSolomon(
            data_symbols=config.RS_DATA_SYMBOLS,
            ecc_symbols=config.RS_ECC_SYMBOLS,
            interleave_depth=config.RS_INTERLEAVE_DEPTH
        )

    def run_on_image(self, cover_image, secret_text, image_name="unknown"):
        """Run all methods on a single image"""
        original_bits = text_to_bits(secret_text)
        
        results = {}
        methods = [
            ('Proposed (DCT+Reed-Solomon)', self._test_proposed),
            ('DCT without EC', self._test_dct_no_ec),
            ('LSB Spatial', self._test_lsb)
        ]

        for name, test_func in methods:
            try:
                results[name] = test_func(cover_image, secret_text, original_bits)
            except Exception as e:
                print(f"   ❌ Error in {name}: {e}")
                results[name] = {'baseline': {'error': str(e)}, 'transformations': {}}

        return results

    

    def run_full_comparison(self, cover_image_path, secret_text):
        print("\n" + "=" * 70)
        print("FULL COMPARATIVE STUDY: Proposed vs LSB vs DCT without EC")
        print("=" * 70)

        cover_image = self.load_or_create_image(cover_image_path)
        original_bits = text_to_bits(secret_text)

        print(f"\n📝 Secret text: '{secret_text[:50]}...' ({len(secret_text)} chars)")
        print(f"📊 Original bits: {len(original_bits)}")

        results = {}
        methods = [
            ('Proposed (DCT+Reed-Solomon)', self._test_proposed),
            ('DCT without EC', self._test_dct_no_ec),
            ('LSB Spatial', self._test_lsb)
        ]

        for name, test_func in methods:
            print("\n" + "=" * 50)
            print(f"METHOD: {name}")
            print("=" * 50)
            results[name] = test_func(cover_image, secret_text, original_bits)

        self._print_summary(results)
        self._print_single_image_transformation_results(results)
        self._save_results(results)
        return results
    
    def _print_single_image_transformation_results(self, results):
        """Print detailed transformation results for single image"""
        print("\n" + "=" * 100)
        print("DETAILED TRANSFORMATION RESULTS (Single Image)")
        print("=" * 100)
        
        for method_name, method_data in results.items():
            print(f"\n📊 {method_name}")
            print("-" * 80)
            
            # Get baseline
            baseline = method_data['baseline']
            print(f"  Baseline (No Transform): Recovery: {baseline['recovery_accuracy']:.1f}%, BER: {baseline['ber']:.4f}, PSNR: {baseline['psnr']:.2f} dB")
            
            # Get transformations
            trans = method_data.get('transformations', {})
            if trans:
                print(f"\n  {'Transformation':<20} | {'Recovery %':<12} | {'BER':<10} | {'PSNR':<10}")
                print(f"  {'-'*20} | {'-'*12} | {'-'*10} | {'-'*10}")
                
                # Group transformations by type
                jpeg_results = {k: v for k, v in trans.items() if k.startswith('jpeg_')}
                resize_results = {k: v for k, v in trans.items() if k.startswith('resize_')}
                crop_results = {k: v for k, v in trans.items() if k.startswith('crop_')}
                noise_results = {k: v for k, v in trans.items() if k.startswith('noise_')}
                combined_results = {k: v for k, v in trans.items() if k.startswith('combined_')}
                
                # Print JPEG results
                if jpeg_results:
                    print(f"\n  JPEG Compression:")
                    for name, metrics in sorted(jpeg_results.items()):
                        rec = metrics.get('recovery_accuracy', 0)
                        ber = metrics.get('ber', 1)
                        psnr = metrics.get('psnr', 0)
                        print(f"    {name:<16} | {rec:<12.1f} | {ber:<10.4f} | {psnr:<10.2f}")
                
                # Print Resize results
                if resize_results:
                    print(f"\n  Resizing:")
                    for name, metrics in sorted(resize_results.items()):
                        rec = metrics.get('recovery_accuracy', 0)
                        ber = metrics.get('ber', 1)
                        psnr = metrics.get('psnr', 0)
                        print(f"    {name:<16} | {rec:<12.1f} | {ber:<10.4f} | {psnr:<10.2f}")
                
                # Print Crop results
                if crop_results:
                    print(f"\n  Cropping:")
                    for name, metrics in sorted(crop_results.items()):
                        rec = metrics.get('recovery_accuracy', 0)
                        ber = metrics.get('ber', 1)
                        psnr = metrics.get('psnr', 0)
                        print(f"    {name:<16} | {rec:<12.1f} | {ber:<10.4f} | {psnr:<10.2f}")
                
                # Print Noise results
                if noise_results:
                    print(f"\n  Gaussian Noise:")
                    for name, metrics in sorted(noise_results.items()):
                        rec = metrics.get('recovery_accuracy', 0)
                        ber = metrics.get('ber', 1)
                        psnr = metrics.get('psnr', 0)
                        print(f"    {name:<16} | {rec:<12.1f} | {ber:<10.4f} | {psnr:<10.2f}")
                
                # Print Combined results
                if combined_results:
                    print(f"\n  Combined Transformations:")
                    for name, metrics in sorted(combined_results.items()):
                        rec = metrics.get('recovery_accuracy', 0)
                        ber = metrics.get('ber', 1)
                        psnr = metrics.get('psnr', 0)
                        print(f"    {name:<16} | {rec:<12.1f} | {ber:<10.4f} | {psnr:<10.2f}")
            else:
                print("  No transformation results available")

    def run_on_dataset(self, secret_text, num_images=10):
        """Run experiments on USC-SIPI and BOSSBase datasets"""
        print("\n" + "=" * 70)
        print("RUNNING ON FULL DATASET")
        print("=" * 70)
        
        all_results = {}
        
        # Load USC-SIPI
        print("\n📁 Loading USC-SIPI dataset...")
        usc_images = self.loader.load_usc_sipi("dataset/usc_sipi/")
        print(f"   Loaded {len(usc_images)} USC-SIPI images")
        
        # Load BOSSBase
        print("\n📁 Loading BOSSBase dataset...")
        boss_images = self.loader.load_bossbase("dataset/bossbase/", num_images=100)
        print(f"   Loaded {len(boss_images)} BOSSBase images")
        
        # Combine datasets
        all_images = usc_images + boss_images
        print(f"\n📊 Total images: {len(all_images)}")
        
        # Split into tuning, validation, test (as per Chapter 3)
        split = self.loader.split_dataset(all_images, tuningset_size=200, valset_size=50, testset_size=50)
        
        print("\n🔬 Running experiments on TEST SET...")
        test_set = split['test']
        print(f"   Test set size: {len(test_set)} images")
        
        all_method_results = {
            'proposed': {'baseline': [], 'transformations': []},
            'dct_no_ec': {'baseline': [], 'transformations': []},
            'lsb': {'baseline': [], 'transformations': []}
        }
        
        # Run on each test image
        for idx, img in enumerate(test_set[:num_images]):  # Limit for speed
            print(f"\n   Testing image {idx+1}/{min(len(test_set), num_images)}...")
            
            # Get image name if available
            img_name = f"image_{idx+1}"
            
            # Run on this image
            try:
                original_bits = text_to_bits(secret_text)
                
                # Test each method
                for method_name, test_func in [
                    ('proposed', self._test_proposed),
                    ('dct_no_ec', self._test_dct_no_ec),
                    ('lsb', self._test_lsb)
                ]:
                    result = test_func(img, secret_text, original_bits)
                    all_method_results[method_name]['baseline'].append(result['baseline'])
                    all_method_results[method_name]['transformations'].append(result['transformations'])
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        # Calculate averages
        print("\n📊 AVERAGE RESULTS ACROSS DATASET:")
        print("=" * 70)
        
        for method_name in ['proposed', 'dct_no_ec', 'lsb']:
            baselines = all_method_results[method_name]['baseline']
            if baselines:
                avg_psnr = np.mean([b['psnr'] for b in baselines])
                avg_ber = np.mean([b['ber'] for b in baselines])
                avg_recovery = np.mean([b['recovery_accuracy'] for b in baselines])
                
                name_map = {
                    'proposed': 'Proposed (DCT+Hamming)',
                    'dct_no_ec': 'DCT without EC',
                    'lsb': 'LSB Spatial'
                }
                
                print(f"\n{name_map[method_name]}:")
                print(f"   Avg PSNR: {avg_psnr:.2f} dB")
                print(f"   Avg BER: {avg_ber:.4f}")
                print(f"   Avg Recovery: {avg_recovery:.1f}%")
        
        # PRINT DETAILED TRANSFORMATION RESULTS
        self._print_detailed_transformation_results(all_method_results)
        
        return all_method_results
    
    def _print_detailed_transformation_results(self, all_method_results):
        """Print detailed transformation results across all images"""
        print("\n" + "=" * 100)
        print("DETAILED TRANSFORMATION RESULTS (Average Across All Images)")
        print("=" * 100)
        
        for method_name, name in [('proposed', 'Proposed EDC-HC'), 
                                 ('dct_no_ec', 'DCT without EC'), 
                                 ('lsb', 'LSB Spatial')]:
            # Get all transformation results for this method
            all_trans = all_method_results[method_name]['transformations']
            
            if all_trans:
                # Get transformation names from first image
                trans_names = list(all_trans[0].keys()) if all_trans else []
                
                print(f"\n📊 {method_name.upper()}:")
                print("-" * 80)
                print(f"  {'Transformation':<20} | {'Recovery %':<12} | {'BER':<10} | {'PSNR':<10}")
                print(f"  {'-'*20} | {'-'*12} | {'-'*10} | {'-'*10}")
                
                for trans_name in sorted(trans_names):
                    # Collect all values for this transformation across images
                    rec_values = []
                    ber_values = []
                    psnr_values = []
                    
                    for img_trans in all_trans:
                        if trans_name in img_trans:
                            rec_values.append(img_trans[trans_name].get('recovery_accuracy', 0))
                            ber_values.append(img_trans[trans_name].get('ber', 1))
                            psnr_values.append(img_trans[trans_name].get('psnr', 0))
                    
                    if rec_values:
                        avg_rec = np.mean(rec_values)
                        avg_ber = np.mean(ber_values)
                        avg_psnr = np.mean(psnr_values)
                        
                        # Shorten transformation name for display
                        display_name = trans_name[:20]
                        print(f"    {display_name:<18} | {avg_rec:<12.1f} | {avg_ber:<10.4f} | {avg_psnr:<10.2f}")
        
        # Also print baseline comparison
        print("\n" + "=" * 100)
        print("BASELINE VS TRANSFORMATION COMPARISON")
        print("=" * 100)
        
        for method_name, name in [('proposed', 'Proposed EDC-HC'), 
                                 ('dct_no_ec', 'DCT without EC'), 
                                 ('lsb', 'LSB Spatial')]:
            baselines = all_method_results[method_name]['baseline']
            trans_all = all_method_results[method_name]['transformations']
            
            if baselines:
                avg_baseline_rec = np.mean([b['recovery_accuracy'] for b in baselines])
                
                # Calculate average recovery across all transformations
                all_trans_rec = []
                for img_trans in trans_all:
                    for trans_name, metrics in img_trans.items():
                        all_trans_rec.append(metrics.get('recovery_accuracy', 0))
                
                if all_trans_rec:
                    avg_trans_rec = np.mean(all_trans_rec)
                    drop = avg_baseline_rec - avg_trans_rec
                    
                    print(f"\n{name}:")
                    print(f"  Baseline Recovery: {avg_baseline_rec:.1f}%")
                    print(f"  Avg Recovery Under Transformations: {avg_trans_rec:.1f}%")
                    print(f"  Performance Drop: {drop:.1f}%")

    def _test_proposed(self, cover_image, secret_text, original_bits):
        """Test the proposed method: DCT + JPEG-guard grid + RS(63,192) + repetition"""
        print("\n" + "=" * 50)
        print("METHOD: Proposed (DCT + JPEG-guard grid + RS + repetition)")
        print("=" * 50)

        # Robust round-trip sanity check on THIS image
        max_msg = self.embedder.max_message_bits(cover_image.shape)
        if len(original_bits) > max_msg:
            print(f"   ⚠️ Message too long: {len(original_bits)} bits > {max_msg}-bit capacity")
            print(f"   Truncating message to fit the frame.")
            original_bits = original_bits[:max_msg]

        test_stego, _ = self.embedder.embed_robust(cover_image, original_bits)
        test_recovered = self.extractor.extract_robust(test_stego)
        test_ok = (test_recovered is not None and
                   len(test_recovered) == len(original_bits) and
                   all(a == b for a, b in zip(original_bits, test_recovered)))
        print(f"\n🔍 Robust round-trip test: {'✅ OK' if test_ok else '❌ FAILED'}")
        if not test_ok:
            print("   ⚠️ WARNING: Round-trip failing on THIS image!")
        print()

        # Baseline embed / extract / decode
        stego, _ = self.embedder.embed_robust(cover_image, original_bits)
        decoded_bits = self.extractor.extract_robust(stego)
        if decoded_bits is None:
            decoded_bits = [0] * len(original_bits)

        recovered = bits_to_text(decoded_bits)

        print(f"   Original bits: {len(original_bits)}")
        print(f"   Recovered: '{recovered[:80]}...'")

        baseline = self.metrics.compute_all_metrics(
            cover_image, stego, original_bits, decoded_bits, secret_text, recovered
        )
        print(f"   PSNR: {baseline['psnr']:.2f} dB, BER: {baseline['ber']:.4f}, "
              f"Recovery: {baseline['recovery_accuracy']:.1f}%")

        # Test transformations
        print("   Testing transformations...")
        trans_results = {}
        transformed_images = self.tester.apply_all_transformations(stego)

        for trans_name, trans_img in transformed_images.items():
            decoded_bits = self.extractor.extract_robust(trans_img)
            if decoded_bits is None:
                decoded_bits = [0] * len(original_bits)

            recovered = bits_to_text(decoded_bits)
            trans_results[trans_name] = self.metrics.compute_all_metrics(
                cover_image, trans_img, original_bits, decoded_bits, secret_text, recovered
            )

        return {
            'baseline': baseline,
            'transformations': trans_results,
            'steganalysis': Steganalysis.chi_square_test(cover_image, stego)
        }

    def _test_dct_no_ec(self, cover_image, secret_text, original_bits):
        stego, num_bits = self.dct_no_ec.embed(cover_image, secret_text)
        extracted = self.dct_no_ec.extract(stego, num_bits)
        
        if len(extracted) > len(original_bits):
            extracted = extracted[:len(original_bits)]
        elif len(extracted) < len(original_bits):
            extracted = extracted + [0] * (len(original_bits) - len(extracted))
        
        recovered = bits_to_text(extracted)

        baseline = self.metrics.compute_all_metrics(
            cover_image, stego, original_bits, extracted, secret_text, recovered
        )

        trans_results = {}
        for name, img in self.tester.apply_all_transformations(stego).items():
            extracted = self.dct_no_ec.extract(img, num_bits)
            if len(extracted) > len(original_bits):
                extracted = extracted[:len(original_bits)]
            elif len(extracted) < len(original_bits):
                extracted = extracted + [0] * (len(original_bits) - len(extracted))
            recovered = bits_to_text(extracted)
            trans_results[name] = self.metrics.compute_all_metrics(
                cover_image, img, original_bits, extracted, secret_text, recovered
            )

        return {
            'baseline': baseline,
            'transformations': trans_results,
            'steganalysis': Steganalysis.chi_square_test(cover_image, stego)
        }

    def _test_lsb(self, cover_image, secret_text, original_bits):
        stego = self.lsb.embed(cover_image, secret_text)
        num_bits = len(original_bits)
        extracted = self.lsb.extract(stego, num_bits)
        
        if len(extracted) > len(original_bits):
            extracted = extracted[:len(original_bits)]
        elif len(extracted) < len(original_bits):
            extracted = extracted + [0] * (len(original_bits) - len(extracted))
        
        recovered = self.lsb.bits_to_text(extracted)

        baseline = self.metrics.compute_all_metrics(
            cover_image, stego, original_bits, extracted, secret_text, recovered
        )

        trans_results = {}
        for name, img in self.tester.apply_all_transformations(stego).items():
            extracted = self.lsb.extract(img, num_bits)
            if len(extracted) > len(original_bits):
                extracted = extracted[:len(original_bits)]
            elif len(extracted) < len(original_bits):
                extracted = extracted + [0] * (len(original_bits) - len(extracted))
            recovered = self.lsb.bits_to_text(extracted)
            trans_results[name] = self.metrics.compute_all_metrics(
                cover_image, img, original_bits, extracted, secret_text, recovered
            )

        return {
            'baseline': baseline,
            'transformations': trans_results,
            'steganalysis': Steganalysis.chi_square_test(cover_image, stego)
        }

    def load_or_create_image(self, image_path):
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print("Creating synthetic image...")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img[i, :, 0] = int(128 + 127 * np.sin(i / 50))
            img[i, :, 1] = int(128 + 127 * np.cos(i / 40))
            img[:, i, 2] = int(128 + 127 * np.sin(i / 30))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    def _print_summary(self, results):
        print("\n" + "=" * 100)
        header = f"{'Method':<25} | {'Avg Recovery':<14} | {'Avg BER':<10} | {'PSNR':<8}"
        print(header)
        print("-" * 75)

        for name, res in results.items():
            trans = list(res['transformations'].values())
            avg_rec = np.mean([t['recovery_accuracy'] for t in trans]) if trans else 0
            avg_ber = np.mean([t['ber'] for t in trans]) if trans else 1
            psnr = res['baseline']['psnr']
            print(f"{name:<25} | {avg_rec:<14.1f}% | {avg_ber:<10.4f} | {psnr:<8.2f}")
        print("=" * 100)

    def _save_results(self, results):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/comparison_{timestamp}.json"
        
        def serialize(obj):
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        with open(filename, 'w') as f:
            json.dump(serialize(results), f, indent=4)
        print(f"\n✓ Results saved to {filename}")


def main():
    print("\n" + "=" * 60)
    print("ROBUST STEGANOGRAPHY COMPARATIVE STUDY")
    print("EDC-HC (Edge-guided DCT with Hamming Code)")
    print("=" * 60)

    sim = SteganographySimulation()
    secret_text = "Hello, this is a secret message for robustness testing! The quick brown fox jumps over the lazy dog."

    # =================== OPTIONS ===================
    # Choose ONE option to run:
    
    # OPTION 1: Run on single image (quick, for testing)
    print("\n📊 Running on single image...")
    image_path = "dataset/sample_images/cover.jpg"
    sim.run_full_comparison(image_path, secret_text)
    
    # OPTION 2: Run on full dataset (USC-SIPI + BOSSBase) - slower
    # print("\n📊 Running on full dataset...")
    # sim.run_on_dataset(secret_text, num_images=10)  # Use 10 images for testing, change to 50 for full
    # ===============================================

    print("\n" + "=" * 60)
    print("✅ COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()