"""
Hamming-based Steganography Test — Full Dataset Runner
======================================================
Runs DCT + Hamming(7,4) on ALL images in:
  - dataset/usc_sipi/
  - dataset/bossbase/

Compares against DCT-no-EC and LSB baselines.
Aggregates metrics across the full dataset.

Run:  python hamming_test.py
"""

import numpy as np
import cv2
import os
import json
import time
from datetime import datetime

import config
from text_utils import text_to_bits, bits_to_text
from error_correction import Hamming74
from embedding import DCTEmbedder
from extraction import DCTExtractor
from robustness_tests import TransformationTester
from evaluation import MetricsCalculator
from baselines import LSBSteganography, DCTWithoutEC


# =========================================================
# 500-character English secret message
# =========================================================
SECRET_MESSAGE = (
    "The rapid advancement of digital communication has created an urgent need for "
    "secure methods of transmitting sensitive information across public networks. "
    "Traditional encryption protects the content of messages but leaves visible "
    "evidence of communication, which is often sufficient to attract unwanted "
    "attention. Steganography addresses this limitation by concealing the very "
    "existence of the message within an innocuous carrier medium such as an image. "
    "When combined with error correction coding, modern steganographic techniques "
    "can ensure that hidden information survives the compression and transformation "
    "that occur routinely on social media platforms and messaging applications, "
    "enabling reliable text recovery for legitimate users who require confidentiality."
)


class HammingDatasetTest:
    """Runs Hamming(7,4) pipeline across all dataset images."""

    def __init__(self):
        self.embedder = DCTEmbedder(config)
        self.extractor = DCTExtractor(config)
        self.tester = TransformationTester(config)
        self.metrics = MetricsCalculator()
        self.lsb = LSBSteganography()
        self.dct_no_ec = DCTWithoutEC(config)

        self.original_bits = text_to_bits(SECRET_MESSAGE)
        print(f"📝 Secret message: {len(SECRET_MESSAGE)} chars / {len(self.original_bits)} bits")

    # =========================================================
    # IMAGE LOADING
    # =========================================================
    def load_usc_sipi(self):
        """Load all images from dataset/usc_sipi/."""
        folder = "dataset/usc_sipi"
        images = []
        if not os.path.exists(folder):
            print(f"⚠️  {folder} not found")
            return images

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append((fname, rgb))

        print(f"✅ Loaded {len(images)} USC-SIPI images from {folder}")
        return images

    def load_bossbase(self, max_images=100):
        """Load up to max_images from dataset/bossbase/."""
        folder = "dataset/bossbase"
        images = []
        if not os.path.exists(folder):
            print(f"⚠️  {folder} not found")
            return images

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if len(files) > max_images:
            # Sample evenly across the dataset
            step = len(files) // max_images
            files = files[::step][:max_images]

        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                images.append((fname, rgb))

        print(f"✅ Loaded {len(images)} BOSSBase images from {folder}")
        return images

    # =========================================================
    # PER-IMAGE TEST
    # =========================================================
    def test_single_image(self, image_name, cover_image, dataset_name):
        """Run Hamming + DCT no EC + LSB on a single image."""
        h, w = cover_image.shape[:2]

        result = {
            'image': image_name,
            'dataset': dataset_name,
            'size': f"{w}x{h}",
            'hamming': {},
            'dct_no_ec': {},
            'lsb': {}
        }

        # --------------------------------------------------
        # HAMMING (DCT + Hamming 7,4)
        # --------------------------------------------------
        try:
            encoded_bits, padding_info = Hamming74.encode_bitstream(self.original_bits)
            capacity = self.embedder.get_capacity(cover_image.shape)

            if len(encoded_bits) > capacity:
                result['hamming']['error'] = f"Message too large ({len(encoded_bits)} > {capacity} bits)"
            else:
                stego, _ = self.embedder.embed(cover_image, encoded_bits)
                extracted = self.extractor.extract(stego, len(encoded_bits))
                decoded, errors = Hamming74.decode_bitstream(extracted, padding_info)

                # Trim/pad
                if len(decoded) > len(self.original_bits):
                    decoded = decoded[:len(self.original_bits)]
                elif len(decoded) < len(self.original_bits):
                    decoded = decoded + [0] * (len(self.original_bits) - len(decoded))

                recovered = bits_to_text(decoded)

                baseline = self.metrics.compute_all_metrics(
                    cover_image, stego, self.original_bits, decoded,
                    SECRET_MESSAGE, recovered
                )

                # Transformations
                trans_results = {}
                for tname, timg in self.tester.apply_all_transformations(stego).items():
                    ext = self.extractor.extract(timg, len(encoded_bits))
                    dec, _ = Hamming74.decode_bitstream(ext, padding_info)
                    if len(dec) > len(self.original_bits):
                        dec = dec[:len(self.original_bits)]
                    elif len(dec) < len(self.original_bits):
                        dec = dec + [0] * (len(self.original_bits) - len(dec))
                    rec = bits_to_text(dec)
                    trans_results[tname] = self.metrics.compute_all_metrics(
                        cover_image, timg, self.original_bits, dec, SECRET_MESSAGE, rec
                    )

                result['hamming'] = {
                    'baseline': baseline,
                    'transformations': trans_results,
                    'errors_corrected': errors,
                    'encoded_bits': len(encoded_bits)
                }
        except Exception as e:
            result['hamming']['error'] = str(e)

        # --------------------------------------------------
        # DCT NO EC
        # --------------------------------------------------
        try:
            stego, num_bits = self.dct_no_ec.embed(cover_image, SECRET_MESSAGE)
            extracted = self.dct_no_ec.extract(stego, num_bits)
            if len(extracted) > len(self.original_bits):
                extracted = extracted[:len(self.original_bits)]
            elif len(extracted) < len(self.original_bits):
                extracted = extracted + [0] * (len(self.original_bits) - len(extracted))

            recovered = bits_to_text(extracted)
            baseline = self.metrics.compute_all_metrics(
                cover_image, stego, self.original_bits, extracted, SECRET_MESSAGE, recovered
            )

            trans_results = {}
            for tname, timg in self.tester.apply_all_transformations(stego).items():
                ext = self.dct_no_ec.extract(timg, num_bits)
                if len(ext) > len(self.original_bits):
                    ext = ext[:len(self.original_bits)]
                elif len(ext) < len(self.original_bits):
                    ext = ext + [0] * (len(self.original_bits) - len(ext))
                rec = bits_to_text(ext)
                trans_results[tname] = self.metrics.compute_all_metrics(
                    cover_image, timg, self.original_bits, ext, SECRET_MESSAGE, rec
                )

            result['dct_no_ec'] = {'baseline': baseline, 'transformations': trans_results}
        except Exception as e:
            result['dct_no_ec']['error'] = str(e)

        # --------------------------------------------------
        # LSB
        # --------------------------------------------------
        try:
            stego = self.lsb.embed(cover_image, SECRET_MESSAGE)
            extracted = self.lsb.extract(stego, len(self.original_bits))
            if len(extracted) > len(self.original_bits):
                extracted = extracted[:len(self.original_bits)]
            elif len(extracted) < len(self.original_bits):
                extracted = extracted + [0] * (len(self.original_bits) - len(extracted))

            recovered = self.lsb.bits_to_text(extracted)
            baseline = self.metrics.compute_all_metrics(
                cover_image, stego, self.original_bits, extracted, SECRET_MESSAGE, recovered
            )

            trans_results = {}
            for tname, timg in self.tester.apply_all_transformations(stego).items():
                ext = self.lsb.extract(timg, len(self.original_bits))
                if len(ext) > len(self.original_bits):
                    ext = ext[:len(self.original_bits)]
                elif len(ext) < len(self.original_bits):
                    ext = ext + [0] * (len(self.original_bits) - len(ext))
                rec = self.lsb.bits_to_text(ext)
                trans_results[tname] = self.metrics.compute_all_metrics(
                    cover_image, timg, self.original_bits, ext, SECRET_MESSAGE, rec
                )

            result['lsb'] = {'baseline': baseline, 'transformations': trans_results}
        except Exception as e:
            result['lsb']['error'] = str(e)

        return result

    # =========================================================
    # AGGREGATION
    # =========================================================
    def aggregate(self, all_results):
        """Compute averages across all images."""
        agg = {}

        for method in ['hamming', 'dct_no_ec', 'lsb']:
            valid = [r for r in all_results if method in r and 'baseline' in r[method]]

            if not valid:
                agg[method] = {'count': 0, 'error': 'No valid results'}
                continue

            # Baseline averages
            psnr_vals = [r[method]['baseline']['psnr'] for r in valid]
            ber_vals = [r[method]['baseline']['ber'] for r in valid]
            rec_vals = [r[method]['baseline']['recovery_accuracy'] for r in valid]

            method_agg = {
                'count': len(valid),
                'baseline': {
                    'psnr_mean': float(np.mean(psnr_vals)),
                    'psnr_std': float(np.std(psnr_vals)),
                    'ber_mean': float(np.mean(ber_vals)),
                    'ber_std': float(np.std(ber_vals)),
                    'recovery_mean': float(np.mean(rec_vals)),
                    'recovery_std': float(np.std(rec_vals)),
                },
                'transformations': {}
            }

            # Transformation averages
            trans_names = set()
            for r in valid:
                trans_names.update(r[method]['transformations'].keys())

            for tname in sorted(trans_names):
                t_vals = [r[method]['transformations'][tname] for r in valid
                          if tname in r[method]['transformations']]
                if t_vals:
                    method_agg['transformations'][tname] = {
                        'psnr_mean': float(np.mean([t['psnr'] for t in t_vals])),
                        'ber_mean': float(np.mean([t['ber'] for t in t_vals])),
                        'recovery_mean': float(np.mean([t['recovery_accuracy'] for t in t_vals])),
                    }

            agg[method] = method_agg

        return agg

    # =========================================================
    # MAIN RUN
    # =========================================================
    def run(self, max_boss=100, use_usc=True, use_boss=True):
        print("\n" + "=" * 70)
        print("HAMMING(7,4) STEGANOGRAPHY — FULL DATASET RUN")
        print("=" * 70)

        all_images = []

        if use_usc:
            usc = self.load_usc_sipi()
            all_images.extend([(name, img, 'USC-SIPI') for name, img in usc])

        if use_boss:
            boss = self.load_bossbase(max_images=max_boss)
            all_images.extend([(name, img, 'BOSSBase') for name, img in boss])

        print(f"\n📊 Total images to process: {len(all_images)}")
        print("=" * 70)

        if not all_images:
            print("❌ No images found. Check dataset folders.")
            return

        all_results = []
        start = time.time()

        for idx, (name, img, ds) in enumerate(all_images, 1):
            elapsed = time.time() - start
            eta = (elapsed / idx) * (len(all_images) - idx) if idx > 1 else 0
            print(f"\n[{idx}/{len(all_images)}] {ds}/{name} ({img.shape[1]}x{img.shape[0]})  "
                  f"[ETA: {eta/60:.1f} min]")

            result = self.test_single_image(name, img, ds)
            all_results.append(result)

            if 'baseline' in result['hamming']:
                b = result['hamming']['baseline']
                print(f"   Hamming: PSNR={b['psnr']:.2f} dB, "
                      f"BER={b['ber']:.4f}, Recovery={b['recovery_accuracy']:.1f}%")
            elif 'error' in result['hamming']:
                print(f"   Hamming: {result['hamming']['error']}")

        # Aggregate
        print("\n" + "=" * 70)
        print("AGGREGATING RESULTS...")
        print("=" * 70)
        agg = self.aggregate(all_results)

        # Save
        self.save_results(all_results, agg)
        self.print_final_summary(agg)

        return all_results, agg

    def print_final_summary(self, agg):
        print("\n" + "=" * 80)
        print("FINAL AVERAGE RESULTS ACROSS FULL DATASET")
        print("=" * 80)

        for method, label in [('hamming', 'Hamming (DCT)'),
                              ('dct_no_ec', 'DCT no EC'),
                              ('lsb', 'LSB')]:
            m = agg.get(method, {})
            if m.get('count', 0) == 0:
                print(f"\n{label}: No valid results")
                continue

            b = m['baseline']
            print(f"\n{label} ({m['count']} images):")
            print(f"   PSNR:  {b['psnr_mean']:.2f} ± {b['psnr_std']:.2f} dB")
            print(f"   BER:   {b['ber_mean']:.4f} ± {b['ber_std']:.4f}")
            print(f"   Recovery: {b['recovery_mean']:.1f} ± {b['recovery_std']:.1f}%")

            print(f"   Under transformations:")
            for tname in sorted(m['transformations'].keys()):
                t = m['transformations'][tname]
                print(f"      {tname:<16} Rec={t['recovery_mean']:5.1f}%  "
                      f"BER={t['ber_mean']:.4f}  PSNR={t['psnr_mean']:.2f}")

        print("\n" + "=" * 80)

    def save_results(self, all_results, agg):
        os.makedirs('results', exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def ser(o):
            if isinstance(o, dict): return {k: ser(v) for k, v in o.items()}
            if isinstance(o, list): return [ser(v) for v in o]
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32, np.int64)): return int(o)
            if isinstance(o, np.bool_): return bool(o)
            return o

        # Per-image detailed results
        detail_path = f"results/hamming_dataset_{ts}.json"
        with open(detail_path, 'w') as f:
            json.dump(ser({'results': all_results, 'aggregate': agg}), f, indent=2)
        print(f"\n✓ Detailed results: {detail_path}")

        # Summary CSV
        csv_path = f"results/hamming_summary_{ts}.csv"
        with open(csv_path, 'w') as f:
            f.write("Method,PSNR_mean,PSNR_std,BER_mean,BER_std,Recovery_mean,Recovery_std\n")
            for method, label in [('hamming', 'Hamming'),
                                  ('dct_no_ec', 'DCT_no_EC'),
                                  ('lsb', 'LSB')]:
                m = agg.get(method, {})
                if m.get('count', 0) == 0:
                    continue
                b = m['baseline']
                f.write(f"{label},{b['psnr_mean']:.2f},{b['psnr_std']:.2f},"
                        f"{b['ber_mean']:.4f},{b['ber_std']:.4f},"
                        f"{b['recovery_mean']:.1f},{b['recovery_std']:.1f}\n")
        print(f"✓ Summary CSV: {csv_path}")


def main():
    print("\n" + "=" * 70)
    print("HAMMING(7,4) FULL DATASET TEST")
    print("=" * 70)
    print(f"📁 USC-SIPI: dataset/usc_sipi/")
    print(f"📁 BOSSBase: dataset/bossbase/")
    print(f"📝 Secret: 500-char English paragraph")
    print("=" * 70)

    test = HammingDatasetTest()

    # Configure: how many BOSS images to use
    # Set max_boss=None to use all 10,000 (takes hours!)
    # Set max_boss=100 for a balanced test set
    test.run(max_boss=100, use_usc=True, use_boss=True)


if __name__ == "__main__":
    main()
