"""Diagnostic: Test embed → extract round-trip on cover.jpg"""

import numpy as np
import cv2
import config
import text_utils
from embedding import DCTEmbedder
from extraction import DCTExtractor
from scipy.fftpack import dct


def diagnose():
    print("=" * 60)
    print("DIAGNOSTIC: Round-trip on cover.jpg")
    print("=" * 60)
    
    # Load cover image
    img = cv2.imread("dataset/sample_images/cover.jpg")
    if img is None:
        print("❌ Could not load cover.jpg")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"\n📷 Image shape: {img.shape}")
    
    # Short test message
    message = "Hello Test"
    bits = text_utils.text_to_bits(message)
    print(f"📝 Message: '{message}' ({len(bits)} bits)")
    
    # Initialize
    embedder = DCTEmbedder(config)
    extractor = DCTExtractor(config)
    print(f"🔧 Embedder alpha: {embedder.alpha}")
    print(f"🔧 Extractor alpha: {extractor.alpha}")
    
    # Embed
    stego, _ = embedder.embed(img, bits)
    
    # Save for inspection
    cv2.imwrite("diagnose_stego.png", cv2.cvtColor(stego, cv2.COLOR_RGB2BGR))
    print(f"💾 Saved stego to diagnose_stego.png")
    
    # Extract
    print(f"\n🔍 Extracting {len(bits)} bits...")
    extracted = extractor.extract(stego, len(bits))
    
    # Compare
    errors = sum(1 for i in range(min(len(bits), len(extracted))) if bits[i] != extracted[i])
    print(f"\n📊 Original:  {bits[:20]}...")
    print(f"📊 Extracted: {extracted[:20]}...")
    print(f"📊 Errors: {errors}/{len(bits)}")
    print(f"📊 BER: {errors/len(bits):.4f}")
    
    # Recover text
    recovered = text_utils.bits_to_text(extracted)
    print(f"\n📝 Recovered: '{recovered}'")
    
    # =========================================================
    # DETAILED BLOCK DIAGNOSTIC
    # =========================================================
    print("\n" + "=" * 60)
    print("🔬 DETAILED BLOCK DIAGNOSTIC")
    print("=" * 60)
    
    # Convert stego and cover to YCbCr float
    ycbcr_stego = cv2.cvtColor(stego, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lum_stego = ycbcr_stego[:, :, 0].copy()
    
    ycbcr_cover = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lum_cover = ycbcr_cover[:, :, 0].copy()
    
    # Check if pixels changed at all
    pixel_diff = np.abs(lum_stego - lum_cover)
    print(f"\n📸 Pixel differences between cover & stego:")
    print(f"   Max diff:  {pixel_diff.max():.4f}")
    print(f"   Mean diff: {pixel_diff.mean():.4f}")
    print(f"   Non-zero pixels: {np.sum(pixel_diff > 0.01)}/{pixel_diff.size}")
    
    if pixel_diff.max() < 0.01:
        print("   🚨 STEGO IMAGE IS IDENTICAL TO COVER! Embedding didn't happen!")
        return
    
    # Check first block
    print(f"\n📦 First block (0,0):")
    print(f"   Cover lum[0:4, 0:4]: {lum_cover[0:4, 0:4].flatten()}")
    print(f"   Stego lum[0:4, 0:4]: {lum_stego[0:4, 0:4].flatten()}")
    
    # DCT of first block
    block_cover = lum_cover[0:8, 0:8]
    block_stego = lum_stego[0:8, 0:8]
    
    dct_cover = dct(dct(block_cover.T, norm='ortho').T, norm='ortho')
    dct_stego = dct(dct(block_stego.T, norm='ortho').T, norm='ortho')
    
    print(f"\n📊 First block DCT coefficients:")
    print(f"   {'Pos':<10} {'Cover':<15} {'Stego':<15} {'Diff':<15}")
    positions = config.MID_FREQ_POSITIONS
    for (u, v) in positions[:5]:
        diff = dct_stego[u, v] - dct_cover[u, v]
        print(f"   [{u},{v}]{'':<5} {dct_cover[u,v]:<15.4f} {dct_stego[u,v]:<15.4f} {diff:<15.4f}")
    
    # What bits should be in first 5 positions?
    print(f"\n🎯 Bit verification for first block:")
    step = 2.0 * 10  # alpha * 10
    
    # Full embedded stream = sync marker + message bits
    full_embedded = extractor.SYNC_MARKER + bits
    print(f"   Expected bits (first 10): {full_embedded[:10]}")
    
    # Manual extraction from first block
    manual_extract = []
    for u, v in positions:
        idx = int(np.round(dct_stego[u, v] / step))
        manual_extract.append(abs(idx) % 2)
    print(f"   Manual extract (first 5): {manual_extract[:5]}")
    
    # What extractor actually returned
    print(f"   Extractor returned (first 5): {extracted[:5]}")
    
    # =========================================================
    # SAVE RELOADED TEST
    # =========================================================
    print("\n" + "=" * 60)
    print("🔄 RELOAD FROM DISK TEST")
    print("=" * 60)
    
    # Reload saved stego from disk
    reloaded = cv2.imread("diagnose_stego.png")
    reloaded = cv2.cvtColor(reloaded, cv2.COLOR_BGR2RGB)
    
    # Extract from reloaded
    extracted2 = extractor.extract(reloaded, len(bits))
    errors2 = sum(1 for i in range(min(len(bits), len(extracted2))) if bits[i] != extracted2[i])
    print(f"\n📊 Reloaded from disk:")
    print(f"   Errors: {errors2}/{len(bits)}")
    print(f"   BER: {errors2/len(bits):.4f}")
    
    # Compare in-memory vs reloaded
    reload_diff = np.abs(stego.astype(int) - reloaded.astype(int))
    print(f"\n   Max pixel diff (in-memory vs reloaded): {reload_diff.max()}")
    print(f"   Mean pixel diff: {reload_diff.mean():.4f}")
    
    if reload_diff.max() == 0:
        print("   ✅ In-memory and reloaded are identical")
    else:
        print(f"   ⚠️ In-memory and reloaded DIFFER by up to {reload_diff.max()}")
    
    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 60)
    if errors == 0:
        print("✅ Round-trip works on THIS image!")
    else:
        print(f"❌ Round-trip FAILS: {errors} errors")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()