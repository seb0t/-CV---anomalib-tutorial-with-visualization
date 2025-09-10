#!/usr/bin/env python3
"""
Enhanced Embeddings Bank Creator

Creates and saves embeddings bank from all good train images.
Features:
- Configurable patch size (default: 5)
- Progress bar with ETA
- Automatic saving and resuming
- Memory efficient processing

Usage:
    python create_bank.py [--patch-size 5] [--no-save]

Arguments:
    --patch-size: Size of patches to extract (default: 5)
    --no-save: Don't save the bank to disk (for testing)
"""

import sys
import os
import argparse
from tqdm import tqdm

# Add the enhanced_dashboard directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from functions.helpers import create_embeddings_bank, load_embeddings_bank
from app import load_mvtec_category


def progress_callback(current, total):
    """Progress callback for the embeddings creation."""
    pass  # We'll use tqdm instead


def main():
    parser = argparse.ArgumentParser(description="Create embeddings bank from train images")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=5,
        help="Size of patches to extract (default: 5)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="wood",
        help="MVTec category to process (default: wood)"
    )

    args = parser.parse_args()

    print("🚀 Enhanced Embeddings Bank Creator")
    print("=" * 50)
    print(f"📊 Category: {args.category}")
    print(f"🔲 Patch size: {args.patch_size}")
    print(f"💾 Save to disk: Yes")
    print()

    # Load dataset
    print("📂 Loading dataset...")
    dataset = load_mvtec_category(args.category)

    if dataset is None:
        print("❌ Error: Could not load dataset. Make sure mvtec_data is available.")
        return 1

    train_images = dataset.get("train_images", [])
    if not train_images:
        print("❌ Error: No train images found in dataset!")
        return 1

    print(f"✅ Found {len(train_images)} train images")
    print()

    # Check for existing bank
    print("🔍 Checking for existing embeddings_bank.pkl...")
    existing_bank = load_embeddings_bank()

    if existing_bank is not None:
        print(f"✅ Embeddings bank already exists! Shape: {existing_bank.shape}")
        print("💡 Overwriting existing bank...")

    # Create progress-aware embeddings bank
    print("🧠 Creating embeddings bank...")
    print(f"⚙️  Processing {len(train_images)} images with patch_size={args.patch_size}")
    print("📈 This may take several minutes depending on your hardware...")

    # Create a custom progress bar
    with tqdm(total=len(train_images), desc="Processing images", unit="img",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        # We'll track progress manually since the function doesn't support direct tqdm
        progress_data = {'current': 0, 'total': len(train_images)}

        def update_progress(current, total):
            progress_data['current'] = current
            pbar.n = current
            pbar.refresh()

        try:
            # Create embeddings bank with progress callback
            embeddings_bank = create_embeddings_bank(
                train_images,
                patch_size=args.patch_size,
                save_bank=True,
                progress_callback=update_progress
            )

            pbar.n = len(train_images)
            pbar.refresh()

        except KeyboardInterrupt:
            print("\n⏹️  Process interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            return 1

    print()
    print("✅ Embeddings bank created successfully!")
    print(f"📊 Final shape: {embeddings_bank.shape}")
    print(f"🗂️  Total embeddings: {len(embeddings_bank)}")
    print(f"🎯 Embedding dimension: {embeddings_bank.shape[1] if len(embeddings_bank.shape) > 1 else 'N/A'}")

    # Verify save
    print("\n🔍 Verifying save...")
    saved_bank = load_embeddings_bank()
    if saved_bank is not None:
        print(f"✅ Verification successful! Saved bank shape: {saved_bank.shape}")
        print("💾 File saved as: embeddings_bank.pkl")
    else:
        print("❌ Error: Bank was not saved correctly!")
        return 1

    print("\n🎉 All done! You can now use the embeddings bank for anomaly detection.")
    return 0


if __name__ == "__main__":
    exit(main())
