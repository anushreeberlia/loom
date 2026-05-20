"""
Pre-compute DINOv2 backbone embeddings for all training images.

Since the DINOv2 backbone is frozen during training, we only need to run it once.
The resulting 768-dim embeddings are saved to an HDF5 file for fast loading during
head training (training then operates purely on small matrices -- no GPU needed).

Usage:
    python train/precompute_backbones.py --data-dir data/polyvore --output data/polyvore/backbone_embeddings.h5

Output:
    HDF5 file with datasets:
        "embeddings" -- shape (N, 768), float32
        "item_ids"   -- shape (N,), variable-length string
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_item_index(data_dir: Path) -> list[dict]:
    """Load item metadata CSV."""
    index_path = data_dir / "item_metadata.csv"
    items = []
    with open(index_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    logger.info("Loaded %d items from index", len(items))
    return items


def precompute_embeddings(
    items: list[dict],
    output_path: Path,
    batch_size: int = 32,
    segment: bool = True,
):
    """
    Run DINOv2 on all item images and save embeddings to HDF5.
    
    Optionally runs segmentation first (recommended for clean embeddings,
    but slower -- ~2x processing time).
    """
    import h5py

    from services.dinov2 import embed_images, EMBEDDING_DIM

    if segment:
        from services.segmentation import segment_for_embedding

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to items with existing images
    valid_items = []
    for item in items:
        if Path(item["image_path"]).exists():
            valid_items.append(item)

    if not valid_items:
        logger.error("No valid images found. Check image paths in item_metadata.csv")
        logger.info("First few paths checked: %s", [items[i]["image_path"] for i in range(min(3, len(items)))])
        sys.exit(1)

    logger.info("Processing %d items (of %d total, %d missing images)",
                len(valid_items), len(items), len(items) - len(valid_items))

    # Process in batches
    all_embeddings = []
    all_item_ids = []
    failed = 0

    for batch_start in range(0, len(valid_items), batch_size):
        batch_items = valid_items[batch_start:batch_start + batch_size]
        batch_bytes = []
        batch_ids = []

        for item in batch_items:
            try:
                with open(item["image_path"], "rb") as f:
                    raw_bytes = f.read()

                if segment:
                    img_bytes = segment_for_embedding(raw_bytes)
                else:
                    img_bytes = raw_bytes

                batch_bytes.append(img_bytes)
                batch_ids.append(item["item_id"])
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.warning("Failed to process %s: %s", item["item_id"], e)

        if batch_bytes:
            embeddings = embed_images(batch_bytes, batch_size=len(batch_bytes))
            all_embeddings.append(embeddings)
            all_item_ids.extend(batch_ids)

        processed = batch_start + len(batch_items)
        if processed % (batch_size * 10) == 0 or processed == len(valid_items):
            logger.info("Progress: %d/%d (%.1f%%)", processed, len(valid_items),
                       100 * processed / len(valid_items))

    # Concatenate and save
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("Final embeddings shape: %s", all_embeddings.shape)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("embeddings", data=all_embeddings, dtype=np.float32)
        dt = h5py.string_dtype()
        ds = f.create_dataset("item_ids", shape=(len(all_item_ids),), dtype=dt)
        for i, item_id in enumerate(all_item_ids):
            ds[i] = item_id

    logger.info("Saved %d embeddings to %s (%.1f MB)",
                len(all_item_ids), output_path,
                os.path.getsize(output_path) / 1024 / 1024)
    logger.info("Failed items: %d", failed)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 backbone embeddings")
    parser.add_argument("--data-dir", type=Path, default=Path("data/polyvore"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-segment", action="store_true",
                       help="Skip segmentation (faster but noisier embeddings)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.data_dir / "backbone_embeddings.h5"

    items = load_item_index(args.data_dir)

    t0 = time.time()
    precompute_embeddings(items, args.output, args.batch_size, segment=not args.no_segment)
    elapsed = time.time() - t0
    logger.info("Total time: %.1f minutes (%.2f items/sec)",
                elapsed / 60, len(items) / elapsed)


if __name__ == "__main__":
    main()
