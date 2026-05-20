"""
Prepare training data from the Polyvore Outfits dataset.

Downloads and processes the Maryland Polyvore dataset into pair CSVs
suitable for contrastive training of multi-head projection layers.

Produces:
    data/polyvore/compat_pairs.csv  -- (item_a_id, item_b_id, label) for compat_head + scorer
    data/polyvore/style_pairs.csv   -- (item_a_id, item_b_id, label) for style_head
    data/polyvore/item_metadata.csv -- (item_id, category, image_path) index

Data source:
    HuggingFace: mvasil/polyvore-outfits (6GB, contains images + outfit JSONs)
    Paper: "Learning Type-Aware Embeddings for Fashion Compatibility" (ECCV 2018)

Usage:
    python train/data/prepare_polyvore.py --data-dir data/polyvore [--download]
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_polyvore(data_dir: Path):
    """
    Download Polyvore Outfits dataset.
    
    Uses xthan/polyvore-dataset (public GitHub, no auth required).
    Contains 21,889 outfits with item metadata in JSON format.
    The JSONs are packed in polyvore.tar.gz inside the repo.
    """
    import subprocess
    import tarfile

    repo_dir = data_dir / "polyvore-dataset"

    # Check if already extracted
    if any(data_dir.rglob("*_no_dup.json")):
        logger.info(f"Dataset JSONs already exist under {data_dir}, skipping download")
        return

    # Clone repo
    if not repo_dir.exists():
        logger.info("Cloning xthan/polyvore-dataset (public, no auth required)...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/xthan/polyvore-dataset.git", str(repo_dir)],
            check=True,
        )

    # Extract polyvore.tar.gz
    tarball = repo_dir / "polyvore.tar.gz"
    if tarball.exists():
        logger.info(f"Extracting {tarball}...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=str(data_dir))
        logger.info(f"Extracted to {data_dir}")
    else:
        logger.error(f"Expected tarball not found at {tarball}")
        sys.exit(1)

    # List what we got
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files after extraction")


def load_outfit_data(data_dir: Path) -> dict:
    """
    Load outfit definitions from the Polyvore dataset.
    
    Searches common path structures:
    - HuggingFace format: polyvore-outfits/nondisjoint/*.json
    - GitHub format: fashion-compatibility/data/polyvore_outfits/*.json
    - Direct: data_dir/*.json
    """
    outfits = {}

    # Search for JSON files in various possible locations
    search_paths = [
        data_dir / "polyvore-dataset",
        data_dir / "polyvore-outfits" / "nondisjoint",
        data_dir / "polyvore-outfits" / "disjoint",
        data_dir / "polyvore-outfits",
        data_dir / "fashion-compatibility" / "data" / "polyvore_outfits",
        data_dir / "fashion-compatibility" / "data",
        data_dir,
    ]

    split_names = [
        "train_no_dup", "valid_no_dup", "test_no_dup",
        "train", "valid", "test",
    ]

    json_files_found = []
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for split in split_names:
            json_path = search_path / f"{split}.json"
            if json_path.exists():
                json_files_found.append(json_path)

    if not json_files_found:
        # Try recursive glob as last resort
        json_files_found = list(data_dir.rglob("train*.json")) + list(data_dir.rglob("valid*.json"))

    if not json_files_found:
        logger.error(f"No outfit JSON files found under {data_dir}")
        logger.error("Expected files like train.json, valid.json, test.json")
        logger.error("Run with --download to fetch from HuggingFace")
        sys.exit(1)

    logger.info(f"Found {len(json_files_found)} JSON files: {[p.name for p in json_files_found]}")

    for json_path in json_files_found:
        with open(json_path) as f:
            data = json.load(f)

        # Handle both list format and dict format
        if isinstance(data, dict):
            outfit_list = list(data.values()) if not isinstance(list(data.values())[0], list) else data.get("outfits", [])
        else:
            outfit_list = data

        for outfit in outfit_list:
            if isinstance(outfit, dict):
                outfit_id = str(outfit.get("set_id", outfit.get("id", "")))
                items = []
                for item in outfit.get("items", []):
                    # xthan format: unique item ID = "{set_id}_{index}"
                    # mvasil format: has "item_id" directly
                    if "item_id" in item:
                        item_id = str(item["item_id"])
                    elif "index" in item and outfit_id:
                        item_id = f"{outfit_id}_{item['index']}"
                    else:
                        continue
                    items.append({
                        "item_id": item_id,
                        "category": str(item.get("categoryid", item.get("category", ""))),
                    })
                if len(items) >= 2:
                    outfits[outfit_id] = items

    logger.info(f"Loaded {len(outfits)} outfits across all splits")
    return outfits


def build_compat_pairs(outfits: dict, neg_ratio: int = 3) -> list[tuple]:
    """
    Build compatibility pairs from outfit co-occurrence.
    
    Positive: (item_a, item_b) from the SAME outfit
    Negative: (item_a, random_item) from DIFFERENT outfits
    
    neg_ratio: number of negatives per positive (3:1 gives good training signal)
    """
    all_item_ids = set()
    positive_pairs = []

    for outfit_id, items in outfits.items():
        item_ids = [item["item_id"] for item in items]
        all_item_ids.update(item_ids)

        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                positive_pairs.append((item_ids[i], item_ids[j], 1))

    all_item_ids_list = list(all_item_ids)
    logger.info("Generated %d positive compat pairs", len(positive_pairs))

    # Pre-compute co-item sets for each item (do this ONCE, not per-pair)
    item_coitems = {}
    for outfit_id, items in outfits.items():
        outfit_item_ids = {item["item_id"] for item in items}
        for item_id in outfit_item_ids:
            if item_id not in item_coitems:
                item_coitems[item_id] = set()
            item_coitems[item_id].update(outfit_item_ids)

    # Generate all negatives in batch (vectorized random sampling)
    n_negatives = len(positive_pairs) * neg_ratio
    random.seed(42)
    logger.info("Generating %d negative pairs...", n_negatives)

    negative_pairs = []
    # Sample anchors from positive pairs (cycle through them)
    anchor_ids = [p[0] for p in positive_pairs] * neg_ratio
    random.shuffle(anchor_ids)
    anchor_ids = anchor_ids[:n_negatives]

    # Batch random sampling
    random_items = random.choices(all_item_ids_list, k=n_negatives)

    for anchor, neg in zip(anchor_ids, random_items):
        # Simple check -- if collision, just use it anyway (noise is fine for training)
        if neg not in item_coitems.get(anchor, set()):
            negative_pairs.append((anchor, neg, 0))
        else:
            # Pick one more random -- don't loop, just take it
            alt = random.choice(all_item_ids_list)
            negative_pairs.append((anchor, alt, 0))

    logger.info("Generated %d negative compat pairs (ratio %d:1)", len(negative_pairs), neg_ratio)

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


def build_style_pairs(outfits: dict, neg_ratio: int = 2) -> list[tuple]:
    """
    Build style pairs. Items in the same outfit share an aesthetic.
    
    This is weaker than dedicated style boards but still useful --
    items styled together typically share a cohesive aesthetic.
    """
    positive_pairs = []
    all_item_ids = set()

    for outfit_id, items in outfits.items():
        item_ids = [item["item_id"] for item in items]
        all_item_ids.update(item_ids)
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                positive_pairs.append((item_ids[i], item_ids[j], 1))

    all_item_ids_list = list(all_item_ids)

    # Negatives: batch random sampling (fast)
    n_neg = len(positive_pairs) // neg_ratio
    random.seed(123)
    anchors = [p[0] for p in positive_pairs[:n_neg]]
    neg_items = random.choices(all_item_ids_list, k=n_neg)
    negative_pairs = [(a, n, 0) for a, n in zip(anchors, neg_items)]

    logger.info("Style pairs: %d positive, %d negative", len(positive_pairs), len(negative_pairs))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


def build_item_index(outfits: dict, data_dir: Path) -> list[dict]:
    """Build item metadata index mapping item_id to category and image path."""
    # Find image directory -- try multiple possible structures
    possible_image_dirs = [
        data_dir / "polyvore-outfits" / "images",
        data_dir / "polyvore-outfits" / "nondisjoint" / "images",
        data_dir / "images",
    ]
    images_dir = None
    for d in possible_image_dirs:
        if d.exists():
            images_dir = d
            break

    if images_dir is None:
        images_dir = data_dir / "images"
        logger.warning(f"No images directory found, using {images_dir} (images may need download)")

    items = {}
    found = 0
    for outfit_id, outfit_items in outfits.items():
        for item in outfit_items:
            item_id = item["item_id"]
            if item_id not in items:
                # Try common image path patterns
                candidates = [
                    images_dir / str(item_id) / f"{item_id}.jpg",
                    images_dir / f"{item_id}.jpg",
                    images_dir / str(item_id) / "1.jpg",
                ]
                image_path = candidates[0]  # default
                for c in candidates:
                    if c.exists():
                        image_path = c
                        found += 1
                        break

                items[item_id] = {
                    "item_id": item_id,
                    "category": item.get("category", ""),
                    "image_path": str(image_path),
                }

    logger.info(f"Item index: {len(items)} unique items, {found} with existing images")
    return list(items.values())


def save_pairs_csv(pairs: list[tuple], output_path: Path):
    """Save pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_a", "item_b", "label"])
        writer.writerows(pairs)
    logger.info("Saved %d pairs to %s", len(pairs), output_path)


def save_item_index(items: list[dict], output_path: Path):
    """Save item metadata index."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "category", "image_path"])
        writer.writeheader()
        writer.writerows(items)
    logger.info("Saved %d item records to %s", len(items), output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare Polyvore training data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/polyvore"))
    parser.add_argument("--download", action="store_true", help="Download dataset from GitHub")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negative:positive ratio for compat")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_polyvore(data_dir)

    outfits = load_outfit_data(data_dir)
    if not outfits:
        logger.error("No outfits loaded. Run with --download to fetch from GitHub.")
        sys.exit(1)

    # Build pairs
    compat_pairs = build_compat_pairs(outfits, neg_ratio=args.neg_ratio)
    style_pairs = build_style_pairs(outfits, neg_ratio=2)

    # Build item index
    item_index = build_item_index(outfits, data_dir)

    # Save outputs
    save_pairs_csv(compat_pairs, data_dir / "compat_pairs.csv")
    save_pairs_csv(style_pairs, data_dir / "style_pairs.csv")
    save_item_index(item_index, data_dir / "item_metadata.csv")

    n_pos_compat = sum(1 for _, _, l in compat_pairs if l == 1)
    n_pos_style = sum(1 for _, _, l in style_pairs if l == 1)

    logger.info(
        f"\nDone! Summary:\n"
        f"  Outfits: {len(outfits)}\n"
        f"  Unique items: {len(item_index)}\n"
        f"  Compat pairs: {len(compat_pairs)} ({100 * n_pos_compat / len(compat_pairs):.0f}% positive)\n"
        f"  Style pairs: {len(style_pairs)} ({100 * n_pos_style / len(style_pairs):.0f}% positive)"
    )


if __name__ == "__main__":
    main()
