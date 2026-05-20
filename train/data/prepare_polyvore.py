"""
Prepare training data from the Polyvore Outfits dataset.

Downloads and processes the Maryland Polyvore dataset into pair CSVs
suitable for contrastive training of multi-head projection layers.

Produces:
    data/polyvore/compat_pairs.csv  -- (item_a_id, item_b_id, label) for compat_head + scorer
    data/polyvore/style_pairs.csv   -- (item_a_id, item_b_id, label) for style_head
    data/polyvore/item_metadata.csv -- (item_id, category, image_path) index

Data source:
    Maryland Polyvore Outfits (https://github.com/iqon/polyvore-dataset)
    Alternate: https://github.com/mvasil/fashion-compatibility (same data, different format)

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
    
    The dataset is hosted on multiple mirrors. This function attempts
    to clone the GitHub repo with outfit JSON metadata + download images.
    """
    import subprocess

    repo_url = "https://github.com/mvasil/fashion-compatibility.git"
    repo_dir = data_dir / "fashion-compatibility"

    if repo_dir.exists():
        logger.info("Repo already exists at %s, skipping clone", repo_dir)
    else:
        logger.info("Cloning Polyvore metadata from %s ...", repo_url)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)

    # The metadata JSONs are in the repo. Images need separate download.
    images_dir = data_dir / "images"
    if images_dir.exists() and len(list(images_dir.iterdir())) > 1000:
        logger.info("Images directory already populated (%s)", images_dir)
    else:
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "\n"
            "=" * 70 + "\n"
            "MANUAL STEP REQUIRED:\n"
            "Download Polyvore images from one of these sources:\n"
            "  1. https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5\n"
            "  2. Request from the dataset authors (UMD)\n"
            "\n"
            "Extract images to: %s\n"
            "Expected structure: %s/<item_id>/<item_id>.jpg\n"
            "=" * 70,
            images_dir, images_dir
        )

    return repo_dir


def load_outfit_data(repo_dir: Path) -> dict:
    """Load outfit definitions from the Polyvore metadata JSONs."""
    outfits = {}
    
    for split in ["train", "valid", "test"]:
        json_path = repo_dir / "data" / "polyvore_outfits" / f"{split}.json"
        if not json_path.exists():
            # Try alternate path structure
            json_path = repo_dir / "data" / f"{split}.json"
        
        if not json_path.exists():
            logger.warning("Split file not found: %s", json_path)
            continue

        with open(json_path) as f:
            data = json.load(f)

        for outfit in data:
            outfit_id = outfit.get("set_id", outfit.get("id"))
            items = []
            for item in outfit.get("items", []):
                items.append({
                    "item_id": item.get("item_id", item.get("index")),
                    "category": item.get("categoryid", item.get("category", "")),
                })
            if len(items) >= 2:
                outfits[outfit_id] = items

    logger.info("Loaded %d outfits across all splits", len(outfits))
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

        # All pairs within an outfit are positive
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                positive_pairs.append((item_ids[i], item_ids[j], 1))

    all_item_ids = list(all_item_ids)
    logger.info("Generated %d positive compat pairs", len(positive_pairs))

    # Build item-to-outfit index for hard negative mining
    item_to_outfits = {}
    for outfit_id, items in outfits.items():
        for item in items:
            item_to_outfits.setdefault(item["item_id"], set()).add(outfit_id)

    # Generate negatives: random items NOT in the same outfit
    negative_pairs = []
    random.seed(42)
    for item_a, item_b, _ in positive_pairs:
        for _ in range(neg_ratio):
            # Pick random item not in any outfit containing item_a
            a_outfits = item_to_outfits.get(item_a, set())
            a_coitems = set()
            for oid in a_outfits:
                a_coitems.update(item["item_id"] for item in outfits[oid])

            neg_item = random.choice(all_item_ids)
            attempts = 0
            while neg_item in a_coitems and attempts < 10:
                neg_item = random.choice(all_item_ids)
                attempts += 1

            negative_pairs.append((item_a, neg_item, 0))

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
    # Group items by their outfits
    positive_pairs = []
    all_item_ids = set()

    for outfit_id, items in outfits.items():
        item_ids = [item["item_id"] for item in items]
        all_item_ids.update(item_ids)
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                positive_pairs.append((item_ids[i], item_ids[j], 1))

    all_item_ids = list(all_item_ids)

    # Negatives: items from stylistically different outfits
    negative_pairs = []
    random.seed(123)
    for item_a, _, _ in positive_pairs[:len(positive_pairs) // neg_ratio]:
        neg_item = random.choice(all_item_ids)
        negative_pairs.append((item_a, neg_item, 0))

    logger.info("Style pairs: %d positive, %d negative", len(positive_pairs), len(negative_pairs))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


def build_item_index(outfits: dict, images_dir: Path) -> list[dict]:
    """Build item metadata index mapping item_id to category and image path."""
    items = {}
    for outfit_id, outfit_items in outfits.items():
        for item in outfit_items:
            item_id = item["item_id"]
            if item_id not in items:
                image_path = images_dir / str(item_id) / f"{item_id}.jpg"
                items[item_id] = {
                    "item_id": item_id,
                    "category": item.get("category", ""),
                    "image_path": str(image_path),
                }
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
    parser.add_argument("--download", action="store_true", help="Download dataset repo")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negative:positive ratio for compat")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        repo_dir = download_polyvore(data_dir)
    else:
        repo_dir = data_dir / "fashion-compatibility"
        if not repo_dir.exists():
            logger.error(
                "Repo not found at %s. Run with --download or clone manually.", repo_dir
            )
            sys.exit(1)

    outfits = load_outfit_data(repo_dir)
    if not outfits:
        logger.error("No outfits loaded. Check data paths.")
        sys.exit(1)

    # Build pairs
    compat_pairs = build_compat_pairs(outfits, neg_ratio=args.neg_ratio)
    style_pairs = build_style_pairs(outfits, neg_ratio=2)

    # Build item index
    images_dir = data_dir / "images"
    item_index = build_item_index(outfits, images_dir)

    # Save outputs
    save_pairs_csv(compat_pairs, data_dir / "compat_pairs.csv")
    save_pairs_csv(style_pairs, data_dir / "style_pairs.csv")
    save_item_index(item_index, data_dir / "item_metadata.csv")

    logger.info(
        "\nDone! Summary:\n"
        "  Outfits: %d\n"
        "  Unique items: %d\n"
        "  Compat pairs: %d (%.0f%% positive)\n"
        "  Style pairs: %d (%.0f%% positive)\n",
        len(outfits),
        len(item_index),
        len(compat_pairs),
        100 * sum(1 for _, _, l in compat_pairs if l == 1) / len(compat_pairs),
        len(style_pairs),
        100 * sum(1 for _, _, l in style_pairs if l == 1) / len(style_pairs),
    )


if __name__ == "__main__":
    main()
