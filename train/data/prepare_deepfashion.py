"""
Prepare training data from DeepFashion / iMaterialist for attribute heads.

Builds pair CSVs for occasion, fit, and material heads using attribute annotations.
Items sharing the same attribute label form positive pairs; items with different
labels form negative pairs.

Data sources:
    - DeepFashion Category & Attribute Prediction (52K images, 1000+ attributes)
      https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
    - iMaterialist Fashion (Kaggle, 1M images, 228 attributes)
      https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6

Usage:
    python train/data/prepare_deepfashion.py --data-dir data/deepfashion [--download]
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Attribute groupings: which DeepFashion attributes map to which head
FIT_ATTRIBUTES = {
    "slim", "skinny", "tight", "fitted", "bodycon", "tailored",
    "loose", "oversized", "relaxed", "baggy", "wide",
    "regular", "straight", "classic",
}

MATERIAL_ATTRIBUTES = {
    "cotton", "silk", "satin", "chiffon", "linen", "wool", "cashmere",
    "denim", "leather", "suede", "velvet", "lace", "mesh", "knit",
    "polyester", "nylon", "tweed", "corduroy", "fleece", "jersey",
}

OCCASION_ATTRIBUTES = {
    "casual", "formal", "business", "sport", "athletic", "party",
    "beach", "evening", "wedding", "work", "office", "streetwear",
    "lounge", "outdoor", "gym", "date",
}


def load_deepfashion_attributes(data_dir: Path) -> dict[str, dict]:
    """
    Load DeepFashion attribute annotations.
    
    Expected file format (list_attr_items.txt or CSV):
        item_id, attribute_1, attribute_2, ...
    
    Returns dict of {item_id: {"fit": [...], "material": [...], "occasion": [...], "image_path": str}}
    """
    items = {}

    # Try multiple possible file formats
    attr_file = data_dir / "list_attr_items.txt"
    csv_file = data_dir / "attributes.csv"

    if csv_file.exists():
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get("image_id", row.get("id", ""))
                attrs = set()
                for key, val in row.items():
                    if key in ("image_id", "id", "image_path"):
                        continue
                    if val and val.strip() in ("1", "true", "yes"):
                        attrs.add(key.lower().strip())

                items[item_id] = {
                    "fit": [a for a in attrs if a in FIT_ATTRIBUTES],
                    "material": [a for a in attrs if a in MATERIAL_ATTRIBUTES],
                    "occasion": [a for a in attrs if a in OCCASION_ATTRIBUTES],
                    "image_path": row.get("image_path", f"images/{item_id}.jpg"),
                }

    elif attr_file.exists():
        with open(attr_file) as f:
            lines = f.readlines()

        # DeepFashion format: first line = count, second = header, rest = data
        if len(lines) > 2:
            header = lines[1].strip().split()
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                item_id = parts[0]
                attr_values = parts[1:]

                active_attrs = set()
                for i, val in enumerate(attr_values):
                    if val == "1" and i < len(header):
                        active_attrs.add(header[i].lower().strip())

                items[item_id] = {
                    "fit": [a for a in active_attrs if a in FIT_ATTRIBUTES],
                    "material": [a for a in active_attrs if a in MATERIAL_ATTRIBUTES],
                    "occasion": [a for a in active_attrs if a in OCCASION_ATTRIBUTES],
                    "image_path": str(data_dir / "images" / item_id),
                }
    else:
        logger.error(
            "No attribute file found. Expected one of:\n"
            "  %s\n  %s\n\n"
            "Download DeepFashion from: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html\n"
            "Or iMaterialist from: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6",
            attr_file, csv_file
        )
        sys.exit(1)

    logger.info("Loaded %d items with attributes", len(items))
    return items


def build_attribute_pairs(
    items: dict[str, dict],
    attribute_key: str,
    neg_ratio: int = 2,
    max_pairs: int = 200000,
) -> list[tuple]:
    """
    Build positive/negative pairs based on shared attribute labels.
    
    Positive: two items sharing at least one label in the attribute group
    Negative: two items with NO shared labels in the attribute group
    """
    # Group items by their attribute labels
    label_to_items = {}
    for item_id, attrs in items.items():
        for label in attrs.get(attribute_key, []):
            label_to_items.setdefault(label, []).append(item_id)

    # Build positives: items sharing a label
    positive_pairs = []
    for label, label_items in label_to_items.items():
        if len(label_items) < 2:
            continue
        # Sample pairs within this group (avoid O(n^2) for large groups)
        n = len(label_items)
        if n * (n - 1) // 2 <= max_pairs // len(label_to_items):
            for i in range(n):
                for j in range(i + 1, n):
                    positive_pairs.append((label_items[i], label_items[j], 1))
        else:
            n_sample = max_pairs // len(label_to_items)
            for _ in range(n_sample):
                i, j = random.sample(range(n), 2)
                positive_pairs.append((label_items[i], label_items[j], 1))

    if len(positive_pairs) > max_pairs:
        random.shuffle(positive_pairs)
        positive_pairs = positive_pairs[:max_pairs]

    logger.info("%s: %d positive pairs", attribute_key, len(positive_pairs))

    # Build negatives: items with different labels
    all_item_ids = list(items.keys())
    negative_pairs = []
    random.seed(42)

    for item_a, _, _ in positive_pairs[:len(positive_pairs) // neg_ratio * neg_ratio]:
        a_labels = set(items[item_a].get(attribute_key, []))
        if not a_labels:
            continue

        # Find a negative: item with no shared labels
        for _ in range(10):
            item_b = random.choice(all_item_ids)
            b_labels = set(items[item_b].get(attribute_key, []))
            if not a_labels.intersection(b_labels) and b_labels:
                negative_pairs.append((item_a, item_b, 0))
                break

    logger.info("%s: %d negative pairs", attribute_key, len(negative_pairs))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


def save_pairs_csv(pairs: list[tuple], output_path: Path):
    """Save pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_a", "item_b", "label"])
        writer.writerows(pairs)
    logger.info("Saved %d pairs to %s", len(pairs), output_path)


def save_item_index(items: dict[str, dict], output_path: Path):
    """Save item metadata for backbone precomputation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "category", "image_path"])
        for item_id, attrs in items.items():
            writer.writerow([item_id, "", attrs.get("image_path", "")])
    logger.info("Saved %d items to %s", len(items), output_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepFashion attribute pairs")
    parser.add_argument("--data-dir", type=Path, default=Path("data/deepfashion"))
    parser.add_argument("--neg-ratio", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=200000)
    args = parser.parse_args()

    data_dir = args.data_dir

    items = load_deepfashion_attributes(data_dir)

    # Build pairs for each attribute head
    for attr_key in ["occasion", "fit", "material"]:
        pairs = build_attribute_pairs(items, attr_key, args.neg_ratio, args.max_pairs)
        if pairs:
            save_pairs_csv(pairs, data_dir / f"{attr_key}_pairs.csv")
        else:
            logger.warning("No pairs generated for %s (not enough labeled items?)", attr_key)

    # Save item index for backbone precomputation
    save_item_index(items, data_dir / "item_metadata.csv")

    logger.info("\nDone! Files saved to %s", data_dir)


if __name__ == "__main__":
    main()
