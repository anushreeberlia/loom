"""
Merge all training data sources into unified pair CSVs.

Combines:
  - Polyvore (academic baseline, co-occurrence)
  - DeepFashion (attribute-level pairs)
  - Pinterest (real-world aesthetic outfits)
  - FashionStylist (expert-curated, attribute-rich)
  - FashionRec (engagement-weighted interactions)

Produces:
    data/merged/item_metadata.csv       -- deduplicated union of all items
    data/merged/compat_pairs.csv        -- weighted merge of compat pairs
    data/merged/style_pairs.csv         -- weighted merge of style pairs
    data/merged/occasion_pairs.csv      -- merged occasion pairs
    data/merged/fit_pairs.csv           -- merged fit pairs
    data/merged/material_pairs.csv      -- merged material pairs
    data/merged/backbone_embeddings.h5  -- (created by precompute step, not this script)

Source weights control how much each dataset contributes.
Pinterest/FashionStylist are upweighted because they represent the
aesthetic signal we want (stylish-together vs just similar).

Usage:
    python train/data/merge_datasets.py --output-dir data/merged
    python train/data/merge_datasets.py --output-dir data/merged --weights pinterest=3.0 fashionstylist=2.5
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

DEFAULT_SOURCE_DIRS = {
    "polyvore": Path("data/polyvore"),
    "deepfashion": Path("data/deepfashion"),
    "pinterest": Path("data/pinterest"),
    "fashionstylist": Path("data/fashionstylist"),
    "fashionrec": Path("data/fashionrec"),
}

DEFAULT_WEIGHTS = {
    "polyvore": 1.0,
    "deepfashion": 1.0,
    "pinterest": 3.0,
    "fashionstylist": 2.5,
    "fashionrec": 1.5,
}

PAIR_TYPES = ["compat_pairs", "style_pairs", "occasion_pairs", "fit_pairs", "material_pairs"]


def load_pairs_csv(path: Path) -> list[tuple]:
    """Load (item_a, item_b, label) triples from a CSV."""
    if not path.exists():
        return []
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["item_a"], row["item_b"], int(row["label"])))
    return pairs


def load_item_metadata(path: Path) -> dict:
    """Load item metadata CSV into {item_id: row_dict}."""
    if not path.exists():
        return {}
    items = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items[row["item_id"]] = row
    return items


def merge_pairs(
    source_dirs: dict[str, Path],
    pair_type: str,
    weights: dict[str, float],
) -> list[tuple]:
    """
    Merge pairs from all sources with oversampling by weight.

    Weight > 1.0 means the source's pairs are replicated (oversampled).
    Weight < 1.0 means the source's pairs are downsampled.
    """
    merged = []

    for source_name, source_dir in source_dirs.items():
        csv_path = source_dir / f"{pair_type}.csv"
        pairs = load_pairs_csv(csv_path)
        if not pairs:
            continue

        source_prefix = source_name[:3]
        prefixed_pairs = [
            (f"{source_prefix}_{a}", f"{source_prefix}_{b}", label)
            for a, b, label in pairs
        ]

        w = weights.get(source_name, 1.0)

        if w >= 1.0:
            full_copies = int(w)
            fractional = w - full_copies
            for _ in range(full_copies):
                merged.extend(prefixed_pairs)
            if fractional > 0:
                n_extra = int(len(prefixed_pairs) * fractional)
                merged.extend(random.sample(prefixed_pairs, min(n_extra, len(prefixed_pairs))))
        else:
            n_keep = int(len(prefixed_pairs) * w)
            merged.extend(random.sample(prefixed_pairs, min(n_keep, len(prefixed_pairs))))

        logger.info("  %s/%s: %d raw pairs * %.1fx weight = %d effective",
                    source_name, pair_type, len(pairs), w,
                    int(len(pairs) * w))

    random.shuffle(merged)
    return merged


def merge_metadata(source_dirs: dict[str, Path]) -> dict:
    """Merge item metadata from all sources with source prefix to avoid collisions."""
    merged = {}
    for source_name, source_dir in source_dirs.items():
        items = load_item_metadata(source_dir / "item_metadata.csv")
        source_prefix = source_name[:3]
        for item_id, meta in items.items():
            new_id = f"{source_prefix}_{item_id}"
            new_meta = dict(meta)
            new_meta["item_id"] = new_id
            new_meta["source"] = source_name
            img = meta.get("image_path", "")
            if img and not img.startswith("/"):
                img_path = Path(img)
                source_str = str(source_dir)
                if img.startswith(source_str + "/") or img.startswith(source_str + os.sep):
                    new_meta["image_path"] = img
                else:
                    new_meta["image_path"] = str(source_dir / img)
            merged[new_id] = new_meta
        if items:
            logger.info("  %s: %d items", source_name, len(items))
    return merged


def save_pairs_csv(pairs: list[tuple], path: Path):
    """Save pairs CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_a", "item_b", "label"])
        writer.writeheader()
        for a, b, label in pairs:
            writer.writerow({"item_a": a, "item_b": b, "label": label})


def save_metadata_csv(items: dict, path: Path):
    """Save merged item metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_id", "image_path", "category", "source"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item_id, meta in items.items():
            writer.writerow(meta)


def parse_weights_arg(weights_str: list[str] | None) -> dict[str, float]:
    """Parse --weights key=value pairs."""
    w = dict(DEFAULT_WEIGHTS)
    if not weights_str:
        return w
    for kv in weights_str:
        if "=" in kv:
            k, v = kv.split("=", 1)
            w[k.strip()] = float(v.strip())
    return w


def main():
    parser = argparse.ArgumentParser(description="Merge all training data sources")
    parser.add_argument("--output-dir", type=str, default="data/merged")
    parser.add_argument("--weights", nargs="*", default=None,
                        help="Source weights as key=value (e.g. pinterest=3.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = parse_weights_arg(args.weights)

    available_dirs = {}
    for name, d in DEFAULT_SOURCE_DIRS.items():
        if d.exists() and any(d.glob("*.csv")):
            available_dirs[name] = d
        else:
            logger.warning("Source '%s' not found at %s, skipping", name, d)

    if not available_dirs:
        logger.error("No data sources found. Run prepare_*.py scripts first.")
        sys.exit(1)

    logger.info("Merging %d sources: %s", len(available_dirs), list(available_dirs.keys()))
    logger.info("Weights: %s", {k: weights.get(k, 1.0) for k in available_dirs})

    logger.info("\n=== Merging item metadata ===")
    merged_items = merge_metadata(available_dirs)
    save_metadata_csv(merged_items, output_dir / "item_metadata.csv")
    logger.info("Total items: %d", len(merged_items))

    for pair_type in PAIR_TYPES:
        logger.info("\n=== Merging %s ===", pair_type)
        merged = merge_pairs(available_dirs, pair_type, weights)
        if merged:
            save_pairs_csv(merged, output_dir / f"{pair_type}.csv")
            n_pos = sum(1 for _, _, l in merged if l == 1)
            n_neg = len(merged) - n_pos
            logger.info("Total %s: %d (pos=%d, neg=%d, ratio=1:%.1f)",
                        pair_type, len(merged), n_pos, n_neg,
                        n_neg / max(n_pos, 1))
        else:
            logger.info("No %s pairs found across any source", pair_type)

    logger.info("\n=== Merge complete ===")
    logger.info("Output directory: %s", output_dir)
    logger.info("Next step: python train/precompute_backbones.py --data-dir %s", output_dir)


if __name__ == "__main__":
    main()
