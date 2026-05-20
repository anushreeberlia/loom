"""
Prepare training data from FashionStylist dataset.

800+ expert-curated outfits with professional stylist annotations including
detailed item-level attributes: category, style, materials, color, pattern,
occasion, season.

The dataset is text-only (no images), so we generate attribute-based training
pairs. Items within the same outfit are positive pairs; items from different
outfits are negative pairs. The rich attributes enable generating head-specific
pairs (occasion, fit, material, style) that teach the heads what attributes
should co-occur in a well-styled outfit.

Source: https://github.com/recsys-benchmark/FashionStylist

Produces:
    data/fashionstylist/item_metadata.csv
    data/fashionstylist/compat_pairs.csv
    data/fashionstylist/style_pairs.csv
    data/fashionstylist/occasion_pairs.csv
    data/fashionstylist/fit_pairs.csv
    data/fashionstylist/material_pairs.csv

Note: Since there are no images, these items won't appear in backbone_embeddings.h5.
The pairs are still useful for the merge step's engagement-weighted oversampling:
we map FashionStylist item IDs to Polyvore/DeepFashion items with matching attributes
in merge_datasets.py when possible.

Usage:
    python train/data/prepare_fashionstylist.py --output-dir data/fashionstylist
"""

import argparse
import csv
import logging
import random
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/recsys-benchmark/FashionStylist.git"


def download_dataset(output_dir: Path) -> Path:
    """Clone the FashionStylist repository."""
    repo_dir = output_dir / "FashionStylist"
    if repo_dir.exists():
        logger.info("FashionStylist repo already exists at %s", repo_dir)
        return repo_dir

    logger.info("Cloning FashionStylist repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        check=True,
    )
    return repo_dir


def load_items(repo_dir: Path) -> dict[str, dict]:
    """Load item labels from all gender directories."""
    items = {}
    for label_csv in sorted(repo_dir.rglob("label_en.csv")):
        gender_dir = label_csv.parent.name.lower()
        gender_prefix = gender_dir[0]  # f, m, c
        with open(label_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = f"{gender_prefix}_{row['itemID']}"
                items[item_id] = {
                    "item_id": item_id,
                    "category": row.get("category", ""),
                    "style": row.get("style", ""),
                    "outline": row.get("outline", ""),
                    "materials": row.get("materials", ""),
                    "color": row.get("color", ""),
                    "pattern": row.get("pattern", ""),
                    "gender": gender_dir,
                    "title": row.get("title", ""),
                }
    logger.info("Loaded %d items across all genders", len(items))
    return items


def load_outfits(repo_dir: Path) -> dict[str, dict]:
    """Load outfit definitions from look CSVs."""
    outfits = {}
    for look_csv in sorted(repo_dir.rglob("look_en.csv")):
        gender_dir = look_csv.parent.name.lower()
        gender_prefix = gender_dir[0]
        with open(look_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                outfit_id = f"{gender_prefix}_{row['outfitID']}"
                items_str = row.get("items", "")
                item_ids = [f"{gender_prefix}_{iid.strip()}" for iid in items_str.split(",") if iid.strip()]

                outfits[outfit_id] = {
                    "outfit_id": outfit_id,
                    "item_ids": item_ids,
                    "look_description": row.get("look", ""),
                    "season": row.get("season", ""),
                    "occasion": row.get("occasion", ""),
                }
    logger.info("Loaded %d outfits", len(outfits))
    return outfits


def build_compat_pairs(outfits: dict, items: dict, neg_ratio: int = 3) -> list[tuple]:
    """Co-occurring items in an outfit = positive, cross-outfit = negative."""
    all_item_ids = list(items.keys())
    positives = []
    item_coitems = {}

    for outfit in outfits.values():
        ids = [iid for iid in outfit["item_ids"] if iid in items]
        oids = set(ids)
        for iid in ids:
            item_coitems.setdefault(iid, set()).update(oids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                positives.append((ids[i], ids[j], 1))

    random.seed(42)
    n_neg = len(positives) * neg_ratio
    negatives = []
    for _ in range(n_neg):
        a, b = random.sample(all_item_ids, 2)
        if b not in item_coitems.get(a, set()):
            negatives.append((a, b, 0))

    pairs = positives + negatives
    random.shuffle(pairs)
    logger.info("Compat pairs: %d positive, %d negative", len(positives), len(negatives))
    return pairs


def build_attribute_pairs(items: dict, attr_name: str, neg_ratio: int = 2) -> list[tuple]:
    """Positive = shared attribute value, negative = different."""
    by_val = {}
    all_ids = list(items.keys())

    for iid, meta in items.items():
        val = meta.get(attr_name, "").lower().strip()
        if val:
            for v in val.split(","):
                v = v.strip()
                if v:
                    by_val.setdefault(v, []).append(iid)

    positives = []
    for val, ids in by_val.items():
        if len(ids) < 2:
            continue
        sampled = ids[:300]
        for i in range(len(sampled)):
            for j in range(i + 1, min(i + 4, len(sampled))):
                positives.append((sampled[i], sampled[j], 1))

    random.seed(hash(attr_name) % 2**31)
    n_neg = len(positives) * neg_ratio
    negatives = [(random.choice(all_ids), random.choice(all_ids), 0) for _ in range(n_neg)]

    pairs = positives + negatives
    random.shuffle(pairs)
    logger.info("%s pairs: %d positive, %d negative", attr_name, len(positives), len(negatives))
    return pairs


def build_occasion_pairs(outfits: dict, items: dict, neg_ratio: int = 2) -> list[tuple]:
    """Use outfit-level occasion labels for occasion-specific pairs."""
    by_occasion = {}
    all_item_ids = list(items.keys())

    for outfit in outfits.values():
        occasion = outfit.get("occasion", "").lower().strip()
        if not occasion:
            continue
        valid_ids = [iid for iid in outfit["item_ids"] if iid in items]
        by_occasion.setdefault(occasion, []).extend(valid_ids)

    positives = []
    for occ, ids in by_occasion.items():
        unique_ids = list(set(ids))
        if len(unique_ids) < 2:
            continue
        sampled = unique_ids[:200]
        for i in range(len(sampled)):
            for j in range(i + 1, min(i + 4, len(sampled))):
                positives.append((sampled[i], sampled[j], 1))

    random.seed(99)
    n_neg = len(positives) * neg_ratio
    negatives = [(random.choice(all_item_ids), random.choice(all_item_ids), 0) for _ in range(n_neg)]

    pairs = positives + negatives
    random.shuffle(pairs)
    logger.info("Occasion pairs: %d positive, %d negative", len(positives), len(negatives))
    return pairs


def save_pairs_csv(pairs: list[tuple], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_a", "item_b", "label"])
        writer.writeheader()
        for a, b, label in pairs:
            writer.writerow({"item_a": a, "item_b": b, "label": label})
    logger.info("Saved %s (%d pairs)", path, len(pairs))


def save_metadata(items: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "category", "style",
                                                "materials", "color", "pattern", "image_path"])
        writer.writeheader()
        for iid, meta in items.items():
            writer.writerow({
                "item_id": iid,
                "category": meta.get("category", ""),
                "style": meta.get("style", ""),
                "materials": meta.get("materials", ""),
                "color": meta.get("color", ""),
                "pattern": meta.get("pattern", ""),
                "image_path": "",
            })
    logger.info("Saved %s (%d items)", path, len(items))


def main():
    parser = argparse.ArgumentParser(description="Prepare FashionStylist training data")
    parser.add_argument("--output-dir", type=str, default="data/fashionstylist")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = download_dataset(output_dir)
    items = load_items(repo_dir)
    outfits = load_outfits(repo_dir)

    if not items or not outfits:
        logger.error("Failed to load data. Check repo at %s", repo_dir)
        sys.exit(1)

    compat_pairs = build_compat_pairs(outfits, items)
    style_pairs = build_attribute_pairs(items, "style")
    occasion_pairs = build_occasion_pairs(outfits, items)
    fit_pairs = build_attribute_pairs(items, "outline")
    material_pairs = build_attribute_pairs(items, "materials")

    save_metadata(items, output_dir / "item_metadata.csv")
    save_pairs_csv(compat_pairs, output_dir / "compat_pairs.csv")
    save_pairs_csv(style_pairs, output_dir / "style_pairs.csv")
    save_pairs_csv(occasion_pairs, output_dir / "occasion_pairs.csv")
    save_pairs_csv(fit_pairs, output_dir / "fit_pairs.csv")
    save_pairs_csv(material_pairs, output_dir / "material_pairs.csv")

    logger.info("\nDone! FashionStylist data at %s", output_dir)
    logger.info("  Items: %d", len(items))
    logger.info("  Outfits: %d", len(outfits))
    logger.info("  Compat pairs: %d", len(compat_pairs))
    logger.info("  Style pairs: %d", len(style_pairs))
    logger.info("  Occasion pairs: %d", len(occasion_pairs))


if __name__ == "__main__":
    main()
