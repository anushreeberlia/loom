"""
Prepare training data from STL (Shop The Look) Dataset.

72K product-scene associations from real e-commerce: products detected in
styled scene images. Products co-occurring in the same scene = positive
compatibility signal (they were styled together in a real look).

Source: https://github.com/kang205/STL-Dataset

Produces:
    data/fashionrec/item_metadata.csv
    data/fashionrec/compat_pairs.csv
    data/fashionrec/style_pairs.csv

Usage:
    python train/data/prepare_fashionrec.py --output-dir data/fashionrec
"""

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/kang205/STL-Dataset.git"


def download_dataset(output_dir: Path) -> Path:
    """Clone STL-Dataset."""
    repo_dir = output_dir / "STL-Dataset"
    if repo_dir.exists():
        logger.info("STL-Dataset repo exists at %s", repo_dir)
        return repo_dir

    logger.info("Cloning STL-Dataset...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
        check=True,
    )
    return repo_dir


def load_stl_data(repo_dir: Path) -> tuple[dict, dict]:
    """
    Load STL fashion data.
    
    fashion.json: JSONL, each line = {product, scene, bbox}
    fashion-cat.json: single JSON mapping product_id -> category
    
    Returns:
        scenes: {scene_id: [product_ids]}
        categories: {product_id: category_string}
    """
    fashion_path = repo_dir / "fashion.json"
    cat_path = repo_dir / "fashion-cat.json"

    categories = {}
    if cat_path.exists():
        with open(cat_path) as f:
            categories = json.load(f)
        logger.info("Loaded %d product categories", len(categories))

    scenes = {}
    if fashion_path.exists():
        with open(fashion_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                product_id = entry.get("product", "")
                scene_id = entry.get("scene", "")
                if product_id and scene_id:
                    scenes.setdefault(scene_id, set()).add(product_id)

    scenes = {sid: list(pids) for sid, pids in scenes.items() if len(pids) >= 2}
    all_products = set()
    for pids in scenes.values():
        all_products.update(pids)

    logger.info("Loaded %d scenes with 2+ products, %d unique products",
                len(scenes), len(all_products))
    return scenes, categories


def build_compat_pairs(scenes: dict, neg_ratio: int = 3) -> list[tuple]:
    """Products in the same scene = positive, cross-scene = negative."""
    all_products = set()
    positives = []
    product_coscene = {}

    for scene_id, products in scenes.items():
        pset = set(products)
        all_products.update(products)
        for pid in products:
            product_coscene.setdefault(pid, set()).update(pset)

        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                positives.append((products[i], products[j], 1))

    all_list = list(all_products)
    random.seed(42)
    n_neg = len(positives) * neg_ratio
    negatives = []
    for _ in range(n_neg):
        a, b = random.sample(all_list, 2)
        if b not in product_coscene.get(a, set()):
            negatives.append((a, b, 0))

    pairs = positives + negatives
    random.shuffle(pairs)
    logger.info("Compat pairs: %d positive, %d negative", len(positives), len(negatives))
    return pairs


def build_style_pairs(scenes: dict, categories: dict, neg_ratio: int = 2) -> list[tuple]:
    """
    Style pairs: products in the same scene that are different categories
    (cross-category co-occurrence is a stronger style signal than same-category).
    """
    positives = []
    all_products = set()

    for scene_id, products in scenes.items():
        all_products.update(products)
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                cat_i = categories.get(products[i], "")
                cat_j = categories.get(products[j], "")
                if cat_i != cat_j or not cat_i:
                    positives.append((products[i], products[j], 1))

    all_list = list(all_products)
    random.seed(123)
    n_neg = len(positives) * neg_ratio
    negatives = [(random.choice(all_list), random.choice(all_list), 0) for _ in range(n_neg)]

    pairs = positives + negatives
    random.shuffle(pairs)
    logger.info("Style pairs: %d positive, %d negative", len(positives), len(negatives))
    return pairs


def save_outputs(output_dir: Path, scenes: dict, categories: dict,
                 compat_pairs: list, style_pairs: list):
    """Write CSVs."""
    all_products = set()
    for pids in scenes.values():
        all_products.update(pids)

    meta_path = output_dir / "item_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "category", "image_path"])
        writer.writeheader()
        for pid in sorted(all_products):
            cat = categories.get(pid, "")
            writer.writerow({"item_id": pid, "category": cat, "image_path": ""})
    logger.info("Saved %s (%d items)", meta_path, len(all_products))

    for name, pairs in [("compat_pairs.csv", compat_pairs), ("style_pairs.csv", style_pairs)]:
        path = output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item_a", "item_b", "label"])
            writer.writeheader()
            for a, b, label in pairs:
                writer.writerow({"item_a": a, "item_b": b, "label": label})
        logger.info("Saved %s (%d pairs)", path, len(pairs))


def main():
    parser = argparse.ArgumentParser(description="Prepare STL/FashionRec training data")
    parser.add_argument("--output-dir", type=str, default="data/fashionrec")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = download_dataset(output_dir)
    scenes, categories = load_stl_data(repo_dir)

    if not scenes:
        logger.error("No scene data found. Check repo at %s", repo_dir)
        sys.exit(1)

    compat_pairs = build_compat_pairs(scenes)
    style_pairs = build_style_pairs(scenes, categories)
    save_outputs(output_dir, scenes, categories, compat_pairs, style_pairs)

    logger.info("\nDone! STL/FashionRec data at %s", output_dir)
    logger.info("  Scenes: %d", len(scenes))
    total_products = set()
    for pids in scenes.values():
        total_products.update(pids)
    logger.info("  Products: %d", len(total_products))
    logger.info("  Compat pairs: %d", len(compat_pairs))
    logger.info("  Style pairs: %d", len(style_pairs))


if __name__ == "__main__":
    main()
