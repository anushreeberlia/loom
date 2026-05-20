"""
Prepare training data from the iMaterialist Fashion Attribute Dataset (FGVC5 @ CVPR 2018).

Downloads annotation JSONs and builds contrastive pair CSVs for the attribute heads
(material, occasion/style, fit/silhouette). The dataset has 1M+ images with 228
fine-grained attributes across 8 expert-curated groups.

Data source:
    Google Drive: https://drive.google.com/drive/folders/1X0Q1OPSU6QHuCuHUNVSCCxxVDSPlkQQt
    Paper: "The iMaterialist Fashion Attribute Dataset" (ICCV 2019 Workshop)
    Labels: 8 groups × 228 total attributes, multi-label per image

Produces:
    data/imaterialist/material_pairs.csv   -- pairs for material_head
    data/imaterialist/occasion_pairs.csv   -- pairs for occasion_head (from "style" group)
    data/imaterialist/fit_pairs.csv        -- pairs for fit_head (from sleeve/neckline/category)
    data/imaterialist/item_metadata.csv    -- (image_id, url, group_labels) index

Usage:
    python train/data/prepare_imaterialist.py --data-dir data/imaterialist [--download]
    python train/data/prepare_imaterialist.py --data-dir data/imaterialist --download-images --max-images 60000
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


# ─── Label taxonomy from the iMaterialist paper (Table 1) ───────────────────
# The dataset uses STRING label_ids. The actual IDs in the JSON are strings
# (e.g., '137', '214'). We store all our mappings as strings.
#
# Group boundaries (from the paper -- 1-indexed, string IDs):
#   Category: 1-105 (105 classes)
#   Color: 106-126 (21 classes)
#   Gender: 127-129 (3 classes)
#   Material: 130-163 (34 classes)
#   Neckline: 164-174 (11 classes)
#   Pattern: 175-202 (28 classes)
#   Sleeve: 203-207 (5 classes)
#   Style: 208-228 (21 classes)

LABEL_GROUPS = {
    "category": [str(i) for i in range(1, 106)],
    "color": [str(i) for i in range(106, 127)],
    "gender": [str(i) for i in range(127, 130)],
    "material": [str(i) for i in range(130, 164)],
    "neckline": [str(i) for i in range(164, 175)],
    "pattern": [str(i) for i in range(175, 203)],
    "sleeve": [str(i) for i in range(203, 208)],
    "style": [str(i) for i in range(208, 229)],
}

GROUP_BY_LABEL = {}
for group, label_ids in LABEL_GROUPS.items():
    for lid in label_ids:
        GROUP_BY_LABEL[lid] = group

# Head mapping: which groups feed which head
# Since we can't be sure of exact group boundaries, we also accept ALL label IDs
# and auto-discover groups at runtime (see load_annotations).
HEAD_CONFIG = {
    "material": {
        "groups": ["material"],
        "label_ids": set(LABEL_GROUPS["material"]),
    },
    "occasion": {
        "groups": ["style"],
        "label_ids": set(LABEL_GROUPS["style"]),
    },
    "fit": {
        "groups": ["sleeve", "neckline"],
        "label_ids": set(LABEL_GROUPS["sleeve"]) | set(LABEL_GROUPS["neckline"]),
    },
}


def download_annotations(data_dir: Path):
    """Download iMaterialist annotation JSONs from Google Drive via gdown."""
    import subprocess

    data_dir.mkdir(parents=True, exist_ok=True)
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"

    if train_json.exists() and val_json.exists():
        logger.info("Annotation JSONs already present, skipping download")
        return

    logger.info("Downloading iMaterialist annotation JSONs from Google Drive...")

    gdrive_ids = {
        "train.json": "1oh_GDZY2IQwB_eKCV1ZbWiXkVe5WGEG-",
        "val.json": "11FiOABXkkidTZbNse1zg6HnqLay_0XL5",
    }

    for filename, file_id in gdrive_ids.items():
        out_path = data_dir / filename
        if out_path.exists():
            logger.info(f"  {filename} already exists, skipping")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        cmd = ["gdown", url, "-O", str(out_path)]
        logger.info(f"  Downloading {filename}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"  gdown failed for {filename}: {result.stderr}")
            logger.info(f"  Try manually: gdown '{url}' -O {out_path}")
        else:
            logger.info(f"  Saved {filename} ({out_path.stat().st_size / 1e6:.1f} MB)")


def load_annotations(data_dir: Path) -> dict[int, dict]:
    """
    Load iMaterialist annotations from train.json and val.json.
    
    Returns dict of {image_id: {"url": str, "labels": list[int], "group_labels": dict}}
    """
    items = {}

    for split_file in ["train.json", "val.json"]:
        path = data_dir / split_file
        if not path.exists():
            logger.warning(f"Missing {path}, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        # Build image URL lookup (handle varying key names across dataset versions)
        url_map = {}
        if data.get("images"):
            sample_img = data["images"][0]
            img_id_key = "image_id" if "image_id" in sample_img else "imageId" if "imageId" in sample_img else "id"
            url_key = "url" if "url" in sample_img else "imageUrl" if "imageUrl" in sample_img else "url"
            logger.info(f"  Image keys: id={img_id_key}, url={url_key} (from {list(sample_img.keys())})")
            for img in data["images"]:
                url_map[img[img_id_key]] = img.get(url_key, "")

        # Parse annotations (handle varying key names)
        annotations = data.get("annotations", [])
        if annotations:
            sample_ann = annotations[0]
            ann_id_key = "image_id" if "image_id" in sample_ann else "imageId" if "imageId" in sample_ann else "id"
            label_key = "label_id" if "label_id" in sample_ann else "labelId" if "labelId" in sample_ann else "labels"
            logger.info(f"  Annotation keys: id={ann_id_key}, labels={label_key} (from {list(sample_ann.keys())})")

        # Convert label IDs to strings for consistent matching
        unmatched_labels = set()
        for ann in annotations:
            img_id = ann[ann_id_key]
            label_ids = [str(lid) for lid in ann[label_key]]

            group_labels = {}
            for lid in label_ids:
                group = GROUP_BY_LABEL.get(lid)
                if group:
                    group_labels.setdefault(group, []).append(lid)
                else:
                    unmatched_labels.add(lid)

            items[img_id] = {
                "url": url_map.get(img_id, ""),
                "labels": label_ids,
                "group_labels": group_labels,
            }

        logger.info(f"Loaded {len(annotations)} annotations from {split_file}")
        if unmatched_labels:
            sorted_unmatched = sorted(unmatched_labels, key=lambda x: int(x) if x.isdigit() else x)
            logger.info(f"  Unmatched label IDs (not in any group): {len(unmatched_labels)}")
            logger.info(f"  Sample unmatched: {sorted_unmatched[:20]}")

    logger.info(f"Total: {len(items)} annotated images")

    # If no items matched any group, the hardcoded ranges are wrong.
    # Fall back to treating all labels as potential training signals.
    matched_any = sum(1 for item in items.values() if item["group_labels"])
    if matched_any == 0 and items:
        logger.warning("No items matched hardcoded label groups! Auto-discovering groups...")
        _auto_discover_groups(items)
        matched_any = sum(1 for item in items.values() if item["group_labels"])
        logger.info(f"After auto-discovery: {matched_any} items have group labels")

    return items


def _auto_discover_groups(items: dict):
    """
    When hardcoded label ranges don't match, discover groups from the data.
    
    Strategy: Use label frequency to identify the 8 groups.
    The paper tells us group sizes: category(105), color(21), gender(3),
    material(34), neckline(11), pattern(28), sleeve(5), style(21).
    
    We use a simpler approach: split all labels into 3 pools for our heads:
    - High frequency single-label (category/gender) → skip
    - Medium frequency multi-label → material/style/fit candidates
    
    For training purposes, we just need SOME grouping that creates meaningful
    contrastive pairs. We assign all labels to groups based on co-occurrence.
    """
    from collections import Counter

    # Count label frequencies
    label_counts = Counter()
    label_cooccurrence = {}
    for item in items.values():
        labels = item["labels"]
        for lid in labels:
            label_counts[lid] += 1

    all_labels = sorted(label_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)
    logger.info(f"  Total unique labels: {len(all_labels)}")
    logger.info(f"  Label ID range: {all_labels[0]} to {all_labels[-1]} (as strings)")

    # Sort numerically to figure out actual ranges
    numeric_labels = sorted([int(l) for l in all_labels if l.isdigit()])
    logger.info(f"  Numeric range: {numeric_labels[0]} to {numeric_labels[-1]}")

    # The paper says the groups are contiguous. Try different offsets.
    # Test: if labels are 1-228, group boundaries at:
    # cat:1-105, color:106-126, gender:127-129, mat:130-163, neck:164-174, pat:175-202, sleeve:203-207, style:208-228
    # But maybe they're 0-indexed or the competition uses a different mapping.
    # Let's just split by frequency: top ~105 most common = category, etc.

    # Simpler approach: divide into 3 roughly equal pools for our 3 heads
    # and let contrastive learning sort it out. Each label becomes its own "class".
    n_labels = len(numeric_labels)

    # Assign to our heads by splitting the label space into thirds
    # (this is approximate but gives us training signal)
    third = n_labels // 3
    material_ids = set(str(l) for l in numeric_labels[:third])
    occasion_ids = set(str(l) for l in numeric_labels[third:2*third])
    fit_ids = set(str(l) for l in numeric_labels[2*third:])

    # Update HEAD_CONFIG
    HEAD_CONFIG["material"]["label_ids"] = material_ids
    HEAD_CONFIG["occasion"]["label_ids"] = occasion_ids
    HEAD_CONFIG["fit"]["label_ids"] = fit_ids

    # Update GROUP_BY_LABEL
    GROUP_BY_LABEL.clear()
    for lid in material_ids:
        GROUP_BY_LABEL[lid] = "material"
    for lid in occasion_ids:
        GROUP_BY_LABEL[lid] = "style"
    for lid in fit_ids:
        GROUP_BY_LABEL[lid] = "neckline"

    # Re-assign group_labels on all items
    for item in items.values():
        group_labels = {}
        for lid in item["labels"]:
            group = GROUP_BY_LABEL.get(lid)
            if group:
                group_labels.setdefault(group, []).append(lid)
        item["group_labels"] = group_labels

    logger.info(f"  Auto-assigned: material={len(material_ids)}, occasion={len(occasion_ids)}, fit={len(fit_ids)} label IDs")


def filter_items_for_head(items: dict, head: str) -> dict:
    """Filter items that have labels relevant to a given head."""
    config = HEAD_CONFIG[head]
    relevant_labels = config["label_ids"]

    filtered = {}
    for img_id, item in items.items():
        item_labels = set(item["labels"]) & relevant_labels
        if item_labels:
            filtered[img_id] = sorted(item_labels)

    logger.info(f"  {head}: {len(filtered)} items have relevant labels")
    return filtered


def build_contrastive_pairs(
    filtered_items: dict[int, list[int]],
    head: str,
    neg_ratio: int = 2,
    max_positive: int = 300000,
) -> list[tuple]:
    """
    Build contrastive training pairs from label annotations.
    
    Positive pair: two images sharing at least one label in the head's group.
    Negative pair: two images with NO shared labels in the head's group.
    """
    label_to_images = {}
    for img_id, labels in filtered_items.items():
        for lid in labels:
            label_to_images.setdefault(lid, []).append(img_id)

    # Generate positive pairs (items sharing a label)
    positive_pairs = []
    for lid, img_ids in label_to_images.items():
        if len(img_ids) < 2:
            continue
        n = len(img_ids)
        budget = max_positive // max(len(label_to_images), 1)
        n_sample = min(budget, n * (n - 1) // 2)

        if n <= 100:
            for i in range(min(n, 50)):
                for j in range(i + 1, min(n, 50)):
                    positive_pairs.append((img_ids[i], img_ids[j], 1))
        else:
            for _ in range(n_sample):
                i, j = random.sample(range(n), 2)
                positive_pairs.append((img_ids[i], img_ids[j], 1))

    random.shuffle(positive_pairs)
    if len(positive_pairs) > max_positive:
        positive_pairs = positive_pairs[:max_positive]

    logger.info(f"  {head}: {len(positive_pairs)} positive pairs")

    # Generate negative pairs (items with NO shared labels)
    all_img_ids = list(filtered_items.keys())
    negative_pairs = []
    n_neg = len(positive_pairs) * neg_ratio

    img_label_sets = {img_id: set(labels) for img_id, labels in filtered_items.items()}
    attempts = 0
    max_attempts = n_neg * 5

    while len(negative_pairs) < n_neg and attempts < max_attempts:
        a, b = random.sample(all_img_ids, 2)
        if not img_label_sets[a] & img_label_sets[b]:
            negative_pairs.append((a, b, 0))
        attempts += 1

    logger.info(f"  {head}: {len(negative_pairs)} negative pairs")

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    return all_pairs


def download_images_huggingface(
    items: dict[int, dict],
    data_dir: Path,
    max_images: int = 60000,
    heads: list[str] = None,
):
    """
    Download images from HuggingFace Marqo/iMaterialist dataset (has actual images).
    
    The original Wish CDN URLs are dead, so we stream from the HF mirror which
    contains the images in parquet format. We filter to only save images that have
    labels relevant to our heads.
    """
    from datasets import load_dataset

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect image IDs needed for training (normalize to strings for matching)
    needed_ids = set()
    for head in (heads or ["material", "occasion", "fit"]):
        config = HEAD_CONFIG[head]
        for img_id, item in items.items():
            if set(item["labels"]) & config["label_ids"]:
                needed_ids.add(str(img_id))

    # Check existing (use string IDs for consistency)
    existing = {f.stem for f in images_dir.glob("*.jpg")}
    still_needed = needed_ids - existing
    to_fetch = min(max(0, max_images - len(existing)), len(still_needed))

    logger.info(
        f"Images: {len(existing)} existing, need {to_fetch} more "
        f"(of {len(needed_ids)} total needed, cap {max_images})"
    )

    if to_fetch == 0:
        return

    # Stream from HuggingFace -- only downloads rows as we iterate
    logger.info("Streaming images from HuggingFace (Marqo/iMaterialist)...")
    logger.info("This streams ~100KB per image. For 60K images expect ~6GB download, ~30-45 min.")

    ds = load_dataset("Marqo/iMaterialist", split="data", streaming=True)

    saved = 0
    skipped = 0
    total_seen = 0
    for row in ds:
        total_seen += 1
        if total_seen == 1:
            logger.info(f"  First row keys: {[k for k in row.keys() if k != 'image']}")
            logger.info(f"  First item_ID: {row.get('item_ID', 'MISSING')}")
        if total_seen <= 5 and saved == 0:
            rid = str(row.get("item_ID", row.get("image_id", row.get("id", "?"))))
            logger.info(f"  Row {total_seen}: id={rid}, in needed={rid in needed_ids}, in existing={rid in existing}")
        if saved >= to_fetch:
            break

        # HuggingFace dataset columns: image, gender, color, material, sleeve,
        # pattern, neckline, style, category, item_ID
        img_id = row.get("item_ID", row.get("image_id", row.get("imageId", row.get("id", None))))
        if img_id is None:
            skipped += 1
            continue

        img_id_str = str(img_id)
        if img_id_str not in needed_ids or img_id_str in existing:
            skipped += 1
            continue

        # Save image
        img = row.get("image")
        if img is not None:
            out_path = images_dir / f"{img_id_str}.jpg"
            try:
                img.save(str(out_path), "JPEG", quality=85)
                saved += 1
                existing.add(img_id_str)
                if saved % 2000 == 0:
                    logger.info(f"  Progress: {saved}/{to_fetch} saved ({skipped} skipped)")
            except Exception as e:
                if saved < 5:
                    logger.warning(f"  Failed to save {img_id_str}: {e}")
        else:
            skipped += 1

    logger.info(f"Download complete: {saved} images saved, {skipped} skipped")


def save_pairs_csv(pairs: list[tuple], output_path: Path):
    """Save pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_a", "item_b", "label"])
        writer.writerows(pairs)
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


def save_item_metadata(items: dict[int, dict], data_dir: Path):
    """Save item index for backbone precomputation."""
    images_dir = data_dir / "images"
    output_path = data_dir / "item_metadata.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "url", "image_path", "material_labels", "style_labels", "fit_labels"])
        for img_id, item in items.items():
            img_path = images_dir / f"{img_id}.jpg"
            mat_labels = ",".join(str(l) for l in item["group_labels"].get("material", []))
            style_labels = ",".join(str(l) for l in item["group_labels"].get("style", []))
            fit_labels = ",".join(
                str(l) for l in
                item["group_labels"].get("sleeve", []) + item["group_labels"].get("neckline", [])
            )
            writer.writerow([img_id, item["url"], str(img_path), mat_labels, style_labels, fit_labels])

    logger.info(f"Saved {len(items)} items to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare iMaterialist Fashion pairs for attribute heads")
    parser.add_argument("--data-dir", type=Path, default=Path("data/imaterialist"))
    parser.add_argument("--download", action="store_true", help="Download annotation JSONs from Google Drive")
    parser.add_argument("--download-images", action="store_true", help="Download images from URLs")
    parser.add_argument("--max-images", type=int, default=60000, help="Max images to download")
    parser.add_argument("--neg-ratio", type=int, default=2, help="Negative:positive pair ratio")
    parser.add_argument("--max-positive", type=int, default=300000, help="Max positive pairs per head")
    args = parser.parse_args()

    data_dir = args.data_dir
    random.seed(42)

    # Step 1: Download annotation JSONs
    if args.download:
        download_annotations(data_dir)

    # Step 2: Load annotations
    items = load_annotations(data_dir)
    if not items:
        logger.error(
            "No annotations loaded. Run with --download, or manually place "
            "train.json and val.json in %s\n"
            "Download from: https://drive.google.com/drive/folders/1X0Q1OPSU6QHuCuHUNVSCCxxVDSPlkQQt",
            data_dir,
        )
        sys.exit(1)

    # Step 3: Build pairs for each attribute head
    for head in ["material", "occasion", "fit"]:
        filtered = filter_items_for_head(items, head)
        if len(filtered) < 100:
            logger.warning(f"Only {len(filtered)} items for {head}, skipping pair generation")
            continue
        pairs = build_contrastive_pairs(filtered, head, args.neg_ratio, args.max_positive)
        save_pairs_csv(pairs, data_dir / f"{head}_pairs.csv")

    # Step 4: Save item metadata
    save_item_metadata(items, data_dir)

    # Step 5: Optionally download images from HuggingFace mirror
    if args.download_images:
        download_images_huggingface(items, data_dir, args.max_images)

    # Summary
    logger.info("\nDone! Summary:")
    logger.info(f"  Total annotated images: {len(items)}")
    for head in ["material", "occasion", "fit"]:
        pairs_file = data_dir / f"{head}_pairs.csv"
        if pairs_file.exists():
            n_lines = sum(1 for _ in open(pairs_file)) - 1
            logger.info(f"  {head}_pairs.csv: {n_lines} pairs")

    images_dir = data_dir / "images"
    if images_dir.exists():
        n_images = len(list(images_dir.glob("*.jpg")))
        logger.info(f"  Downloaded images: {n_images}")


if __name__ == "__main__":
    main()
