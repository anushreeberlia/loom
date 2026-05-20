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
# The dataset uses integer label_ids 0-227 across 8 groups.
# Group boundaries (from the paper, confirmed by label counts):
#   Category: 0-104 (105 classes)
#   Color: 105-125 (21 classes)
#   Gender: 126-128 (3 classes)
#   Material: 129-162 (34 classes)
#   Neckline: 163-173 (11 classes)
#   Pattern: 174-201 (28 classes)
#   Sleeve: 202-206 (5 classes)
#   Style: 207-227 (21 classes)

LABEL_GROUPS = {
    "category": list(range(0, 105)),
    "color": list(range(105, 126)),
    "gender": list(range(126, 129)),
    "material": list(range(129, 163)),
    "neckline": list(range(163, 174)),
    "pattern": list(range(174, 202)),
    "sleeve": list(range(202, 207)),
    "style": list(range(207, 228)),
}

GROUP_BY_LABEL = {}
for group, label_ids in LABEL_GROUPS.items():
    for lid in label_ids:
        GROUP_BY_LABEL[lid] = group

# Representative attribute names per group (from paper examples + domain knowledge).
# Not exhaustive but covers most common ones for our head training objectives.
MATERIAL_LABELS = {
    129: "nylon", 130: "organza", 131: "patent", 132: "plush", 133: "rayon",
    134: "silk", 135: "satin", 136: "cotton", 137: "linen", 138: "wool",
    139: "cashmere", 140: "denim", 141: "leather", 142: "suede", 143: "velvet",
    144: "lace", 145: "mesh", 146: "knit", 147: "polyester", 148: "chiffon",
    149: "tweed", 150: "corduroy", 151: "fleece", 152: "jersey", 153: "spandex",
    154: "faux_fur", 155: "sequin", 156: "metallic", 157: "crochet",
    158: "canvas", 159: "rubber", 160: "vinyl", 161: "sheer", 162: "chambray",
}

STYLE_LABELS = {
    207: "asymmetric", 208: "bohemian", 209: "classic", 210: "elegant",
    211: "gothic", 212: "military", 213: "minimalist", 214: "preppy",
    215: "punk", 216: "romantic", 217: "sporty", 218: "streetwear",
    219: "summer", 220: "tunic", 221: "vintage_retro", 222: "wrap",
    223: "workwear", 224: "casual", 225: "formal", 226: "party", 227: "evening",
}

SLEEVE_LABELS = {
    202: "long_sleeved", 203: "puff_sleeves", 204: "short_sleeves",
    205: "sleeveless", 206: "strapless",
}

NECKLINE_LABELS = {
    163: "racerback", 164: "shoulder_drapes", 165: "square_necked",
    166: "turtleneck", 167: "u_neck", 168: "v_neck", 169: "boat_neck",
    170: "crew_neck", 171: "collar", 172: "sweetheart", 173: "off_shoulder",
}

# Head mapping: which groups feed which head
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

        # Build image URL lookup
        url_map = {}
        for img in data.get("images", []):
            url_map[img["image_id"]] = img["url"]

        # Parse annotations
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            label_ids = ann["label_id"]

            group_labels = {}
            for lid in label_ids:
                group = GROUP_BY_LABEL.get(lid)
                if group:
                    group_labels.setdefault(group, []).append(lid)

            items[img_id] = {
                "url": url_map.get(img_id, ""),
                "labels": label_ids,
                "group_labels": group_labels,
            }

        logger.info(f"Loaded {len(data.get('annotations', []))} annotations from {split_file}")

    logger.info(f"Total: {len(items)} annotated images")
    return items


def filter_items_for_head(items: dict[int, dict], head: str) -> dict[int, list[int]]:
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


def download_images(
    items: dict[int, dict],
    data_dir: Path,
    max_images: int = 60000,
    heads: list[str] = None,
):
    """
    Download a subset of images needed for training.
    
    Prioritizes images that have labels for the requested heads.
    Uses concurrent downloads for speed.
    """
    import concurrent.futures
    import urllib.request
    from urllib.error import URLError, HTTPError

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect image IDs needed for training (those with relevant labels)
    needed_ids = set()
    for head in (heads or ["material", "occasion", "fit"]):
        config = HEAD_CONFIG[head]
        for img_id, item in items.items():
            if set(item["labels"]) & config["label_ids"]:
                needed_ids.add(img_id)

    # Filter out already downloaded
    existing = {int(f.stem) for f in images_dir.glob("*.jpg") if f.stem.isdigit()}
    to_download = [(img_id, items[img_id]["url"]) for img_id in needed_ids if img_id not in existing]
    random.shuffle(to_download)
    to_download = to_download[:max(0, max_images - len(existing))]

    logger.info(
        f"Images: {len(existing)} existing, {len(to_download)} to download "
        f"(of {len(needed_ids)} needed, cap {max_images})"
    )

    if not to_download:
        return

    def _download_one(args):
        img_id, url = args
        out_path = images_dir / f"{img_id}.jpg"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(out_path, "wb") as f:
                    f.write(resp.read())
            return True
        except (URLError, HTTPError, OSError):
            return False

    success = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = list(pool.map(_download_one, to_download))
        for ok in futures:
            if ok:
                success += 1
            else:
                fail += 1
            total = success + fail
            if total % 5000 == 0:
                logger.info(f"  Progress: {total}/{len(to_download)} ({success} ok, {fail} failed)")

    logger.info(f"Download complete: {success} succeeded, {fail} failed")


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

    # Step 5: Optionally download images
    if args.download_images:
        download_images(items, data_dir, args.max_images)

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
