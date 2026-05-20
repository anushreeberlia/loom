"""
Scrape Pinterest outfit boards and extract individual garment crops.

Uses open-source pinterest-crawler (free, no API key) to download outfit
images, then runs YOLO detection + segmentation to extract individual
garment crops per outfit.

Produces:
    data/pinterest/item_metadata.csv  -- (item_id, outfit_id, category, image_path)
    data/pinterest/compat_pairs.csv   -- (item_a, item_b, label) for compat_head
    data/pinterest/style_pairs.csv    -- (item_a, item_b, label) for style_head
    data/pinterest/images/            -- individual garment crop JPEGs

Usage:
    pip install pinterest-crawler
    python train/data/prepare_pinterest.py --output-dir data/pinterest --n-pins 5000
    python train/data/prepare_pinterest.py --output-dir data/pinterest --n-pins 5000 --skip-scrape
"""

import argparse
import csv
import io
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTFIT_QUERIES = [
    "outfit inspo women",
    "street style outfit",
    "fall outfit ideas",
    "spring outfit ideas",
    "work outfit professional",
    "date night outfit",
    "bold outfit ideas",
    "pattern mixing outfit",
    "maximalist fashion outfit",
    "chic outfit women",
    "casual outfit ideas women",
    "elegant evening outfit",
    "trendy outfit 2025",
    "color blocking outfit",
    "layered outfit ideas",
]


def scrape_pinterest(output_dir: Path, n_pins_per_query: int = 400):
    """Download outfit images from Pinterest using pinterest-crawler."""
    raw_dir = output_dir / "raw_pins"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pinterest_crawler import PinterestCrawler
        crawler = PinterestCrawler()
    except ImportError:
        logger.info("pinterest-crawler not found, trying pinterest-dl...")
        try:
            import subprocess
            for query in OUTFIT_QUERIES:
                q_dir = raw_dir / query.replace(" ", "_")
                q_dir.mkdir(exist_ok=True)
                logger.info("Scraping: %s (%d pins)", query, n_pins_per_query)
                subprocess.run([
                    sys.executable, "-m", "pinterest_dl",
                    "--query", query,
                    "--output", str(q_dir),
                    "--limit", str(n_pins_per_query),
                ], check=False, timeout=300)
            return raw_dir
        except Exception as e:
            logger.error("No Pinterest scraper available: %s", e)
            logger.error("Install: pip install pinterest-crawler  OR  pip install pinterest-dl")
            sys.exit(1)

    for query in OUTFIT_QUERIES:
        q_dir = raw_dir / query.replace(" ", "_")
        q_dir.mkdir(exist_ok=True)
        logger.info("Scraping: %s (%d pins)", query, n_pins_per_query)
        try:
            crawler.crawl(query, n_pins_per_query, str(q_dir))
        except Exception as e:
            logger.warning("Failed to scrape '%s': %s", query, e)

    n_images = len(list(raw_dir.rglob("*.jpg"))) + len(list(raw_dir.rglob("*.png")))
    logger.info("Scraped %d total images across %d queries", n_images, len(OUTFIT_QUERIES))
    return raw_dir


def detect_and_crop_garments(raw_dir: Path, output_dir: Path, min_area_ratio: float = 0.02):
    """
    Run YOLO on each outfit image, crop detected objects, segment backgrounds.
    Returns list of (outfit_id, item_crops) where item_crops is a list of
    (item_id, crop_bytes, image_path).
    """
    from services.object_tracker import YOLODetector
    from services.segmentation import segment_for_embedding

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(confidence=0.35)

    all_image_paths = sorted(
        list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) +
        list(raw_dir.rglob("*.jpeg")) + list(raw_dir.rglob("*.webp"))
    )
    logger.info("Processing %d outfit images through YOLO + segmentation", len(all_image_paths))

    outfits = {}
    item_counter = 0

    for img_idx, img_path in enumerate(all_image_paths):
        if img_idx % 100 == 0:
            logger.info("  [%d/%d] Processing %s", img_idx, len(all_image_paths), img_path.name)

        try:
            img_bytes = img_path.read_bytes()
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
        except Exception:
            continue

        detections = detector.detect_from_bytes(img_bytes)
        if len(detections) < 2:
            continue

        outfit_id = f"pin_{img_idx:06d}"
        outfit_items = []

        for det_idx, det in enumerate(detections):
            bx, by, bw, bh = det["bbox"]
            area_ratio = (bw * bh) / (w * h)
            if area_ratio < min_area_ratio:
                continue

            x1, y1 = max(0, int(bx)), max(0, int(by))
            x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            crop_bytes = buf.getvalue()

            try:
                seg_bytes = segment_for_embedding(crop_bytes)
            except Exception:
                seg_bytes = crop_bytes

            item_id = f"pin_{img_idx:06d}_{det_idx:02d}"
            item_path = images_dir / f"{item_id}.jpg"
            item_path.write_bytes(seg_bytes)

            outfit_items.append({
                "item_id": item_id,
                "outfit_id": outfit_id,
                "image_path": str(item_path.relative_to(output_dir)),
                "confidence": det["confidence"],
            })
            item_counter += 1

        if len(outfit_items) >= 2:
            outfits[outfit_id] = outfit_items

    logger.info("Extracted %d items from %d outfits (from %d images)",
                item_counter, len(outfits), len(all_image_paths))
    return outfits


def build_pairs(outfits: dict, neg_ratio: int = 3) -> tuple[list, list]:
    """Build compat and style pairs from Pinterest outfits."""
    all_item_ids = set()
    compat_positives = []

    for outfit_id, items in outfits.items():
        ids = [it["item_id"] for it in items]
        all_item_ids.update(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                compat_positives.append((ids[i], ids[j], 1))

    all_ids_list = list(all_item_ids)
    logger.info("Pinterest compat positives: %d", len(compat_positives))

    item_coitems = {}
    for outfit_id, items in outfits.items():
        oids = {it["item_id"] for it in items}
        for iid in oids:
            item_coitems.setdefault(iid, set()).update(oids)

    random.seed(42)
    n_neg = len(compat_positives) * neg_ratio
    anchors = [p[0] for p in compat_positives] * neg_ratio
    random.shuffle(anchors)
    anchors = anchors[:n_neg]
    neg_items = random.choices(all_ids_list, k=n_neg)
    compat_negatives = []
    for a, n in zip(anchors, neg_items):
        if n not in item_coitems.get(a, set()):
            compat_negatives.append((a, n, 0))
        else:
            alt = random.choice(all_ids_list)
            compat_negatives.append((a, alt, 0))

    compat_pairs = compat_positives + compat_negatives
    random.shuffle(compat_pairs)

    style_pairs = list(compat_positives) + random.sample(
        compat_negatives, min(len(compat_positives), len(compat_negatives))
    )
    random.shuffle(style_pairs)

    logger.info("Pinterest pairs: %d compat, %d style", len(compat_pairs), len(style_pairs))
    return compat_pairs, style_pairs


def save_outputs(output_dir: Path, outfits: dict, compat_pairs: list, style_pairs: list):
    """Write CSVs in the format expected by train_heads.py."""
    meta_path = output_dir / "item_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "outfit_id", "image_path"])
        writer.writeheader()
        for outfit_id, items in outfits.items():
            for it in items:
                writer.writerow({
                    "item_id": it["item_id"],
                    "outfit_id": outfit_id,
                    "image_path": it["image_path"],
                })
    logger.info("Saved %s", meta_path)

    for name, pairs in [("compat_pairs.csv", compat_pairs), ("style_pairs.csv", style_pairs)]:
        path = output_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item_a", "item_b", "label"])
            writer.writeheader()
            for a, b, label in pairs:
                writer.writerow({"item_a": a, "item_b": b, "label": label})
        logger.info("Saved %s (%d pairs)", path, len(pairs))


def main():
    parser = argparse.ArgumentParser(description="Prepare Pinterest outfit training data")
    parser.add_argument("--output-dir", type=str, default="data/pinterest")
    parser.add_argument("--n-pins", type=int, default=5000,
                        help="Total pins to scrape (split across queries)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, process existing raw_pins/")
    parser.add_argument("--neg-ratio", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_scrape:
        n_per_query = max(1, args.n_pins // len(OUTFIT_QUERIES))
        raw_dir = scrape_pinterest(output_dir, n_per_query)
    else:
        raw_dir = output_dir / "raw_pins"
        if not raw_dir.exists():
            logger.error("No raw_pins/ directory found. Run without --skip-scrape first.")
            sys.exit(1)

    outfits = detect_and_crop_garments(raw_dir, output_dir)
    if not outfits:
        logger.error("No outfits extracted. Check YOLO detection or image quality.")
        sys.exit(1)

    compat_pairs, style_pairs = build_pairs(outfits, neg_ratio=args.neg_ratio)
    save_outputs(output_dir, outfits, compat_pairs, style_pairs)

    logger.info("Done! Pinterest data ready at %s", output_dir)
    logger.info("  Outfits: %d", len(outfits))
    logger.info("  Items: %d", sum(len(v) for v in outfits.values()))
    logger.info("  Compat pairs: %d", len(compat_pairs))
    logger.info("  Style pairs: %d", len(style_pairs))


if __name__ == "__main__":
    main()
