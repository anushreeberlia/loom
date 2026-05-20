"""
Download Polyvore item images from URLs stored in the outfit JSONs.

The original polyvore.com is dead, but images are hosted on CDNs that
partially still work (e.g., akamaized.net). Falls back to web archive.

Downloads images to data/polyvore/images/{set_id}_{index}.jpg

Usage:
    python train/download_images.py --data-dir data/polyvore --workers 32
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_image_urls(data_dir: Path) -> dict[str, str]:
    """Extract item_id -> image_url mapping from outfit JSONs."""
    search_paths = [
        data_dir / "polyvore-dataset",
        data_dir,
    ]

    json_files = []
    for sp in search_paths:
        if not sp.exists():
            continue
        for name in ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]:
            p = sp / name
            if p.exists():
                json_files.append(p)

    if not json_files:
        json_files = list(data_dir.rglob("*_no_dup.json"))

    items = {}
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
        for outfit in data:
            set_id = str(outfit.get("set_id", ""))
            for item in outfit.get("items", []):
                index = item.get("index", "")
                image_url = item.get("image", "")
                if set_id and index is not None and image_url:
                    item_id = f"{set_id}_{index}"
                    items[item_id] = image_url

    logger.info(f"Found {len(items)} items with image URLs")
    return items


def download_one(item_id: str, url: str, images_dir: Path) -> bool:
    """Download a single image. Returns True on success."""
    output_path = images_dir / f"{item_id}.jpg"
    if output_path.exists() and output_path.stat().st_size > 100:
        return True

    # Try original URL first
    urls_to_try = [url]

    # Try web archive as fallback
    if "polyvore.com" in url or "polyvore-int" in url:
        urls_to_try.append(f"https://web.archive.org/web/2018/{url}")

    for try_url in urls_to_try:
        try:
            resp = httpx.get(try_url, timeout=10.0, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 100:
                content_type = resp.headers.get("content-type", "")
                if "image" in content_type or resp.content[:3] in [b'\xff\xd8\xff', b'\x89PN']:
                    output_path.write_bytes(resp.content)
                    return True
        except Exception:
            continue

    return False


def main():
    parser = argparse.ArgumentParser(description="Download Polyvore images")
    parser.add_argument("--data-dir", type=Path, default=Path("data/polyvore"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Limit downloads (0=all)")
    args = parser.parse_args()

    items = load_image_urls(args.data_dir)
    if not items:
        logger.error("No image URLs found. Run prepare_polyvore.py first.")
        sys.exit(1)

    images_dir = args.data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already exist
    existing = sum(1 for item_id in items if (images_dir / f"{item_id}.jpg").exists())
    logger.info(f"Already have {existing}/{len(items)} images")

    to_download = [(k, v) for k, v in items.items()
                   if not (images_dir / f"{k}.jpg").exists()]

    if args.limit > 0:
        to_download = to_download[:args.limit]

    if not to_download:
        logger.info("All images already downloaded!")
        return

    logger.info(f"Downloading {len(to_download)} images with {args.workers} workers...")

    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, item_id, url, images_dir): item_id
            for item_id, url in to_download
        }

        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success += 1
            else:
                failed += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    f"Progress: {i+1}/{len(to_download)} | "
                    f"success={success} failed={failed} | "
                    f"{rate:.0f} items/sec | "
                    f"ETA: {(len(to_download) - i - 1) / rate / 60:.1f} min"
                )

    elapsed = time.time() - t0
    logger.info(
        f"\nDone in {elapsed/60:.1f} minutes.\n"
        f"  Success: {success}\n"
        f"  Failed: {failed}\n"
        f"  Total images on disk: {success + existing}"
    )


if __name__ == "__main__":
    main()
