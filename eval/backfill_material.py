#!/usr/bin/env python3
"""
Backfill material (and other Florence fields) for catalog items that are missing it.

Downloads each item's image from Cloudinary, sends it to the Fashion Florence
HF Space API, and updates the material column in catalog_items.

Usage:
  # Dry run (see what would be updated, no DB writes):
  python eval/backfill_material.py --dry-run

  # Process all items missing material:
  python eval/backfill_material.py

  # Process only first 50:
  python eval/backfill_material.py --limit 50

  # Also update style_tags, occasion_tags, season_tags, fit:
  python eval/backfill_material.py --update-all
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")
FLORENCE_API_URL = os.getenv("FLORENCE_API_URL", "").rstrip("/")


def fetch_image(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
    return None


def call_florence(image_bytes: bytes) -> dict | None:
    """Call Florence API, return raw output or None on failure."""
    if not FLORENCE_API_URL:
        raise ValueError("FLORENCE_API_URL not set in .env")

    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{FLORENCE_API_URL}/analyze",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=120.0,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 503:
                logger.info("  Space waking up (503), waiting 30s...")
                time.sleep(30)
                continue
            logger.warning("  Florence returned %d", resp.status_code)
            return None
        except httpx.TimeoutException:
            logger.warning("  Timeout (attempt %d/3)", attempt + 1)
            time.sleep(10)
        except Exception as e:
            logger.warning("  Error: %s", e)
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill material via Fashion Florence")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process")
    parser.add_argument("--update-all", action="store_true",
                        help="Also update style_tags, occasion_tags, season_tags, fit")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between API calls (be nice to free Space)")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()

    query = """
        SELECT id, name, category, image_url
        FROM catalog_items
        WHERE image_url IS NOT NULL
          AND embedding IS NOT NULL
          AND (material IS NULL OR material = '')
        ORDER BY id
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = cur.fetchall()
    logger.info("Found %d items missing material", len(rows))

    if not rows:
        print("Nothing to backfill!")
        return

    success = 0
    failed = 0

    for i, (item_id, name, category, image_url) in enumerate(rows):
        logger.info("[%d/%d] %s (id=%s, cat=%s)", i + 1, len(rows), name, item_id, category)

        image_bytes = fetch_image(image_url)
        if not image_bytes:
            failed += 1
            continue

        result = call_florence(image_bytes)
        if not result:
            failed += 1
            continue

        material = result.get("material")
        if not material:
            material = "unknown"

        logger.info("  -> material=%s, style_tags=%s", material, result.get("style_tags"))

        if args.dry_run:
            success += 1
            continue

        if args.update_all:
            from services.fashion_florence import expand_florence_output
            expanded = expand_florence_output(result)
            cur.execute("""
                UPDATE catalog_items
                SET material = %s,
                    style_tags = %s,
                    occasion_tags = %s,
                    season_tags = %s,
                    fit = %s,
                    tagged_at = NOW()
                WHERE id = %s
            """, (
                expanded["material"],
                expanded["style_tags"],
                expanded["occasion_tags"],
                expanded["season_tags"],
                expanded["fit"],
                item_id,
            ))
        else:
            cur.execute("""
                UPDATE catalog_items SET material = %s, tagged_at = NOW() WHERE id = %s
            """, (material, item_id))

        conn.commit()
        success += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    cur.close()
    conn.close()

    print(f"\nDone! Success: {success}, Failed: {failed}, Total: {len(rows)}")
    if args.dry_run:
        print("(dry run -- no DB changes made)")


if __name__ == "__main__":
    main()
