"""
Backfill DINOv2 multi-head embeddings for existing closet and catalog items.

Processes items that have a legacy FashionCLIP embedding but no backbone_embedding.
Downloads images from Cloudinary, runs segmentation + DINOv2 + multi-head projections,
and writes the 6 new vectors (backbone + 5 heads) back to the database.

Usage:
    python scripts/backfill_multihead.py [--batch-size 32] [--table user_closet_items]

Environment:
    DATABASE_URL -- PostgreSQL connection string
"""

import argparse
import logging
import os
import sys
import time

import httpx
import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


def get_items_needing_backfill(conn, table: str, batch_size: int) -> list[tuple]:
    """Fetch items that have legacy embedding but no backbone_embedding yet."""
    cur = conn.cursor()
    cur.execute(
        f"""SELECT id, image_url FROM {table}
            WHERE embedding IS NOT NULL
              AND backbone_embedding IS NULL
            ORDER BY id
            LIMIT %s""",
        (batch_size,),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def download_image(url: str) -> bytes | None:
    """Download image from Cloudinary URL."""
    try:
        resp = httpx.get(url, timeout=15.0)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        logger.warning("Download failed for %s: %s", url, e)
    return None


def process_batch(items: list[tuple]) -> list[tuple]:
    """
    Process a batch of (id, image_url) into (id, backbone, heads) tuples.
    Returns only successfully processed items.
    """
    from services.segmentation import segment_for_embedding
    from services.dinov2 import embed_images
    from services.multihead import compute_multihead_embeddings_batch

    # Download and segment all images
    valid_items = []
    segmented_bytes_list = []

    for item_id, image_url in items:
        raw = download_image(image_url)
        if raw is None:
            logger.warning("Skipping item %d: download failed", item_id)
            continue
        try:
            seg = segment_for_embedding(raw)
            segmented_bytes_list.append(seg)
            valid_items.append(item_id)
        except Exception as e:
            logger.warning("Skipping item %d: segmentation failed: %s", item_id, e)

    if not valid_items:
        return []

    # Batch DINOv2 encoding
    backbone_embeddings = embed_images(segmented_bytes_list, batch_size=16)

    # Batch multi-head projection
    head_embeddings = compute_multihead_embeddings_batch(backbone_embeddings)

    results = []
    for i, item_id in enumerate(valid_items):
        backbone = backbone_embeddings[i]
        heads = {name: head_embeddings[name][i] for name in head_embeddings}
        results.append((item_id, backbone, heads))

    return results


def save_batch(conn, table: str, results: list[tuple]):
    """Write backbone + head embeddings back to the database."""
    cur = conn.cursor()
    for item_id, backbone, heads in results:
        cur.execute(
            f"""UPDATE {table} SET
                backbone_embedding = %s,
                style_embedding = %s,
                fit_embedding = %s,
                material_embedding = %s,
                compat_embedding = %s,
                occasion_embedding = %s
            WHERE id = %s""",
            (
                backbone.tolist(),
                heads["style"].tolist(),
                heads["fit"].tolist(),
                heads["material"].tolist(),
                heads["compat"].tolist(),
                heads["occasion"].tolist(),
                item_id,
            ),
        )
    conn.commit()
    cur.close()


def run_backfill(table: str, batch_size: int):
    """Main backfill loop -- processes all items missing multi-head embeddings."""
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    total_processed = 0

    logger.info("Starting backfill for table=%s, batch_size=%d", table, batch_size)

    while True:
        items = get_items_needing_backfill(conn, table, batch_size)
        if not items:
            logger.info("Backfill complete. Total processed: %d", total_processed)
            break

        logger.info("Processing batch of %d items (ids %d-%d)...", len(items), items[0][0], items[-1][0])
        t0 = time.time()

        results = process_batch(items)

        if results:
            save_batch(conn, table, results)
            total_processed += len(results)
            elapsed = time.time() - t0
            logger.info(
                "Batch done: %d/%d succeeded in %.1fs (%.2f items/sec)",
                len(results), len(items), elapsed, len(results) / elapsed,
            )
        else:
            logger.warning("Batch produced no results (all downloads/segmentations failed)")
            # Mark these as processed to avoid infinite loop
            cur = conn.cursor()
            for item_id, _ in items:
                cur.execute(
                    f"UPDATE {table} SET backbone_embedding = %s WHERE id = %s",
                    (np.zeros(768).tolist(), item_id),
                )
            conn.commit()
            cur.close()
            total_processed += len(items)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill DINOv2 multi-head embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Items per batch")
    parser.add_argument(
        "--table",
        choices=["user_closet_items", "catalog_items"],
        default="user_closet_items",
        help="Which table to backfill",
    )
    args = parser.parse_args()

    run_backfill(args.table, args.batch_size)
