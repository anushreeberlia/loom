"""
Background ingestion worker for closet items.

Processes pending items through the vision + embedding pipeline with:
- Controlled concurrency (semaphore-based, default 4)
- Retry with exponential backoff (3 attempts)
- Individual item isolation (one failure doesn't block others)
- Startup recovery (stale 'processing' items reset to 'pending')

Item state machine:
    pending → processing → ready (success)
    processing → pending (transient failure, retry with backoff)
    processing → error (permanent failure after max retries)
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import psycopg2
from dotenv import load_dotenv

from services.item_processor import process_item_from_image
from services.image_processor import process_clothing_image

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

WORKER_CONCURRENCY = 4
MAX_RETRIES = 3
BACKOFF_BASE_SEC = 5
ORPHAN_SWEEP_INTERVAL = 30


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def process_batch_items(
    batch_id: str,
    items: list[tuple[int, str]],
    precomputed_embeddings: dict[int, list] | None = None,
    raw_image_bytes: dict[int, bytes] | None = None,
):
    """
    Process a batch of items. Called from FastAPI BackgroundTasks.

    Args:
        batch_id: UUID grouping these items
        items: list of (item_id, image_url) tuples
        precomputed_embeddings: optional dict of {item_id: embedding} from temporal
            aggregation (video pipeline). Items with pre-computed embeddings skip
            the embedding step and only run vision for metadata.
        raw_image_bytes: optional dict of {item_id: original_bytes} -- the raw
            upload bytes before any processing. Used for better segmentation
            and vision analysis. Falls back to downloading from Cloudinary if missing.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _process_batch_async(
                batch_id, items,
                precomputed_embeddings or {},
                raw_image_bytes or {},
            )
        )
    finally:
        loop.close()


async def _process_batch_async(
    batch_id: str,
    items: list[tuple[int, str]],
    precomputed_embeddings: dict[int, list],
    raw_image_bytes: dict[int, bytes],
):
    """Process items with controlled concurrency."""
    semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)

    async def _process_one(item_id: int, image_url: str):
        async with semaphore:
            pre_emb = precomputed_embeddings.get(item_id)
            raw_bytes = raw_image_bytes.get(item_id)
            await _process_single_item(
                item_id, image_url,
                precomputed_embedding=pre_emb,
                raw_bytes=raw_bytes,
            )

    tasks = [_process_one(item_id, url) for item_id, url in items]
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"Batch {batch_id}: finished processing {len(items)} items")


async def _process_single_item(
    item_id: int,
    image_url: str,
    precomputed_embedding: list | None = None,
    raw_bytes: bytes | None = None,
):
    """Process one closet item through vision + embedding pipeline."""
    _update_status(item_id, "processing")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _blocking_process, item_id, image_url, precomputed_embedding, raw_bytes
        )

        if result["success"]:
            _save_processed_item(item_id, result["base_item"], result["embedding"])
            logger.info(f"Item {item_id} ready: {result['base_item'].get('category')}")
        else:
            _handle_failure(item_id, result["error"], result.get("is_transient", False))

    except Exception as e:
        _handle_failure(item_id, str(e), is_transient=False)


def _blocking_process(
    item_id: int,
    image_url: str,
    precomputed_embedding: list | None = None,
    raw_bytes: bytes | None = None,
) -> dict:
    """
    Blocking I/O: vision → segmentation → embedding.

    Uses raw_bytes (original upload) when available for best segmentation quality.
    Falls back to downloading from Cloudinary (the processed display image) for
    retries and orphan sweeps where raw bytes are no longer in memory.

    If precomputed_embedding is provided (from temporal aggregation), skips the
    embedding computation and only runs vision for metadata extraction.
    """
    try:
        if raw_bytes:
            image_bytes = raw_bytes
        else:
            response = httpx.get(image_url, timeout=15.0)
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Image download failed: HTTP {response.status_code}",
                    "is_transient": response.status_code >= 500,
                }
            image_bytes = response.content

        if precomputed_embedding:
            from services.vision import analyze_image
            base_item = analyze_image(image_bytes)
            embedding = precomputed_embedding
            logger.debug(f"Item {item_id}: using temporal aggregated embedding")
        else:
            _description, base_item, embedding = process_item_from_image(image_bytes)

        return {"success": True, "base_item": base_item, "embedding": embedding}

    except Exception as e:
        err_str = str(e)
        is_transient = any(k in err_str.lower() for k in [
            "rate_limit", "429", "too many requests",
            "503", "timeout", "timed out", "connection",
        ])
        return {"success": False, "error": err_str, "is_transient": is_transient}


def _handle_failure(item_id: int, error: str, is_transient: bool):
    """Handle a processing failure with retry logic."""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT retry_count FROM user_closet_items WHERE id = %s", (item_id,)
        )
        row = cur.fetchone()
        if not row:
            return
        retry_count = row[0]

        if is_transient and retry_count < MAX_RETRIES:
            wait = BACKOFF_BASE_SEC * (2 ** retry_count)
            cur.execute(
                """UPDATE user_closet_items
                   SET status = 'pending', retry_count = retry_count + 1
                   WHERE id = %s""",
                (item_id,),
            )
            conn.commit()
            logger.warning(
                f"Item {item_id} retry {retry_count + 1}/{MAX_RETRIES} "
                f"(backoff {wait}s): {error[:200]}"
            )
            time.sleep(wait)
        else:
            cur.execute(
                """UPDATE user_closet_items
                   SET status = 'error', processing_error = %s
                   WHERE id = %s""",
                (error[:500], item_id),
            )
            conn.commit()
            logger.error(f"Item {item_id} permanently failed: {error[:200]}")
    finally:
        cur.close()
        conn.close()


def _save_processed_item(item_id: int, base_item: dict, embedding: list):
    """Update item with vision results and embedding. Marks status='ready'."""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        name = f"{base_item.get('primary_color', '')} {base_item.get('category', 'item')}".strip().title()
        cur.execute(
            """UPDATE user_closet_items SET
               status = 'ready',
               name = %s,
               category = %s,
               primary_color = %s,
               secondary_colors = %s,
               style_tags = %s,
               season_tags = %s,
               occasion_tags = %s,
               material = %s,
               fit = %s,
               embedding = %s,
               processed_at = NOW(),
               processing_error = NULL
            WHERE id = %s""",
            (
                name,
                base_item.get("category", "top"),
                base_item.get("primary_color"),
                base_item.get("secondary_colors"),
                base_item.get("style_tags"),
                base_item.get("season_tags"),
                base_item.get("occasion_tags"),
                base_item.get("material"),
                base_item.get("fit"),
                embedding,
                item_id,
            ),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def _update_status(item_id: int, status: str):
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE user_closet_items SET status = %s WHERE id = %s",
            (status, item_id),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


# --- Startup recovery ---

def recover_stale_items():
    """
    On server startup: reset items stuck in 'processing' (server crashed mid-work).
    They become 'pending' and will be picked up by the orphan sweep.
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """UPDATE user_closet_items
               SET status = 'pending'
               WHERE status = 'processing'
               RETURNING id"""
        )
        recovered = cur.fetchall()
        conn.commit()
        if recovered:
            logger.info(f"Recovered {len(recovered)} stale processing items on startup")
    finally:
        cur.close()
        conn.close()


async def orphan_sweep_loop():
    """
    Periodic task: find pending items that need processing (orphans from crashes
    or retries ready for another attempt) and process them.
    """
    while True:
        await asyncio.sleep(ORPHAN_SWEEP_INTERVAL)
        try:
            conn = _get_conn()
            cur = conn.cursor()
            try:
                cur.execute(
                    """SELECT id, image_url FROM user_closet_items
                       WHERE status = 'pending'
                       ORDER BY created_at
                       LIMIT 10"""
                )
                orphans = cur.fetchall()
            finally:
                cur.close()
                conn.close()

            if orphans:
                logger.info(f"Orphan sweep: processing {len(orphans)} pending items")
                items = [(row[0], row[1]) for row in orphans]
                await _process_batch_async("orphan-sweep", items, {}, {})
        except Exception as e:
            logger.error(f"Orphan sweep error: {e}")
