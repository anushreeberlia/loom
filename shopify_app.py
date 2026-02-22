"""
Shopify App FastAPI backend.

Endpoints:
  POST /shopify/install          - Store access token after OAuth
  POST /shopify/catalog/sync     - Fetch + process merchant's catalog
  POST /shopify/catalog/process  - Process next batch of unprocessed items
  GET  /shopify/outfits          - Get pre-generated outfits for a product
  POST /shopify/webhooks/product_created  - New product added to store
  POST /shopify/webhooks/app_uninstalled  - Clean up on uninstall
"""

import asyncio
import hashlib
import hmac
import logging
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psycopg2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.item_processor import process_item_from_image_url
from services.outfit_generator import run_outfit_generation
from services.shopify_catalog import (
    fetch_shopify_products,
    get_all_catalog_item_stubs,
    get_generated_outfits,
    get_shopify_catalog_items,
    get_unprocessed_items,
    save_generated_outfits,
    save_processed_item,
    save_processing_error,
    update_store_stats,
    upsert_shopify_catalog_item,
)
from services.collage import create_grid_collage

import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")

app = FastAPI(title="Loom Shopify App Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class InstallRequest(BaseModel):
    shop_domain: str
    access_token: str
    scope: str = ""


class SyncRequest(BaseModel):
    shop_domain: str
    access_token: str


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def get_store(shop_domain: str) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT shop_domain, access_token FROM shopify_stores WHERE shop_domain = %s AND uninstalled_at IS NULL",
            (shop_domain,),
        )
        row = cur.fetchone()
        return {"shop_domain": row[0], "access_token": row[1]} if row else None
    finally:
        cur.close()
        conn.close()


# ── Webhook verification ───────────────────────────────────────────────────────

def verify_shopify_webhook(data: bytes, hmac_header: str) -> bool:
    if not SHOPIFY_API_SECRET:
        return True  # Skip in dev
    digest = hmac.new(
        SHOPIFY_API_SECRET.encode("utf-8"), data, hashlib.sha256
    ).hexdigest()
    import base64
    computed = base64.b64encode(digest.encode()).decode()
    return hmac.compare_digest(computed, hmac_header or "")


# ── Background processing ──────────────────────────────────────────────────────

def process_single_item(shop_domain: str, item: dict) -> bool:
    """
    Run one catalog item through the vision + embedding pipeline.
    Downloads the image, describes it, parses tags, generates embedding.
    Retries up to 4 times on OpenAI rate limit errors (429).
    Returns True on success.
    """
    import time, re
    max_retries = 4
    for attempt in range(max_retries):
        try:
            description, base_item, embedding = process_item_from_image_url(
                item["image_url"], item_name=item.get("name", "")
            )
            save_processed_item(
                item_id=item["id"],
                category=base_item.get("category", "top"),
                description=description,
                base_item=base_item,
                embedding=embedding,
            )
            logger.info(f"Processed: {item['name']} → {base_item.get('category')}")
            return True

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "rate_limit_exceeded" in err_str or "429" in err_str or "Too Many Requests" in err_str
            if is_rate_limit and attempt < max_retries - 1:
                # Parse retry-after from error message if available
                wait = 5 * (attempt + 1)
                m = re.search(r'try again in ([0-9.]+)s', err_str)
                if m:
                    wait = float(m.group(1)) + 1
                logger.warning(f"Rate limit hit for {item['name']}, retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            logger.error(f"Failed to process {item['name']}: {e}")
            save_processing_error(item["id"], str(e))
            return False
    return False


def generate_outfits_for_item(shop_domain: str, item: dict):
    """
    Generate outfits for one processed catalog item and save to DB cache.
    Uses the same full scoring + collage pipeline as the on-demand endpoint.
    """
    try:
        outfits = _run_outfit_generation(shop_domain, item)
        if outfits:
            save_generated_outfits(shop_domain, item["shopify_product_id"], outfits)
            logger.info(f"Cached {len(outfits)} outfits for {item['name']}")
    except Exception as e:
        logger.error(f"generate_outfits_for_item failed for {item.get('name')}: {e}", exc_info=True)

def generate_all_outfits(shop_domain: str):
    """Generate outfits for all processed items in a store that don't have outfits yet."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, shopify_product_id, name, category, image_url, product_url,
                   price, primary_color, secondary_colors, style_tags, season_tags,
                   occasion_tags, material, fit, embedding::text
            FROM shopify_catalog_items
            WHERE shop_domain = %s
              AND processed_at IS NOT NULL
              AND shopify_product_id NOT IN (
                  SELECT DISTINCT shopify_product_id
                  FROM shopify_generated_outfits
                  WHERE shop_domain = %s
              )
            """,
            (shop_domain, shop_domain),
        )
        rows = cur.fetchall()
        cols = ["id", "shopify_product_id", "name", "category", "image_url",
                "product_url", "price", "primary_color", "secondary_colors",
                "style_tags", "season_tags", "occasion_tags", "material", "fit",
                "embedding_text"]
        items = []
        for row in rows:
            d = dict(zip(cols, row))
            emb_text = d.pop("embedding_text", None)
            d["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")] if emb_text else []
            items.append(d)
    finally:
        cur.close()
        conn.close()

    logger.info(f"Generating outfits for {len(items)} items in {shop_domain}")
    for item in items:
        generate_outfits_for_item(shop_domain, item)

    update_store_stats(shop_domain)
    logger.info(f"Outfit generation complete for {shop_domain}")


def process_catalog_batch(shop_domain: str, batch_size: int = 20):
    """
    Two-pass pipeline:
      Pass 1 — run vision+embedding on all unprocessed items (builds full catalog).
      Pass 2 — generate + cache outfits for every item once the catalog is complete.
    Generating after the full catalog is in place lets the scoring pipeline
    pick the best matches across all available products.
    """
    items = get_unprocessed_items(shop_domain, limit=batch_size)
    if not items:
        logger.info(f"No unprocessed items for {shop_domain}")
        update_store_stats(shop_domain)
        # Still try to generate outfits for any items that are processed but not yet cached
        generate_all_outfits(shop_domain)
        return

    logger.info(f"Pass 1: processing {len(items)} items for {shop_domain}")
    success = sum(process_single_item(shop_domain, item) for item in items)
    logger.info(f"Pass 1 complete: {success}/{len(items)} succeeded")
    update_store_stats(shop_domain)

    # Only run outfit generation when there are no more unprocessed items,
    # so the scoring pipeline has the full catalog to work with.
    remaining = get_unprocessed_items(shop_domain, limit=1)
    if not remaining:
        logger.info("Pass 2: full catalog ready — generating outfits for all items")
        generate_all_outfits(shop_domain)
    else:
        logger.info("More items still unprocessed — skipping outfit generation until catalog is complete")
def reprocess_catalog(shop_domain: str):
    """Re-run vision + parser + embedding for ALL items (same as non-Shopify app); refresh categories (e.g. layer) and embeddings; then regenerate outfits."""
    items = get_all_catalog_item_stubs(shop_domain)
    if not items:
        logger.info(f"No catalog items to reprocess for {shop_domain}")
        return
    logger.info(f"Reprocessing {len(items)} items for {shop_domain} (vision + parse + embed)")
    success = sum(1 for item in items if process_single_item(shop_domain, item))
    logger.info(f"Reprocess complete: {success}/{len(items)} succeeded")
    update_store_stats(shop_domain)
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM shopify_generated_outfits WHERE shop_domain = %s", (shop_domain,))
        conn.commit()
    finally:
        cur.close()
        conn.close()
    logger.info("Cleared outfit cache; generating outfits with updated catalog")
    generate_all_outfits(shop_domain)


def full_catalog_sync(shop_domain: str, access_token: str):
    """Full pipeline: fetch all products → insert → process → generate outfits."""
    try:
        # 1. Fetch all products from Shopify
        products = fetch_shopify_products(shop_domain, access_token)
        logger.info(f"Fetched {len(products)} products for {shop_domain}")

        live_ids = {p["shopify_product_id"] for p in products}

        # 2. Remove products that no longer exist in Shopify
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT shopify_product_id FROM shopify_catalog_items WHERE shop_domain = %s",
                (shop_domain,),
            )
            db_ids = {row[0] for row in cur.fetchall()}
            deleted_ids = db_ids - live_ids
            if deleted_ids:
                logger.info(f"Removing {len(deleted_ids)} deleted products from catalog: {deleted_ids}")
                cur.execute(
                    "DELETE FROM shopify_catalog_items WHERE shop_domain = %s AND shopify_product_id = ANY(%s)",
                    (shop_domain, list(deleted_ids)),
                )
                cur.execute(
                    "DELETE FROM shopify_generated_outfits WHERE shop_domain = %s AND shopify_product_id = ANY(%s)",
                    (shop_domain, list(deleted_ids)),
                )
                # Also purge outfits whose outfit_items reference a deleted product
                for deleted_id in deleted_ids:
                    cur.execute(
                        """
                        DELETE FROM shopify_generated_outfits
                        WHERE shop_domain = %s
                          AND EXISTS (
                              SELECT 1 FROM jsonb_array_elements(outfit_items) AS item
                              WHERE item->>'shopify_product_id' = %s
                          )
                        """,
                        (shop_domain, deleted_id),
                    )
                conn.commit()
        finally:
            cur.close()
            conn.close()

        # 3. Upsert all products into DB (without processing yet)
        for product in products:
            upsert_shopify_catalog_item(shop_domain, product)

        # 4. Process in batches of 20 until done
        while True:
            unprocessed = get_unprocessed_items(shop_domain, limit=1)
            if not unprocessed:
                break
            process_catalog_batch(shop_domain, batch_size=20)

        update_store_stats(shop_domain)
        logger.info(f"Full sync complete for {shop_domain}")

        # Generate outfits for all processed items
        generate_all_outfits(shop_domain)

    except Exception as e:
        logger.error(f"Full sync failed for {shop_domain}: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/shopify/health")
async def health():
    return {"status": "ok"}


@app.post("/shopify/install")
async def install(req: InstallRequest, background_tasks: BackgroundTasks):
    """
    Called after OAuth completes. Stores access token and kicks off catalog sync.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO shopify_stores (shop_domain, access_token, scope)
            VALUES (%s, %s, %s)
            ON CONFLICT (shop_domain) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                scope = EXCLUDED.scope,
                uninstalled_at = NULL
            """,
            (req.shop_domain, req.access_token, req.scope),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    # Kick off catalog sync in background
    background_tasks.add_task(full_catalog_sync, req.shop_domain, req.access_token)

    return {"status": "installed", "shop": req.shop_domain}


@app.post("/shopify/catalog/sync")
async def sync_catalog(req: SyncRequest, background_tasks: BackgroundTasks):
    """Manually trigger a full catalog re-sync."""
    background_tasks.add_task(full_catalog_sync, req.shop_domain, req.access_token)
    return {"status": "sync_started", "shop": req.shop_domain}


@app.post("/shopify/outfits/generate")
async def generate_outfits_endpoint(shop_domain: str, background_tasks: BackgroundTasks):
    """Manually trigger outfit generation for all processed items."""
    background_tasks.add_task(generate_all_outfits, shop_domain)
    return {"status": "generation_started", "shop": shop_domain}


@app.post("/shopify/outfits/regenerate-product")
async def regenerate_product_outfits(shop_domain: str, shopify_product_id: str, background_tasks: BackgroundTasks):
    """Delete and re-generate outfits for a single product."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM shopify_generated_outfits WHERE shop_domain = %s AND shopify_product_id = %s",
            (shop_domain, shopify_product_id),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    def regen_one():
        conn2 = get_db()
        cur2 = conn2.cursor()
        try:
            cur2.execute(
                """
                SELECT id, shopify_product_id, name, category, image_url, product_url,
                       price, primary_color, secondary_colors, style_tags, season_tags,
                       occasion_tags, material, fit, embedding::text
                FROM shopify_catalog_items
                WHERE shop_domain = %s AND shopify_product_id = %s AND processed_at IS NOT NULL
                """,
                (shop_domain, shopify_product_id),
            )
            row = cur2.fetchone()
            if row:
                cols = ["id","shopify_product_id","name","category","image_url","product_url",
                        "price","primary_color","secondary_colors","style_tags","season_tags",
                        "occasion_tags","material","fit","embedding_text"]
                d = dict(zip(cols, row))
                emb_text = d.pop("embedding_text", None)
                d["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")] if emb_text else []
                generate_outfits_for_item(shop_domain, d)
                update_store_stats(shop_domain)
        finally:
            cur2.close()
            conn2.close()

    background_tasks.add_task(regen_one)
    return {"status": "regeneration_started", "product_id": shopify_product_id}




@app.post("/shopify/catalog/resync")
async def resync_catalog(shop_domain: str, background_tasks: BackgroundTasks):
    """Trigger a full re-sync using the stored access token (no token param needed)."""
    store = get_store(shop_domain)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found or not installed")
    background_tasks.add_task(full_catalog_sync, store["shop_domain"], store["access_token"])
    return {"status": "resync_started", "shop": shop_domain}


@app.post("/shopify/catalog/reprocess")
async def reprocess_catalog_endpoint(shop_domain: str, background_tasks: BackgroundTasks):
    """Re-parse all products (vision + parse + embed, same as main app); refresh categories and regenerate outfits."""
    store = get_store(shop_domain)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found or not installed")
    background_tasks.add_task(reprocess_catalog, shop_domain)
    return {"status": "reprocess_started", "shop": shop_domain}


@app.delete("/shopify/catalog/item")
async def delete_catalog_item(shop_domain: str, item_id: int, background_tasks: BackgroundTasks):
    """Remove a catalog item by internal ID and purge any outfits that reference it."""
    conn = get_db()
    cur = conn.cursor()
    try:
        # Get its shopify_product_id before deleting
        cur.execute(
            "SELECT shopify_product_id FROM shopify_catalog_items WHERE id = %s AND shop_domain = %s",
            (item_id, shop_domain),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        shopify_pid = row[0]

        # Delete the catalog item
        cur.execute("DELETE FROM shopify_catalog_items WHERE id = %s AND shop_domain = %s", (item_id, shop_domain))

        # Delete outfits where this item is the anchor
        cur.execute(
            "DELETE FROM shopify_generated_outfits WHERE shop_domain = %s AND shopify_product_id = %s",
            (shop_domain, shopify_pid),
        )

        # Delete outfits whose outfit_items JSON references this catalog item id
        cur.execute(
            """
            DELETE FROM shopify_generated_outfits
            WHERE shop_domain = %s
              AND EXISTS (
                  SELECT 1 FROM jsonb_array_elements(outfit_items) AS it
                  WHERE (it->>'id')::int = %s
              )
            """,
            (shop_domain, item_id),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    store = get_store(shop_domain)
    if store:
        background_tasks.add_task(generate_all_outfits, shop_domain)
    update_store_stats(shop_domain)
    return {"status": "deleted", "item_id": item_id, "regenerating": store is not None}

@app.get("/shopify/catalog/status")
async def catalog_status(shop_domain: str):
    """Check sync status for a store."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT product_count, outfit_count, catalog_synced_at,
                   (SELECT COUNT(*) FROM shopify_catalog_items
                    WHERE shop_domain = %s AND processed_at IS NULL AND processing_error IS NULL) AS pending
            FROM shopify_stores WHERE shop_domain = %s
            """,
            (shop_domain, shop_domain),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Store not found")
        return {
            "product_count": row[0],
            "outfit_count": row[1],
            "synced_at": row[2].isoformat() if row[2] else None,
            "pending_processing": row[3],
        }
    finally:
        cur.close()
        conn.close()


def _generate_and_upload_collage(outfit_items: list[dict], base_category: str, direction: str, product_key: str) -> str | None:
    """Generate a PIL collage for an outfit and upload to Cloudinary. Returns secure_url or None."""
    try:
        # Build items list for collage (all items including anchor)
        items_for_collage = [{"slot": i["slot"], "image_url": i["image_url"]} for i in outfit_items]
        anchor = next((i for i in outfit_items if i.get("is_anchor")), None)
        base_item_for_collage = {
            "category": base_category,
            "image_url": anchor["image_url"] if anchor else "",
        }

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            collage_path = Path(tmp.name)

        create_grid_collage(items_for_collage, collage_path, base_item=base_item_for_collage)

        safe_key = product_key.replace("/", "_").replace(":", "_")
        upload_result = cloudinary.uploader.upload(
            str(collage_path),
            folder="loom_shopify_collages",
            public_id=f"{safe_key}_{direction.lower()}",
            overwrite=True,
        )
        collage_path.unlink(missing_ok=True)
        return upload_result.get("secure_url")
    except Exception as e:
        logger.warning(f"Collage generation failed ({direction}): {e}")
        return None


def _fetch_catalog_item(shop_domain: str, shopify_product_id: str) -> dict | None:
    """Fetch a processed catalog item dict (with embedding) from the DB."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, shopify_product_id, name, category, image_url, product_url,
                   price, primary_color, secondary_colors, style_tags, season_tags,
                   occasion_tags, material, fit, embedding::text
            FROM shopify_catalog_items
            WHERE shop_domain = %s AND shopify_product_id = %s AND processed_at IS NOT NULL
            """,
            (shop_domain, shopify_product_id),
        )
        row = cur.fetchone()
    finally:
        cur.close()
        conn.close()
    if not row:
        return None
    cols = ["id", "shopify_product_id", "name", "category", "image_url", "product_url",
            "price", "primary_color", "secondary_colors", "style_tags", "season_tags",
            "occasion_tags", "material", "fit", "embedding_text"]
    item = dict(zip(cols, row))
    emb_text = item.pop("embedding_text", None)
    item["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")] if emb_text else []
    return item


def _run_outfit_generation(shop_domain: str, item: dict) -> list[dict]:
    """Same outfit pipeline as non-Shopify app (services/outfit_generator); add collage for Shopify."""
    outfits = run_outfit_generation(item, shop_domain=shop_domain)
    base_category = item.get("category", "top")
    result = []
    for o in outfits:
        collage_url = _generate_and_upload_collage(
            o["outfit_items"], base_category, o["direction"], item.get("shopify_product_id", "")
        )
        result.append({
            "direction": o["direction"],
            "explanation": o["explanation"],
            "outfit_items": o["outfit_items"],
            "collage_url": collage_url,
        })
    return result

def _normalize_product_id(product_id: str) -> str:
    """Accept numeric id or gid://shopify/Product/123; return full GID for DB lookup."""
    if not product_id or not product_id.strip():
        return product_id
    s = product_id.strip()
    if s.isdigit():
        return f"gid://shopify/Product/{s}"
    if s.startswith("gid://shopify/Product/"):
        return s
    return s


@app.get("/shopify/outfits")
async def get_outfits(shop_domain: str, product_id: str):
    """
    Returns outfits for a product page.
    Checks DB cache first; generates + caches on miss.
    """
    product_id = _normalize_product_id(product_id)
    # Check cache
    cached = get_generated_outfits(shop_domain, product_id)
    if cached:
        return {"outfits": cached, "status": "ready"}

    # Cache miss — generate on-demand
    item = await asyncio.to_thread(_fetch_catalog_item, shop_domain, product_id)
    if not item:
        return {"outfits": [], "status": "not_ready"}

    outfits = await asyncio.to_thread(_run_outfit_generation, shop_domain, item)
    if not outfits:
        return {"outfits": [], "status": "not_ready"}

    # Save to cache for next visit
    await asyncio.to_thread(save_generated_outfits, shop_domain, product_id, outfits)
    return {"outfits": outfits, "status": "ready"}


# Which anchor categories should have outfits invalidated when a new item of a given
# category is added. E.g. a new "layer" can appear in outfits for tops/bottoms/shoes/dresses.
_AFFECTS_OUTFITS_FOR = {
    "top":       ["bottom", "shoes", "dress"],
    "bottom":    ["top", "shoes"],
    "shoes":     ["top", "bottom", "dress"],
    "layer":     ["top", "bottom", "shoes", "dress"],
    "accessory": ["top", "bottom", "shoes", "dress", "layer"],
    "dress":     ["shoes"],
}


def _invalidate_and_regen_affected(shop_domain: str, new_category: str):
    """
    Delete cached outfits for anchor products whose outfits *could* include the
    new item (i.e. the new category appears as a component slot for those anchors).
    Then regenerate only those outfits.
    """
    affected_anchor_cats = _AFFECTS_OUTFITS_FOR.get(new_category, [])
    if not affected_anchor_cats:
        logger.info(f"New {new_category}: no other outfit caches to invalidate")
        return

    conn = get_db()
    cur = conn.cursor()
    try:
        # Find anchor shopify_product_ids in the affected categories
        cur.execute(
            """
            SELECT shopify_product_id FROM shopify_catalog_items
            WHERE shop_domain = %s AND category = ANY(%s) AND processed_at IS NOT NULL
            """,
            (shop_domain, affected_anchor_cats),
        )
        affected_ids = [r[0] for r in cur.fetchall()]
        if not affected_ids:
            logger.info(f"No affected anchors for new {new_category}")
            return

        # Delete their cached outfits so they regenerate with the new item available
        cur.execute(
            """
            DELETE FROM shopify_generated_outfits
            WHERE shop_domain = %s AND shopify_product_id = ANY(%s)
            """,
            (shop_domain, affected_ids),
        )
        conn.commit()
        logger.info(
            f"Invalidated outfits for {len(affected_ids)} {affected_anchor_cats} "
            f"products (new {new_category} added)"
        )
    finally:
        cur.close()
        conn.close()

    # Regenerate only the invalidated outfits
    generate_all_outfits(shop_domain)


def _process_and_generate(shop_domain: str, item_stub: dict):
    """
    Process a new product then do smart incremental outfit invalidation:
    1. Generate outfits for the new product itself.
    2. Invalidate cached outfits for existing products that could NOW use the
       new item as a component (by category), then regenerate only those.
    """
    success = process_single_item(shop_domain, item_stub)
    if success:
        shopify_product_id = item_stub.get("shopify_product_id")
        if shopify_product_id:
            full_item = _fetch_catalog_item(shop_domain, shopify_product_id)
            if full_item:
                # Step 1: generate outfits for the new product
                generate_outfits_for_item(shop_domain, full_item)
                logger.info(f"Outfits generated for new product: {full_item.get('name')} ({full_item.get('category')})")
                # Step 2: invalidate + regenerate outfits for products that benefit from this new item
                _invalidate_and_regen_affected(shop_domain, full_item.get("category", ""))


@app.post("/shopify/webhooks/product_created")
async def webhook_product_created(
    request: Request,
    background_tasks: BackgroundTasks,
    x_shopify_shop_domain: str = Header(None),
    x_shopify_hmac_sha256: str = Header(None),
):
    """When a merchant adds a new product, process it automatically."""
    body = await request.body()

    if not verify_shopify_webhook(body, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    import json
    product_data = json.loads(body)
    shop_domain = x_shopify_shop_domain

    store = get_store(shop_domain)
    if not store:
        return {"status": "store_not_found"}

    images = product_data.get("images", [])
    if not images:
        return {"status": "no_image"}

    product = {
        "shopify_product_id": f"gid://shopify/Product/{product_data['id']}",
        "shopify_variant_id": None,
        "name": product_data["title"],
        "product_type": product_data.get("product_type", ""),
        "tags": product_data.get("tags", "").split(", ") if product_data.get("tags") else [],
        "image_url": images[0]["src"],
        "product_url": None,
        "price": None,
    }

    item_id = upsert_shopify_catalog_item(shop_domain, product)
    background_tasks.add_task(
        _process_and_generate,
        shop_domain,
        {
            "id": item_id,
            "name": product["name"],
            "image_url": product["image_url"],
            "shopify_product_id": product["shopify_product_id"],
        },
    )

    return {"status": "queued"}


@app.post("/shopify/webhooks/app_uninstalled")
async def webhook_app_uninstalled(
    request: Request,
    x_shopify_shop_domain: str = Header(None),
    x_shopify_hmac_sha256: str = Header(None),
):
    """Mark store as uninstalled."""
    body = await request.body()

    if not verify_shopify_webhook(body, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE shopify_stores SET uninstalled_at = NOW() WHERE shop_domain = %s",
            (x_shopify_shop_domain,),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

    return {"status": "uninstalled"}
