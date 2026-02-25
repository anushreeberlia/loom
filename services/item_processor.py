"""
Unified item processing: single vision call + FashionCLIP embedding.

Used by:
- Shopify app (shopify_catalog_items)
- Non-Shopify app (catalog_items, user_closet_items)

Call process_item_from_image() to get (description, base_item, embedding);
then persist to your store (Shopify table, closet table, etc.).
"""

import logging
import httpx

from services.vision import analyze_image
from services.embedding import embed_item_image

logger = logging.getLogger(__name__)


def _build_description(base_item: dict) -> str:
    """Build a text description from structured tags (for backward-compatible storage)."""
    parts = []
    if base_item.get("primary_color"):
        parts.append(base_item["primary_color"])
    if base_item.get("material"):
        parts.append(base_item["material"])
    if base_item.get("fit") and base_item["fit"] != "unknown":
        parts.append(base_item["fit"])
    if base_item.get("category"):
        parts.append(base_item["category"])
    if base_item.get("style_tags"):
        parts.append(f"({', '.join(base_item['style_tags'][:3])})")
    return " ".join(parts) if parts else "clothing item"


def process_item_from_image(image_bytes: bytes, item_name: str = "") -> tuple:
    """
    Single-call vision analysis + FashionCLIP image embedding.
    Returns (description, base_item, embedding) for backward compatibility.
    """
    base_item = analyze_image(image_bytes)
    embedding = embed_item_image(image_bytes)
    description = _build_description(base_item)
    if item_name:
        logger.info(f"Processed: {item_name} -> {base_item.get('category')}")
    return description, base_item, embedding


def process_item_from_image_url(image_url: str, item_name: str = "") -> tuple:
    """Download image from URL, then run same pipeline. Returns (description, base_item, embedding)."""
    response = httpx.get(image_url, timeout=20.0)
    if response.status_code != 200:
        raise Exception(f"Image download failed: {response.status_code}")
    return process_item_from_image(response.content, item_name=item_name)
