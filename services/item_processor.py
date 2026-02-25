"""
Unified item processing: same vision + parser + embedding pipeline for all apps.

Used by:
- Shopify app (shopify_catalog_items)
- Non-Shopify app (catalog_items, user_closet_items)

Call process_item_from_image() to get (description, base_item, embedding);
then persist to your store (Shopify table, closet table, etc.).
"""

import logging
import httpx

from services.vision import describe_image
from services.parser import parse_description
from services.embedding import embed_item_image, embed_base_item

logger = logging.getLogger(__name__)


def process_item_from_image(image_bytes: bytes, item_name: str = "") -> tuple:
    """
    Run the vision -> parse -> embed pipeline.
    Embedding uses FashionCLIP image encoder directly (no text intermediary).
    Returns (description, base_item, embedding).
    """
    description = describe_image(image_bytes)
    base_item = parse_description(description)
    embedding = embed_item_image(image_bytes)
    if item_name:
        logger.info(f"Processed: {item_name} -> {base_item.get('category')}")
    return description, base_item, embedding


def process_item_from_image_url(image_url: str, item_name: str = "") -> tuple:
    """Download image from URL, then run same pipeline. Returns (description, base_item, embedding)."""
    response = httpx.get(image_url, timeout=20.0)
    if response.status_code != 200:
        raise Exception(f"Image download failed: {response.status_code}")
    return process_item_from_image(response.content, item_name=item_name)
