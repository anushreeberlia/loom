"""
Unified item processing: vision analysis + segmentation + embedding (legacy + multi-head).

Used by:
- Shopify app (shopify_catalog_items)
- Non-Shopify app (catalog_items, user_closet_items)

Pipeline:
1. Vision analysis (Florence) on ORIGINAL image for metadata extraction
2. Segmentation (U2-Net) to produce clean garment cutout
3. FashionCLIP embedding on SEGMENTED image (legacy, 512-dim)
4. DINOv2 backbone + multi-head projections (5 x 128-dim)

Call process_item_from_image() to get (description, base_item, embedding, multihead_embeddings);
the 4th return value is a dict of head_name -> 128-dim numpy array (or None if DINOv2 unavailable).
"""

import logging
import httpx
import numpy as np

from services.vision import analyze_image
from services.embedding import embed_item_blended
from services.segmentation import segment_for_embedding

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


def _compute_multihead(segmented_bytes: bytes) -> tuple[np.ndarray | None, dict[str, np.ndarray] | None]:
    """
    Run DINOv2 + multi-head projections on segmented image.
    Returns (backbone_embedding, head_embeddings) or (None, None) on failure.
    """
    try:
        from services.dinov2 import embed_image as dinov2_embed
        from services.multihead import compute_multihead_embeddings

        backbone = dinov2_embed(segmented_bytes)
        heads = compute_multihead_embeddings(backbone)
        return backbone, heads
    except Exception as e:
        logger.warning("DINOv2 multi-head failed (non-fatal, legacy embedding still computed): %s", e)
        return None, None


def process_item_from_image(image_bytes: bytes, item_name: str = "", backend: str = None) -> tuple:
    """
    Vision analysis + segmented embedding pipeline.

    1. Run vision on the ORIGINAL image (full context helps metadata extraction)
    2. Segment the garment (remove background)
    3. Embed the SEGMENTED cutout with FashionCLIP (legacy)
    4. Embed the SEGMENTED cutout with DINOv2 + multi-head projections

    Returns (description, base_item, embedding, multihead_result) where multihead_result is:
        {"backbone": np.ndarray(768), "style": np.ndarray(128), ...} or None
    """
    base_item = analyze_image(image_bytes, backend=backend)

    segmented_bytes = segment_for_embedding(image_bytes)

    embedding = embed_item_blended(segmented_bytes, base_item)

    backbone, heads = _compute_multihead(segmented_bytes)
    multihead_result = None
    if backbone is not None and heads is not None:
        multihead_result = {"backbone": backbone, **heads}

    description = _build_description(base_item)
    if item_name:
        logger.info(f"Processed: {item_name} -> {base_item.get('category')}")
    return description, base_item, embedding, multihead_result


def process_item_from_image_url(image_url: str, item_name: str = "", backend: str = None) -> tuple:
    """Download image from URL, then run same pipeline. Returns (description, base_item, embedding, multihead_result)."""
    response = httpx.get(image_url, timeout=20.0)
    if response.status_code != 200:
        raise Exception(f"Image download failed: {response.status_code}")
    return process_item_from_image(response.content, item_name=item_name, backend=backend)
