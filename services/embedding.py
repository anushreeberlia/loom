"""
Embedding service — delegates to FashionCLIP for all vector operations.

Image items are embedded via the CLIP image encoder (512-dim).
Text queries are embedded via the CLIP text encoder (512-dim).
Both share the same vector space, enabling cross-modal search.
"""

import logging
from services.fashion_clip import embed_text, embed_image, get_embedding_dim

logger = logging.getLogger(__name__)

EMBEDDING_DIM = get_embedding_dim()  # 512


def build_embedding_text(item: dict) -> str:
    """Build deterministic text string for embedding from BaseItem."""
    parts = []

    if item.get("category"):
        parts.append(f"Category: {item['category']}")

    if item.get("primary_color"):
        parts.append(f"Color: {item['primary_color']}")

    if item.get("fit") and item["fit"] != "unknown":
        parts.append(f"Fit: {item['fit']}")

    if item.get("material"):
        parts.append(f"Material: {item['material']}")

    if item.get("style_tags"):
        parts.append(f"Style: {', '.join(item['style_tags'])}")

    if item.get("occasion_tags"):
        parts.append(f"Occasion: {', '.join(item['occasion_tags'])}")

    if item.get("season_tags"):
        parts.append(f"Season: {', '.join(item['season_tags'])}")

    return ". ".join(parts) + "." if parts else ""


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from FashionCLIP text encoder."""
    return embed_text(text)


def embed_base_item(base_item: dict) -> list[float]:
    """Generate text embedding for a BaseItem dict (used as fallback when no image available)."""
    text = build_embedding_text(base_item)
    return get_embedding(text)


def embed_item_image(image_bytes: bytes) -> list[float]:
    """Generate image embedding directly from clothing photo (preferred over text embedding)."""
    return embed_image(image_bytes)


IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3


def embed_item_blended(image_bytes: bytes, base_item: dict) -> list[float]:
    """
    Blend image embedding with text metadata embedding.
    
    Image alone can't distinguish athletic from casual tops (similar on hangers).
    Adding text ("sporty fitted polyester") gives the embedding semantic identity
    that differentiates items at query time.
    """
    import numpy as np

    img_emb = np.array(embed_image(image_bytes))

    text = build_embedding_text(base_item)
    if not text:
        return img_emb.tolist()

    txt_emb = np.array(embed_text(text))

    blended = IMAGE_WEIGHT * img_emb + TEXT_WEIGHT * txt_emb
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.tolist()


def blend_existing_embedding(image_embedding: list[float], base_item: dict) -> list[float]:
    """
    Blend a pre-computed image embedding with text metadata.
    Used for migration of existing items (no image bytes needed).
    """
    import numpy as np

    img_emb = np.array(image_embedding)

    text = build_embedding_text(base_item)
    if not text:
        return image_embedding

    txt_emb = np.array(embed_text(text))

    blended = IMAGE_WEIGHT * img_emb + TEXT_WEIGHT * txt_emb
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.tolist()
