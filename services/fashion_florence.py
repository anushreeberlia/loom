"""
Fashion Florence vision service — calls the Fashion Florence HF Space API.

The model runs on a free HuggingFace Space (GPU), keeping Railway lean.
This module is just an HTTP client + schema expansion logic.

Output from API: {category, primary_color, material, style_tags}
Missing fields (secondary_colors, fit, occasion_tags, season_tags) are derived
from model output using rule-based mappings.
"""

import io
import os
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FLORENCE_API_URL = os.getenv("FLORENCE_API_URL", "").rstrip("/")

MAX_RETRIES = 2
FIRST_TIMEOUT = 120.0  # generous for cold starts
RETRY_TIMEOUT = 90.0


def _call_florence_api(image_bytes: bytes) -> dict:
    """POST image to the HF Space /analyze endpoint with retry for cold starts."""
    if not FLORENCE_API_URL:
        raise ValueError("FLORENCE_API_URL not set")

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        timeout = FIRST_TIMEOUT if attempt == 0 else RETRY_TIMEOUT
        try:
            response = httpx.post(
                f"{FLORENCE_API_URL}/analyze",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()

            # 503 = Space is waking up from sleep
            if response.status_code == 503 and attempt < MAX_RETRIES:
                logger.warning(
                    "Florence API cold start (503), retry %d/%d...",
                    attempt + 1, MAX_RETRIES,
                )
                continue

            response.raise_for_status()

        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Florence API timeout, retry %d/%d...",
                    attempt + 1, MAX_RETRIES,
                )
                continue
            raise

        except httpx.HTTPStatusError as e:
            last_error = e
            raise

    raise last_error or RuntimeError("Florence API call failed")


# ── Schema expansion: derive missing fields ──────────────────────────────────

STYLE_TO_OCCASION = {
    "sporty": ["gym", "workout"],
    "athletic": ["gym", "workout"],
    "activewear": ["gym", "workout"],
    "casual": ["everyday", "casual"],
    "classic": ["work", "everyday"],
    "workwear": ["work"],
    "elegant": ["dinner", "party", "formal"],
    "glamorous": ["party", "formal"],
    "sexy": ["going-out", "clubbing", "date", "party"],
    "chic": ["dinner", "brunch"],
    "romantic": ["date", "dinner"],
    "bohemian": ["vacation", "casual"],
    "streetwear": ["everyday", "casual"],
    "preppy": ["work", "everyday"],
    "edgy": ["going-out"],
    "vintage": ["everyday", "casual"],
    "statement": ["party", "going-out"],
    "minimalist": ["everyday", "work"],
    "trendy": ["everyday", "casual"],
}

CATEGORY_OCCASION_DEFAULTS = {
    "dress": ["dinner"],
    "layer": ["everyday"],
    "shoes": ["everyday"],
    "accessory": ["everyday"],
    "top": ["everyday"],
    "bottom": ["everyday"],
}

MATERIAL_TO_SEASON = {
    "wool": ["fall", "winter"],
    "cashmere": ["fall", "winter"],
    "fleece": ["fall", "winter"],
    "flannel": ["fall", "winter"],
    "velvet": ["fall", "winter"],
    "corduroy": ["fall", "winter"],
    "tweed": ["fall", "winter"],
    "knit": ["fall", "winter"],
    "leather": ["fall", "winter"],
    "suede": ["fall"],
    "linen": ["spring", "summer"],
    "chiffon": ["spring", "summer"],
    "lace": ["spring", "summer"],
    "satin": ["spring", "summer"],
    "silk": ["spring", "summer"],
    "organza": ["spring", "summer"],
}

STYLE_TO_FIT = {
    "sexy": "bodycon",
    "glamorous": "fitted",
    "elegant": "fitted",
    "sporty": "fitted",
    "athletic": "fitted",
    "activewear": "fitted",
    "streetwear": "relaxed",
    "bohemian": "relaxed",
    "casual": "relaxed",
    "classic": "straight",
    "workwear": "straight",
    "preppy": "straight",
    "minimalist": "straight",
    "edgy": "slim",
    "chic": "slim",
}


def _derive_occasion_tags(category: str, style_tags: list[str]) -> list[str]:
    occasions = set()
    for style in style_tags:
        for occ in STYLE_TO_OCCASION.get(style, []):
            occasions.add(occ)
    if not occasions:
        for occ in CATEGORY_OCCASION_DEFAULTS.get(category, ["everyday"]):
            occasions.add(occ)
    return sorted(occasions)


def _derive_season_tags(material: str | None) -> list[str]:
    if not material:
        return ["all_season"]
    return MATERIAL_TO_SEASON.get(material.lower().strip(), ["all_season"])


def _derive_fit(style_tags: list[str]) -> str:
    for style in style_tags:
        if style in STYLE_TO_FIT:
            return STYLE_TO_FIT[style]
    return "unknown"


def expand_florence_output(raw: dict) -> dict:
    """Expand Florence's 4-field output to the full Loom 8-field schema."""
    category = raw.get("category", "top")
    style_tags = raw.get("style_tags", [])
    material = raw.get("material")

    return {
        "category": category,
        "primary_color": raw.get("primary_color", "unknown"),
        "secondary_colors": [],
        "material": material,
        "fit": _derive_fit(style_tags),
        "style_tags": style_tags,
        "occasion_tags": _derive_occasion_tags(category, style_tags),
        "season_tags": _derive_season_tags(material),
    }


def analyze_image(image_bytes: bytes) -> dict:
    """
    Analyze a clothing image via the Fashion Florence HF Space.
    Returns full Loom schema dict after expansion + validation.
    """
    from services.tagging import validate_tags

    raw = _call_florence_api(image_bytes)
    logger.info("Florence raw output: %s", raw)
    raw_color = raw.get("primary_color", "unknown")
    expanded = expand_florence_output(raw)
    validated = validate_tags(expanded, include_category=True)

    # Preserve Florence's color even if it's outside the standard palette
    if validated["primary_color"] == "unknown" and raw_color.lower().strip() != "unknown":
        validated["primary_color"] = raw_color.lower().strip()

    logger.info("Florence final output: %s", validated)
    return validated
