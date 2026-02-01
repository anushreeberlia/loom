"""
Shared tagging constants and validation logic.
Used by both catalog tagging and user upload parsing.
"""

# Allowed values for validation
ALLOWED_COLORS = {"black", "white", "gray", "beige", "brown", "blue", "navy", "green", "yellow", "orange", "red", "pink", "purple", "metallic", "multi", "unknown"}
ALLOWED_FIT = {"fitted", "slim", "straight", "relaxed", "oversized", "wide", "cropped", "loose", "unknown"}
ALLOWED_STYLE = {"minimalist", "classic", "edgy", "romantic", "sporty", "bohemian", "streetwear", "preppy", "elegant", "casual", "chic", "vintage", "statement", "workwear"}
ALLOWED_OCCASION = {"everyday", "casual", "work", "dinner", "party", "formal", "vacation", "lounge", "wedding_guest"}
ALLOWED_SEASON = {"spring", "summer", "fall", "winter", "all_season"}
ALLOWED_CATEGORY = {"top", "bottom", "dress", "layer", "shoes", "accessory"}

# Color mapping for shades → base colors
COLOR_MAP = {
    "magenta": "pink", "fuchsia": "pink", "rose": "pink", "coral": "pink", "salmon": "pink",
    "violet": "purple", "lavender": "purple", "plum": "purple", "mauve": "purple",
    "teal": "green", "olive": "green", "mint": "green", "emerald": "green", "khaki": "green",
    "burgundy": "red", "maroon": "red", "crimson": "red", "wine": "red",
    "tan": "beige", "cream": "beige", "ivory": "beige", "sand": "beige", "nude": "beige", "camel": "beige",
    "charcoal": "gray", "silver": "gray", "grey": "gray",
    "gold": "metallic", "bronze": "metallic", "copper": "metallic",
    "indigo": "blue", "cobalt": "blue", "turquoise": "blue", "aqua": "blue", "sky": "blue", "denim": "blue",
    "mustard": "yellow", "lemon": "yellow",
    "rust": "orange", "peach": "orange", "terracotta": "orange",
    "coffee": "brown", "chocolate": "brown", "espresso": "brown", "mocha": "brown",
    "off-white": "white", "offwhite": "white",
}


def normalize_color(color: str) -> str:
    """Map shade to base color, or return unknown if not recognized."""
    if not color:
        return "unknown"
    color = color.lower().strip()
    if color in ALLOWED_COLORS:
        return color
    if color in COLOR_MAP:
        return COLOR_MAP[color]
    return "unknown"


def validate_tags(tags: dict, include_category: bool = False) -> dict:
    """
    Validate tags against allowed values, fix what we can.
    
    Args:
        tags: Dict with tag values
        include_category: If True, validate category field (for user uploads)
    """
    # Category (only for user uploads)
    if include_category:
        category = tags.get("category", "top").lower()
        if category not in ALLOWED_CATEGORY:
            category = "top"
        tags["category"] = category
    
    # Primary color: normalize to allowed palette
    primary = tags.get("primary_color", "unknown")
    tags["primary_color"] = normalize_color(primary)
    
    # Secondary colors: normalize each, filter out unknowns
    secondary = tags.get("secondary_colors", [])
    normalized_secondary = [normalize_color(c) for c in secondary]
    tags["secondary_colors"] = [c for c in normalized_secondary if c != "unknown"]
    
    # Fit: must be in allowed set, default to unknown
    fit = tags.get("fit", "unknown")
    if isinstance(fit, str):
        fit = fit.lower()
    if fit not in ALLOWED_FIT:
        fit = "unknown"
    tags["fit"] = fit
    
    # Style tags: filter to allowed values only
    style = tags.get("style_tags", [])
    tags["style_tags"] = [s.lower() for s in style if s.lower() in ALLOWED_STYLE]
    
    # Occasion tags: filter to allowed values only
    occasion = tags.get("occasion_tags", [])
    tags["occasion_tags"] = [o.lower() for o in occasion if o.lower() in ALLOWED_OCCASION]
    
    # Season tags: fix all-season → all_season, enforce exclusivity
    season = tags.get("season_tags", [])
    # Normalize: replace hyphens with underscores
    season = [s.lower().replace("-", "_") for s in season]
    # Filter to allowed values
    season = [s for s in season if s in ALLOWED_SEASON]
    # If all_season is present, use only that
    if "all_season" in season:
        season = ["all_season"]
    tags["season_tags"] = season if season else ["all_season"]
    
    # Material: convert empty string to null
    material = tags.get("material")
    if material == "" or material == "unknown":
        tags["material"] = None
    
    return tags

