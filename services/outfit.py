"""
Outfit structure definitions and assembly logic.
"""
import numpy as np

# Neutral colors that go with everything
NEUTRALS = {"black", "white", "gray", "beige", "brown", "navy", "metallic"}

# Color complements for contrast
COMPLEMENTS = {
    "purple": {"white", "black", "gray", "beige", "navy", "green", "yellow", "metallic"},
    "pink": {"white", "black", "gray", "navy", "beige", "green", "metallic"},
    "red": {"white", "black", "gray", "beige", "navy", "metallic"},
    "blue": {"white", "black", "beige", "brown", "orange", "metallic"},
    "green": {"white", "black", "beige", "brown", "pink", "metallic"},
    "yellow": {"white", "black", "gray", "navy", "purple", "metallic"},
    "orange": {"white", "black", "navy", "blue", "beige", "metallic"},
    "multi": {"white", "black", "gray", "beige", "metallic"},
}

# Color families for within-outfit harmony (Fix 3)
COLOR_FAMILIES = {
    "warm": {"red", "orange", "yellow", "brown", "beige", "coral", "rust", "burgundy"},
    "cool": {"blue", "green", "purple", "navy", "teal", "mint", "lavender"},
    "neutral": {"black", "white", "gray", "beige", "navy", "metallic", "brown"},
}

# Slot-specific color preferences (what colors work best per slot)
SLOT_COLOR_PREFS = {
    "shoes": {"black", "white", "beige", "gray", "brown", "metallic", "navy"},
    "accessory": {"black", "beige", "brown", "white", "metallic", "navy"},
    "bottom": None,  # Use direction-based logic
    "top": None,
    "layer": {"black", "white", "gray", "beige", "navy", "brown"},
}

# ── Formality: continuous scoring (replaces 3-bucket system) ──────────────────

GARMENT_TYPE_BASE_SCORES = [
    ({"gown", "tuxedo", "suit", "stiletto", "cocktail dress", "evening dress"}, 5),
    ({"blouse", "chinos", "trousers", "dress", "blazer", "heel", "pump", "wedge", "oxford"}, 4),
    ({"jeans", "cardigan", "boot", "flat", "loafer", "skirt", "sweater", "shirt"}, 3),
    ({"t-shirt", "tee", "tank", "hoodie", "sweatshirt", "sneaker", "sandal", "cami", "camisole"}, 2),
    ({"sports bra", "gym shorts", "flip flop", "jogger", "legging", "athletic"}, 1),
]

TEXTURE_FORMALITY_NUDGE = {
    "satin": 0.4, "silk": 0.4, "patent": 0.4, "sequin": 0.5, "metallic": 0.4,
    "chiffon": 0.4, "sheer": 0.4, "mesh": 0.3, "lace": 0.3, "organza": 0.4,
    "velvet": 0.3, "crepe": 0.2,
    "leather": 0.1, "suede": 0.1,
    "cotton": 0.0, "polyester": 0.0, "denim": 0.0, "nylon": 0.0, "blend": 0.0,
    "jersey": 0.0, "rayon": 0.0, "viscose": 0.0, "linen": 0.0,
    "ribbed": -0.1,
    "fleece": -0.3, "knit": -0.3, "faux fur": -0.3, "terry": -0.3, "wool": -0.1,
    "corduroy": -0.4, "canvas": -0.4,
}

STYLE_TAG_FORMALITY_NUDGE = {
    "dressy": 0.5, "elegant": 0.5, "work": 0.5, "formal": 0.5, "cocktail": 0.5,
    "workwear": 0.5, "glamorous": 0.5,
    "casual": -0.5, "basic": -0.5, "athletic": -0.5, "sporty": -0.5,
    "lounge": -0.5, "activewear": -0.5,
}

# ── Occasion compatibility ────────────────────────────────────────────────────

_OCCASION_GROUPS = {
    "going-out": {"party", "going-out", "clubbing", "date", "dinner", "cocktail", "evening"},
    "work":      {"work", "office", "professional", "business"},
    "casual":    {"everyday", "casual", "weekend", "brunch", "errand"},
    "active":    {"workout", "gym", "athletic", "sport", "hiking"},
}

_DRESSY_MATERIALS = {"chiffon", "satin", "silk", "mesh", "lace", "organza", "velvet", "sequin"}
_CASUAL_MATERIALS = {"fleece", "terry", "canvas", "corduroy"}

_OCCASION_COMPAT = {
    ("going-out", "going-out"): 0.08,
    ("going-out", "work"):      0.02,
    ("going-out", "casual"):   -0.06,
    ("going-out", "active"):   -0.10,
    ("work",      "work"):      0.06,
    ("work",      "casual"):   -0.03,
    ("work",      "going-out"): 0.02,
    ("work",      "active"):   -0.08,
    ("casual",    "casual"):    0.04,
    ("casual",    "going-out"):-0.04,
    ("casual",    "work"):     -0.02,
    ("casual",    "active"):    0.00,
    ("active",    "active"):    0.04,
}


_WEAK_CASUAL_TAGS = {"everyday", "casual"}

_STRONG_GOING_OUT_TAGS = {"party", "clubbing", "cocktail", "dinner", "date",
                          "date-night", "evening", "night-out", "going-out"}
_STRONG_WORK_TAGS = {"work", "office", "business", "professional"}
_STRONG_ACTIVE_TAGS = {"gym", "workout", "sport", "athletic", "hiking"}


def infer_outfit_occasion(item: dict) -> str:
    """
    Derive the dominant occasion group from an item's occasion_tags, style_tags,
    and material. Returns one of: 'going-out', 'work', 'casual', 'active'.

    Explicit occasion tags (party, work, gym) are weighted heavily.
    'everyday' is treated as a weak default that doesn't override specific signals.
    """
    occ_tags = set(t.lower() for t in (item.get("occasion_tags") or []))
    style_tags = set(t.lower() for t in (item.get("style_tags") or []))
    material = (item.get("material") or "").lower()

    scores = {"going-out": 0, "work": 0, "casual": 0, "active": 0}

    strong_go = occ_tags & _STRONG_GOING_OUT_TAGS
    strong_wk = occ_tags & _STRONG_WORK_TAGS
    strong_ac = occ_tags & _STRONG_ACTIVE_TAGS
    has_specific_occ = bool(strong_go or strong_wk or strong_ac)

    scores["going-out"] += len(strong_go) * 3
    scores["work"] += len(strong_wk) * 3
    scores["active"] += len(strong_ac) * 3

    has_everyday = "everyday" in occ_tags
    has_casual_occ = "casual" in occ_tags
    if has_specific_occ:
        if has_everyday:
            scores["casual"] += 1
        if has_casual_occ:
            scores["casual"] += 1
    else:
        if has_everyday:
            scores["casual"] += 1
        if has_casual_occ:
            scores["casual"] += 2

    if any(m in material for m in _DRESSY_MATERIALS):
        scores["going-out"] += 3
    if any(m in material for m in _CASUAL_MATERIALS):
        scores["casual"] += 2

    _GOING_OUT_MATERIALS = {"leather", "suede", "velvet", "patent"}
    if any(m in material for m in _GOING_OUT_MATERIALS):
        scores["going-out"] += 2

    dressy_styles = style_tags & {"elegant", "chic", "sexy", "glamorous", "edgy"}
    scores["going-out"] += len(dressy_styles) * 2

    work_styles = style_tags & {"workwear", "professional", "classic"}
    scores["work"] += len(work_styles) * 2

    casual_styles = style_tags & {"basic", "relaxed", "sporty"}
    scores["casual"] += len(casual_styles)

    if "casual" in style_tags and not dressy_styles and not has_specific_occ:
        scores["casual"] += 1

    fit = (item.get("fit") or "").lower()
    color = (item.get("primary_color") or "").lower()
    _NIGHT_OUT_COLORS = {"black", "red", "navy", "white", "purple"}
    _VERSATILE_MATS = {"cotton", "jersey", "polyester", "knit", "ribbed knit",
                       "synthetic", "spandex"}
    if (fit == "fitted"
        and color in _NIGHT_OUT_COLORS
        and any(m in material for m in _VERSATILE_MATS)):
        scores["going-out"] += 2

    if max(scores.values()) == 0:
        return "casual"
    return max(scores, key=scores.get)


def _item_occasion_group(item: dict) -> str:
    """Get occasion group for a single item (cheaper version for scoring)."""
    occ_tags = set(t.lower() for t in (item.get("occasion_tags") or []))
    style_tags = set(t.lower() for t in (item.get("style_tags") or []))
    material = (item.get("material") or "").lower()
    all_signals = occ_tags | style_tags

    scores = {"going-out": 0, "work": 0, "casual": 0, "active": 0}
    for group, keywords in _OCCASION_GROUPS.items():
        scores[group] += len(all_signals & keywords) * 2
    if any(m in material for m in _DRESSY_MATERIALS):
        scores["going-out"] += 3
    if any(m in material for m in _CASUAL_MATERIALS):
        scores["casual"] += 2
    scores["going-out"] += len(all_signals & {"elegant", "chic", "sexy", "glamorous", "edgy"})
    scores["work"] += len(all_signals & {"workwear", "professional", "classic"})
    scores["casual"] += len(all_signals & {"basic", "relaxed", "sporty"})
    if max(scores.values()) == 0:
        return "casual"
    return max(scores, key=scores.get)


def check_occasion_coherence(base_item: dict, items_by_slot: dict) -> float:
    """
    Score occasion compatibility across all items. Returns a penalty (negative)
    for mismatches and a bonus (positive) for strong agreement.
    """
    anchor_occ = infer_outfit_occasion(base_item)
    total = 0.0
    count = 0
    for slot, item in items_by_slot.items():
        if not item:
            continue
        item_occ = _item_occasion_group(item)
        pair = (anchor_occ, item_occ)
        reverse = (item_occ, anchor_occ)
        compat = _OCCASION_COMPAT.get(pair, _OCCASION_COMPAT.get(reverse, 0.0))
        total += compat
        count += 1
    return total

COLOR_FORMALITY_NUDGE = {
    "black": 0.2, "navy": 0.2,
    "white": -0.2, "beige": -0.2, "neon": -0.2, "pastel": -0.2,
}

FORMALITY_RANGES = {
    "cardigan": (2.0, 4.0), "denim jacket": (2.0, 3.5),
    "jeans": (1.5, 3.0), "sneaker": (1.5, 3.0), "white sneaker": (1.5, 3.0),
    "blazer": (3.0, 5.0), "trench": (3.0, 5.0),
    "hoodie": (1.0, 2.0), "sweatshirt": (1.0, 2.0), "jogger": (1.0, 1.5),
    "stiletto": (4.0, 5.0), "gown": (4.5, 5.5), "tuxedo": (4.5, 5.5),
    "t-shirt": (1.5, 2.5), "tee": (1.5, 2.5), "tank": (1.5, 2.5),
    "blouse": (3.5, 5.0), "silk blouse": (3.5, 5.0),
    "dress": (3.0, 5.0), "skirt": (2.5, 4.5),
    "loafer": (2.5, 4.0), "flat": (2.5, 4.0), "boot": (2.0, 4.0),
    "heel": (3.5, 5.0), "pump": (3.5, 5.0),
    "legging": (1.0, 2.0), "flip flop": (1.0, 1.5), "sandal": (1.5, 3.0),
}

# ── Texture: 2-axis classification ───────────────────────────────────────────

MATERIAL_STRUCTURE = {
    "rigid": {"leather", "denim", "suede", "canvas", "corduroy", "tweed"},
    "fluid": {"chiffon", "silk", "satin", "lace", "linen", "organza", "tulle", "rayon"},
    "soft": {"knit", "wool", "cashmere", "fleece", "velvet", "faux fur", "mohair", "sherpa"},
    "neutral": {"cotton", "polyester", "nylon", "viscose", "jersey", "blend"},
}

MATERIAL_SURFACE = {
    "shiny": {"satin", "silk", "patent", "sequin", "metallic", "organza", "lamé"},
    "matte": {"cotton", "denim", "canvas", "wool", "linen", "knit", "suede", "corduroy", "jersey"},
    "textured": {"tweed", "corduroy", "faux fur", "fleece", "lace", "velvet", "bouclé", "sherpa"},
}

_TEXTURE_ANCHORS = {
    "structure": {
        "rigid": "structured stiff shape-holding leather denim suede canvas tweed garment",
        "fluid": "flowing draped lightweight silk chiffon satin lace organza garment",
        "soft": "soft cozy yielding knit wool cashmere fleece velvet garment",
    },
    "surface": {
        "shiny": "shiny glossy reflective lustrous satin silk patent sequin garment",
        "matte": "matte flat non-reflective cotton denim canvas wool suede garment",
        "textured": "textured nubby rough woven tweed bouclé fleece lace velvet garment",
    },
}
_FORMALITY_ANCHORS = {
    1: "very casual athletic gym shorts flip flops joggers leggings sports bra activewear",
    2: "casual everyday t-shirt tank top hoodie sweatshirt sneakers sandals",
    3: "smart casual jeans cardigan boots flats loafers sweater casual shirt",
    4: "dressy polished blouse trousers blazer heels pumps chinos dress",
    5: "formal elegant gown tuxedo suit stilettos cocktail dress evening wear",
}
_inference_anchor_cache = {}

# ── Volume classification ────────────────────────────────────────────────────

VOLUME_KEYWORDS = {
    "slim": {"skinny", "pencil", "slim", "tight", "fitted", "bodycon", "tapered"},
    "relaxed": {"wide", "oversized", "baggy", "flowy", "loose", "palazzo", "relaxed", "boyfriend"},
    "cropped": {"cropped", "crop", "bolero", "shrug"},
    "long": {"long", "duster", "maxi", "floor-length", "midi"},
}

FIT_TO_VOLUME = {
    "fitted": "slim", "bodycon": "slim", "slim": "slim",
    "relaxed": "relaxed", "oversized": "relaxed", "wide": "relaxed", "loose": "relaxed",
    "cropped": "cropped",
    "straight": "regular",
}

# ── Item role detection ──────────────────────────────────────────────────────

STATEMENT_KEYWORDS = {
    "print", "pattern", "statement", "sequin", "embellished", "bold",
    "graphic", "floral", "animal", "metallic", "neon", "glitter",
    "embroidered", "beaded", "rhinestone",
}

# ── Color harmony ────────────────────────────────────────────────────────────

ANALOGOUS = {
    "red": {"orange", "pink", "burgundy", "coral", "rust"},
    "orange": {"red", "yellow", "coral", "rust"},
    "yellow": {"orange", "green"},
    "green": {"yellow", "teal", "blue"},
    "blue": {"green", "purple", "navy", "teal"},
    "purple": {"blue", "pink", "lavender"},
    "pink": {"red", "purple", "coral", "lavender"},
}

# ── Shoe-bottom harmony ─────────────────────────────────────────────────────

CHUNKY_SHOE_KEYWORDS = {"chunky", "platform", "combat", "lug sole", "wedge sneaker"}

# ── Layer justification ──────────────────────────────────────────────────────

TOP_COVERAGE = {
    "exposed": {"cami", "camisole", "tank", "sleeveless", "strapless", "halter", "spaghetti", "tube", "bandeau"},
    "short_sleeve": {"t-shirt", "tee", "short sleeve", "polo", "crop top"},
    "full": {"long sleeve", "turtleneck", "hoodie", "sweatshirt", "mock neck", "pullover"},
}


# What slots to fill based on input item category
OUTFIT_SLOTS = {
    "top": ["bottom", "shoes", "accessory"],
    "bottom": ["top", "shoes", "accessory"],
    "dress": ["shoes", "layer", "accessory"],
    "shoes": ["top", "bottom", "accessory"],
    "layer": ["top", "bottom", "shoes"],  # Layer input: top underneath, no accessory needed
    "accessory": ["top", "bottom", "shoes"],
}

# Style direction definitions with color policies
STYLE_DIRECTIONS = {
    "Classic": {
        "style_tags": ["classic", "minimalist", "elegant", "preppy"],
        "vibe": "timeless and polished",
        "color_policy": "neutrals",  # Prefer neutral colors
        "allow_base_color": False,   # Avoid same color as input
    },
    "Trendy": {
        "style_tags": ["streetwear", "chic", "statement", "casual"],
        "vibe": "modern and fashion-forward",
        "color_policy": "two_tone",  # One accent allowed
        "allow_base_color": True,    # One item can match
    },
    "Bold": {
        "style_tags": ["edgy", "statement", "romantic", "bohemian"],
        "vibe": "daring and eye-catching",
        "color_policy": "contrast",  # Use complementary colors
        "allow_base_color": False,
    },
}

# Layer is now handled dynamically by scoring - see generate_candidate_outfits()


def get_slots_for_category(category: str) -> list[str]:
    """Get required slots to fill based on input item category."""
    return OUTFIT_SLOTS.get(category, ["top", "bottom", "shoes"])


def get_preferred_colors(direction: str, base_color: str, slot: str) -> set[str]:
    """
    Get preferred colors for a slot based on direction and base item color.
    
    Returns set of colors to prefer, or None if no preference.
    """
    dir_info = STYLE_DIRECTIONS.get(direction, {})
    color_policy = dir_info.get("color_policy", "neutrals")
    
    # Start with slot-specific preferences if any
    slot_prefs = SLOT_COLOR_PREFS.get(slot)
    
    if color_policy == "neutrals":
        # Classic: prefer neutrals
        return NEUTRALS
    elif color_policy == "contrast":
        # Bold: use complements
        return COMPLEMENTS.get(base_color, NEUTRALS)
    elif color_policy == "two_tone":
        # Trendy: neutrals + one accent allowed
        return NEUTRALS
    
    return slot_prefs or NEUTRALS


def get_avoid_colors(direction: str, base_color: str, slot: str) -> set[str]:
    """
    Get colors to avoid for a slot.
    """
    dir_info = STYLE_DIRECTIONS.get(direction, {})
    allow_base = dir_info.get("allow_base_color", False)
    
    avoid = set()
    
    # Shoes and accessories should almost always avoid base color
    if slot in ["shoes", "accessory"]:
        avoid.add(base_color)
    elif not allow_base:
        # For other slots, avoid if direction says so
        avoid.add(base_color)
    
    return avoid


def enforce_monochrome_cap(items_by_slot: dict, base_color: str, max_matching: int = 1) -> dict:
    """
    Ensure we don't have too many items matching the base color.
    
    Args:
        items_by_slot: Dict of slot → item
        base_color: The input item's primary color
        max_matching: Max items allowed to match base color
    
    Returns:
        Modified items_by_slot (may set some to None if over cap)
    """
    matching_slots = []
    for slot, item in items_by_slot.items():
        if item and item.get("primary_color") == base_color:
            matching_slots.append(slot)
    
    # If over cap, we'd need to flag slots to re-retrieve
    # For V1, just log the issue
    if len(matching_slots) > max_matching:
        # Priority to drop: accessory > shoes > bottom
        drop_priority = ["accessory", "shoes", "layer", "bottom", "top"]
        for drop_slot in drop_priority:
            if drop_slot in matching_slots and len(matching_slots) > max_matching:
                items_by_slot[drop_slot] = None  # Will need re-retrieval
                matching_slots.remove(drop_slot)
    
    return items_by_slot


def get_slots_for_outfit(base_category: str, outfit_index: int = 0) -> list[str]:
    """
    Get slots for a specific outfit.
    
    For top/bottom/shoes inputs, layer is always included as a candidate.
    The scoring will determine if the layer improves the outfit.
    
    Args:
        base_category: Category of the user's input item
        outfit_index: 0, 1, or 2 (unused now, kept for compatibility)
    
    Returns:
        List of slots to fill for this outfit
    """
    base_slots = get_slots_for_category(base_category)
    
    # Always include layer for top/bottom/shoes - scoring decides if it helps
    if "layer" not in base_slots and base_category in ["top", "bottom", "shoes"]:
        return base_slots + ["layer"]
    
    return base_slots


def build_base_item_text(base_item: dict) -> str:
    """Build a concise text description of the base item for query building."""
    parts = []
    
    if base_item.get("category"):
        parts.append(base_item["category"])
    
    if base_item.get("primary_color"):
        parts.append(base_item["primary_color"])
    
    if base_item.get("fit") and base_item["fit"] != "unknown":
        parts.append(f"{base_item['fit']} fit")
    
    if base_item.get("style_tags"):
        parts.append(" ".join(base_item["style_tags"][:2]))
    
    return " ".join(parts)


def generate_explanation(direction: str, base_item: dict, items_by_slot: dict[str, dict]) -> str:
    """Generate a direction-specific explanation for the outfit."""
    base_color = base_item.get("primary_color", "")
    base_style = (base_item.get("style_tags") or [])[:1]
    base_style_str = base_style[0] if base_style else ""
    
    # Get key items for explanation
    bottom = items_by_slot.get("bottom", {})
    shoes = items_by_slot.get("shoes", {})
    accessory = items_by_slot.get("accessory", {})
    
    bottom_type = ""
    if bottom:
        name_lower = bottom.get("name", "").lower()
        if "skirt" in name_lower:
            bottom_type = "skirt"
        elif "jeans" in name_lower:
            bottom_type = "jeans"
        elif "trousers" in name_lower or "pants" in name_lower:
            bottom_type = "tailored trousers"
        else:
            bottom_type = "bottom"
    
    if direction == "Classic":
        if bottom_type:
            return f"Timeless pairing: {bottom_type} and neutral tones create a polished, versatile look."
        return "A refined combination with neutral colors and clean silhouettes."
    
    elif direction == "Trendy":
        if accessory:
            acc_name = accessory.get("name", "").split()[0:2]
            return f"On-trend styling with a {' '.join(acc_name).lower()} to complete the look."
        return "Contemporary mix with fresh proportions and current-season appeal."
    
    elif direction == "Bold":
        if base_color and base_color not in {"black", "white", "grey", "beige"}:
            return f"Statement look: your {base_color} piece anchored by high-contrast items for maximum impact."
        return "Eye-catching ensemble with statement pieces and confident styling."
    
    return f"{direction} styling for a complete look."


# Hard penalty violations
BANNED_KEYWORDS = {
    "swimsuit", "swimwear", "bikini", "swim",
    "stockings", "tights", "hosiery",
    "girl's", "girls", "kid", "kids", "children", "boy", "boys",
    "dupatta", "innerwear", "underwear", "bra", "lingerie",
}

# ============== FIX 1: Outfit Intent Vector ==============

def compute_outfit_intent_vector(
    base_embedding: list,
    direction: str,
    taste_vector: list = None,
    dislike_vector: list = None
) -> list:
    """
    Create a unified "intent vector" that all outfit items should be close to.
    
    Combines:
    - base_embedding: The input item's embedding (weight: 0.6)
    - direction modifier: Slight adjustment based on style direction (weight: 0.2)
    - taste_vector: User's preference vector if available (weight: 0.15)
    - dislike_vector: Items to avoid, subtracted (weight: -0.05)
    
    Returns:
        Combined intent vector (same dimension as inputs)
    """
    if not base_embedding:
        return []
    
    dim = len(base_embedding)
    result = list(base_embedding)  # Start with base
    
    # Direction modifier weights (slight adjustment to push toward direction)
    # These are learned intuitions, not full embeddings
    direction_weights = {
        "Classic": {"neutral_boost": 0.05, "statement_dampen": -0.02},
        "Trendy": {"variety_boost": 0.03, "statement_boost": 0.03},
        "Bold": {"contrast_boost": 0.05, "statement_boost": 0.05},
    }
    
    # Blend taste vector if available (emphasize user preferences)
    if taste_vector and len(taste_vector) == dim:
        taste_weight = 0.15
        for i in range(dim):
            result[i] = result[i] * (1 - taste_weight) + taste_vector[i] * taste_weight
    
    # Subtract dislike vector if available (push away from disliked styles)
    if dislike_vector and len(dislike_vector) == dim:
        dislike_weight = 0.05
        for i in range(dim):
            result[i] = result[i] - dislike_vector[i] * dislike_weight
    
    return result


# ============== PHASE 1: Rich Item Inference ==============

def _get_inference_anchors():
    """Lazy-load and cache CLIP text anchor embeddings for fallback classification."""
    global _inference_anchor_cache
    if _inference_anchor_cache:
        return _inference_anchor_cache

    from services.retrieval import get_batch_embeddings
    import numpy as np

    texts = []
    keys = []

    for axis, anchors in _TEXTURE_ANCHORS.items():
        for label, text in anchors.items():
            keys.append(("texture", axis, label))
            texts.append(text)
    for level, text in _FORMALITY_ANCHORS.items():
        keys.append(("formality", level))
        texts.append(text)

    embeddings = get_batch_embeddings(texts)
    for key, emb in zip(keys, embeddings):
        _inference_anchor_cache[key] = np.array(emb)
    return _inference_anchor_cache


def _embedding_classify(item_embedding, anchor_group: str, axis: str = None):
    """Classify an item by cosine similarity to cached text anchors."""
    import numpy as np
    anchors = _get_inference_anchors()
    item_emb = np.array(item_embedding)
    norm = np.linalg.norm(item_emb)
    if norm == 0:
        return None

    best_label, best_sim = None, -1.0
    for key, anchor_emb in anchors.items():
        if anchor_group == "texture" and key[0] == "texture" and key[1] == axis:
            label = key[2]
        elif anchor_group == "formality" and key[0] == "formality":
            label = key[1]
        else:
            continue
        sim = float(np.dot(item_emb, anchor_emb) / (norm * np.linalg.norm(anchor_emb)))
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return best_label


def _match_material(material_str, name_str=""):
    """Find a known material keyword in material field or item name."""
    all_materials = set()
    for mats in MATERIAL_STRUCTURE.values():
        all_materials |= mats
    combined = (material_str + " " + name_str).lower()
    for mat in all_materials:
        if mat in combined:
            return mat
    return None


def infer_formality_continuous(item: dict) -> tuple:
    """
    Infer continuous formality score from item metadata.
    Returns (point_score, range_min, range_max).
    """
    if not item:
        return (3.0, 2.5, 3.5)

    name_lower = item.get("name", "").lower()
    category = (item.get("category") or "").lower()
    combined = name_lower + " " + category

    base = 3.0
    matched = False
    for keywords, score in GARMENT_TYPE_BASE_SCORES:
        if any(kw in combined for kw in keywords):
            base = score
            matched = True
            break

    if not matched:
        emb = item.get("embedding")
        if emb:
            level = _embedding_classify(emb, "formality")
            if level is not None:
                base = float(level)

    material = (item.get("material") or "").lower()
    mat_nudge = 0.0
    for mat_kw, nudge in TEXTURE_FORMALITY_NUDGE.items():
        if mat_kw in material or mat_kw in name_lower:
            mat_nudge = nudge
            break

    tag_nudge = 0.0
    for tag in (item.get("style_tags") or []):
        tag_lower = tag.lower()
        if tag_lower in STYLE_TAG_FORMALITY_NUDGE:
            tag_nudge = STYLE_TAG_FORMALITY_NUDGE[tag_lower]
            break

    color = (item.get("primary_color") or "").lower()
    color_nudge = COLOR_FORMALITY_NUDGE.get(color, 0.0)

    point = base + mat_nudge + tag_nudge + color_nudge

    range_min, range_max = point - 0.5, point + 0.5
    for kw, (rmin, rmax) in FORMALITY_RANGES.items():
        if kw in combined:
            range_min, range_max = rmin, rmax
            break

    return (round(point, 1), range_min, range_max)


def classify_texture(item: dict) -> tuple:
    """
    Classify item material along two axes: structure and surface.
    Returns (structure, surface) -- e.g. ("rigid", "matte").
    """
    if not item:
        return ("neutral", "matte")

    material = (item.get("material") or "").lower()
    name_lower = item.get("name", "").lower()
    mat_key = _match_material(material, name_lower)

    structure = "neutral"
    surface = "matte"

    if mat_key:
        for s, mats in MATERIAL_STRUCTURE.items():
            if mat_key in mats:
                structure = s
                break
        for s, mats in MATERIAL_SURFACE.items():
            if mat_key in mats:
                surface = s
                break
    else:
        emb = item.get("embedding")
        if emb:
            s = _embedding_classify(emb, "texture", "structure")
            if s and s != "neutral":
                structure = s
            sf = _embedding_classify(emb, "texture", "surface")
            if sf:
                surface = sf

    return (structure, surface)


_FLOWY_MATERIALS = {"chiffon", "sheer", "organza", "tulle", "georgette", "crepe"}
_STRUCTURED_MATERIALS = {"knit", "wool", "fleece", "denim", "tweed", "quilted"}


def infer_volume_class(item: dict) -> str:
    """Infer volume class from fit field, name keywords, and material drape."""
    if not item:
        return "regular"

    fit = (item.get("fit") or "").lower()
    material = (item.get("material") or "").lower()
    name_lower = item.get("name", "").lower()
    category = (item.get("category") or "").lower()

    vol = None
    if fit and fit in FIT_TO_VOLUME:
        vol = FIT_TO_VOLUME[fit]
    if not vol:
        for vol_class, keywords in VOLUME_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                vol = vol_class
                break
    if not vol:
        vol = "regular"

    is_flowy = any(m in material for m in _FLOWY_MATERIALS)
    is_structured = any(m in material for m in _STRUCTURED_MATERIALS)

    if is_flowy and vol == "relaxed":
        vol = "relaxed"
    elif is_structured and vol == "relaxed" and category == "layer":
        vol = "regular"

    return vol


def infer_item_role(item: dict) -> str:
    """
    Classify item as statement, supporting, or finisher.
    Shoes and accessories are always finishers.
    """
    if not item:
        return "supporting"

    category = (item.get("category") or "").lower()
    if category in ("shoes", "accessory"):
        return "finisher"

    name_lower = item.get("name", "").lower()
    tags = " ".join(item.get("style_tags") or []).lower()
    combined = name_lower + " " + tags

    if any(kw in combined for kw in STATEMENT_KEYWORDS):
        return "statement"

    color = (item.get("primary_color") or "").lower()
    if color and color not in NEUTRALS and color not in {"white", "beige", "brown"}:
        bold_colors = {"red", "orange", "yellow", "pink", "purple", "neon", "metallic", "multi"}
        if color in bold_colors:
            return "statement"

    return "supporting"


def enrich_items(base_item: dict, items_by_slot: dict) -> dict:
    """Run all inferences on base + slot items. Returns enriched metadata keyed by slot."""
    enriched = {}

    def _enrich(item):
        point, rmin, rmax = infer_formality_continuous(item)
        return {
            "formality": point,
            "formality_range": (rmin, rmax),
            "texture": classify_texture(item),
            "volume": infer_volume_class(item),
            "role": infer_item_role(item),
        }

    enriched["base"] = _enrich(base_item)

    for slot, item in items_by_slot.items():
        if item:
            enriched[slot] = _enrich(item)

    return enriched


# ============== PHASE 2: Outfit Composition Scoring ==============

def get_color_family(color: str) -> str:
    """Get the color family (warm/cool/neutral) for a color."""
    if not color:
        return "neutral"
    color_lower = color.lower()
    for family, colors in COLOR_FAMILIES.items():
        if color_lower in colors:
            return family
    return "neutral"


def check_formality_coherence(enriched: dict) -> float:
    """
    Check that all items' formality ranges overlap.
    Returns penalty (0 = all coherent, positive = gap exists).
    """
    ranges = [v["formality_range"] for v in enriched.values()]
    if len(ranges) < 2:
        return 0.0

    overall_min = max(r[0] for r in ranges)
    overall_max = min(r[1] for r in ranges)
    gap = overall_min - overall_max

    if gap <= 0:
        return 0.0
    return 0.15 * gap


def check_bookend_score(enriched: dict, items_by_slot: dict) -> float:
    """
    Sandwich/bookend check: outermost piece (layer or top) should echo shoes.
    Returns net bonus/penalty.
    """
    shoes_data = enriched.get("shoes")
    if not shoes_data:
        return 0.0

    outer_key = "layer" if "layer" in enriched and items_by_slot.get("layer") else "base"
    if outer_key == "base" and "top" in enriched:
        outer_key = "top"
    outer_data = enriched.get(outer_key)
    if not outer_data:
        outer_data = enriched.get("base")
    if not outer_data:
        return 0.0

    score = 0.0

    formality_diff = abs(outer_data["formality"] - shoes_data["formality"])
    if formality_diff <= 0.5:
        score += 0.05
    elif formality_diff > 1.5:
        score -= 0.08

    o_struct, o_surf = outer_data["texture"]
    s_struct, s_surf = shoes_data["texture"]
    if o_struct != "neutral" and s_struct != "neutral":
        if o_struct != s_struct:
            score += 0.03
        elif o_surf != s_surf:
            score += 0.03

    outer_item = items_by_slot.get(outer_key if outer_key != "base" else None)
    shoes_item = items_by_slot.get("shoes")
    if outer_item and shoes_item:
        o_color = get_color_family(outer_item.get("primary_color", ""))
        s_color = get_color_family(shoes_item.get("primary_color", ""))
        if o_color == s_color or (o_color == "neutral" and s_color == "neutral"):
            score += 0.03

    return round(score, 3)


def check_color_composition(base_item: dict, items_by_slot: dict) -> float:
    """
    60-30-10 color composition check.
    Returns bonus (positive) or penalty (negative).
    """
    colors = []
    bc = base_item.get("primary_color", "")
    if bc:
        colors.append(bc.lower())
    for item in items_by_slot.values():
        if item:
            c = item.get("primary_color", "")
            if c:
                colors.append(c.lower())

    if len(colors) < 2:
        return 0.0

    non_neutrals = [c for c in colors if c not in NEUTRALS]
    families = [get_color_family(c) for c in colors]
    unique_non_neutrals = set(non_neutrals)
    count = len(unique_non_neutrals)

    score = 0.0

    all_families = set(families)
    if len(all_families) == 1:
        score += 0.03

    if count == 0:
        score += 0.04
    elif count == 1:
        score += 0.04
    elif count == 2:
        c_list = list(unique_non_neutrals)
        a, b = c_list[0], c_list[1]
        is_complement = b in COMPLEMENTS.get(a, set()) or a in COMPLEMENTS.get(b, set())
        is_analogous = b in ANALOGOUS.get(a, set()) or a in ANALOGOUS.get(b, set())
        if not (is_complement or is_analogous):
            score -= 0.05
    else:
        score -= 0.08 * (count - 2)

    return round(score, 3)


def check_one_hero_rule(enriched: dict) -> float:
    """Penalize multiple statement pieces."""
    statement_count = sum(1 for v in enriched.values() if v["role"] == "statement")
    if statement_count <= 1:
        return 0.0
    if statement_count == 2:
        return 0.08
    return 0.15


_VOL_ORDER = {"slim": 0, "cropped": 1, "regular": 2, "relaxed": 3, "long": 4}


def check_proportion_balance(enriched: dict) -> float:
    """
    One-volume-at-a-time check.
    Returns bonus (positive) or penalty (negative).
    """
    top_vol = None
    bottom_vol = None
    layer_vol = None

    for key in ("base", "top"):
        if key in enriched:
            top_vol = enriched[key]["volume"]
            break

    if "bottom" in enriched:
        bottom_vol = enriched["bottom"]["volume"]
    if "layer" in enriched:
        layer_vol = enriched["layer"]["volume"]

    score = 0.0

    if top_vol and bottom_vol:
        if top_vol == "relaxed" and bottom_vol == "relaxed":
            score -= 0.06
        elif (top_vol == "slim" and bottom_vol == "relaxed") or \
             (top_vol == "relaxed" and bottom_vol == "slim"):
            score += 0.03

    if layer_vol and bottom_vol:
        if layer_vol in ("long", "relaxed") and bottom_vol == "relaxed":
            score -= 0.05
        elif layer_vol == "cropped" and bottom_vol == "relaxed":
            score += 0.03
        elif layer_vol == "long" and bottom_vol == "slim":
            score += 0.03

    if layer_vol and top_vol:
        layer_rank = _VOL_ORDER.get(layer_vol, 2)
        top_rank = _VOL_ORDER.get(top_vol, 2)
        if layer_rank < top_rank:
            score -= 0.07
        elif layer_rank >= top_rank + 1:
            score += 0.03

    return round(score, 3)


# ============== PHASE 3: Pairwise Checks ==============

def check_texture_contrast(enriched: dict, items_by_slot: dict) -> float:
    """Check texture contrast between adjacent items (top+layer, top+bottom)."""
    score = 0.0

    def _pair_score(a_tex, b_tex):
        a_struct, a_surf = a_tex
        b_struct, b_surf = b_tex
        if a_struct == "neutral" or b_struct == "neutral":
            return 0.0
        if a_struct == b_struct and a_surf == b_surf:
            return -0.04
        if a_struct != b_struct:
            return 0.03
        return 0.0

    top_key = "top" if "top" in enriched else "base"

    if top_key in enriched and "layer" in enriched:
        score += _pair_score(enriched[top_key]["texture"], enriched["layer"]["texture"])

    if top_key in enriched and "bottom" in enriched:
        score += _pair_score(enriched[top_key]["texture"], enriched["bottom"]["texture"])

    return round(score, 3)


def check_shoe_bottom_harmony(enriched: dict, items_by_slot: dict) -> float:
    """Check shoe-bottom proportion and formality harmony."""
    if "shoes" not in enriched or "bottom" not in enriched:
        return 0.0

    score = 0.0
    bottom_vol = enriched["bottom"]["volume"]
    shoes_item = items_by_slot.get("shoes")
    shoe_name = (shoes_item.get("name", "") if shoes_item else "").lower()

    if bottom_vol == "relaxed":
        if any(kw in shoe_name for kw in CHUNKY_SHOE_KEYWORDS):
            score -= 0.06

    if bottom_vol == "slim" and not any(kw in shoe_name for kw in CHUNKY_SHOE_KEYWORDS):
        score += 0.02

    f_diff = abs(enriched["shoes"]["formality"] - enriched["bottom"]["formality"])
    if f_diff > 2.0:
        score -= 0.05

    return round(score, 3)


def check_layer_justification(enriched: dict, items_by_slot: dict,
                               weather_context: dict = None) -> float:
    """Check whether a layer earns its place in the outfit."""
    layer_item = items_by_slot.get("layer")
    if not layer_item or "layer" not in enriched:
        return 0.0

    if weather_context:
        if weather_context.get("force_layer"):
            return 0.0
        if weather_context.get("skip_layer"):
            material = (layer_item.get("material") or "").lower()
            heavy_mats = {"wool", "fleece", "cashmere", "leather", "shearling", "down", "puffer"}
            if any(m in material for m in heavy_mats):
                return -0.06
            return 0.0

    top_key = "top" if "top" in enriched else "base"
    top_data = enriched.get(top_key, {})
    layer_data = enriched["layer"]

    top_item = items_by_slot.get("top")
    top_name = (top_item.get("name", "") if top_item else "").lower()
    if not top_name:
        top_name = (items_by_slot.get("_base", {}).get("name", "")).lower()

    coverage = "full"
    for cov_type, keywords in TOP_COVERAGE.items():
        if any(kw in top_name for kw in keywords):
            coverage = cov_type
            break

    if coverage == "exposed":
        return 0.0
    if coverage == "short_sleeve":
        return -0.02

    score = 0.0
    justified = False

    l_struct = layer_data["texture"][0]
    t_struct = top_data.get("texture", ("neutral", "matte"))[0]
    if l_struct == "rigid" and t_struct in ("soft", "neutral"):
        justified = True

    if not justified:
        if l_struct != t_struct and l_struct != "neutral" and t_struct != "neutral":
            justified = True

    if not justified:
        if layer_data["formality"] > top_data.get("formality", 3.0) + 0.5:
            justified = True

    if not justified:
        l_tex = layer_data["texture"]
        t_tex = top_data.get("texture", ("neutral", "matte"))
        same_texture = (l_tex[0] == t_tex[0] and l_tex[1] == t_tex[1])
        l_range = layer_data["formality_range"]
        t_range = top_data.get("formality_range", (2.5, 3.5))
        overlapping = l_range[0] <= t_range[1] and t_range[0] <= l_range[1]
        if same_texture and overlapping:
            score -= 0.06

    layer_name = layer_item.get("name", "").lower()
    pullover_kws = {"pullover", "crewneck", "sweatshirt"}
    bulky_top_kws = {"turtleneck", "hoodie", "sweatshirt"}
    is_pullover_layer = any(kw in layer_name for kw in pullover_kws)
    is_bulky_top = any(kw in top_name for kw in bulky_top_kws)

    if is_pullover_layer and is_bulky_top:
        score -= 0.08

    if layer_data["volume"] == "relaxed" and top_data.get("volume", "regular") == "relaxed":
        score -= 0.05

    return round(score, 3)


# ============== HIERARCHICAL SCORING ==============

_LOUD_COLORS = {
    "red", "orange", "yellow", "pink", "purple", "green", "blue",
    "coral", "fuchsia", "magenta", "teal", "turquoise", "lime", "neon",
}


def compute_visual_loudness(item: dict, enriched_data: dict) -> float:
    """How much attention does this item demand? Returns 0.0 - 1.0."""
    loudness = 0.0

    color = (item.get("primary_color") or "").lower()
    if color in _LOUD_COLORS:
        loudness += 0.30
    elif color and color not in NEUTRALS:
        loudness += 0.15

    tex = enriched_data.get("texture", ("neutral", "matte"))
    if tex[1] == "shiny":
        loudness += 0.20
    elif tex[1] == "textured":
        loudness += 0.10

    name = (item.get("name") or "").lower()
    tags = " ".join(item.get("style_tags") or []).lower()
    combined = name + " " + tags
    if any(kw in combined for kw in STATEMENT_KEYWORDS):
        loudness += 0.30

    vol = enriched_data.get("volume", "regular")
    if vol in ("cropped", "long", "relaxed"):
        loudness += 0.10

    return min(loudness, 1.0)


def compute_visual_cohesion(
    base_item: dict,
    items_by_slot: dict[str, dict],
    intent_vector: list,
) -> dict:
    """
    Outfit-level visual measurements: centroid, intent alignment, spread.
    Returns dict with 'centroid', 'intent_alignment', 'spread', 'embeddings'.
    """
    embeddings = []
    base_emb = base_item.get("embedding") or []
    if base_emb:
        embeddings.append(base_emb)
    for item in items_by_slot.values():
        if item:
            emb = item.get("embedding") or []
            if emb:
                embeddings.append(emb)

    if not embeddings:
        return {"centroid": [], "intent_alignment": 0.5, "spread": 0.3, "embeddings": []}

    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        for i in range(min(dim, len(emb))):
            centroid[i] += emb[i]
    centroid = [c / len(embeddings) for c in centroid]

    intent_alignment = cosine_similarity(centroid, intent_vector) if intent_vector else 0.5

    distances = [1.0 - cosine_similarity(centroid, emb) for emb in embeddings]
    spread = sum(distances) / len(distances) if distances else 0.3

    return {
        "centroid": centroid,
        "intent_alignment": intent_alignment,
        "spread": spread,
        "embeddings": embeddings,
    }


# ── Level 1: Silhouette & Proportion ─────────────────────────────────────────

def score_silhouette(
    enriched: dict,
    items_by_slot: dict,
    cohesion: dict,
) -> tuple[float, bool]:
    """
    Foundation level. Proportion balance + shoe-bottom harmony + embedding spread.
    Returns (score, is_solid).
    """
    proportion = check_proportion_balance(enriched)
    shoe_bottom = check_shoe_bottom_harmony(enriched, items_by_slot)

    spread = cohesion.get("spread", 0.3)
    spread_adj = 0.0
    if spread < 0.25:
        spread_adj = 0.03
    elif spread > 0.40:
        spread_adj = -0.05 * ((spread - 0.40) / 0.10)
        spread_adj = max(spread_adj, -0.15)

    score = proportion + shoe_bottom + spread_adj
    is_solid = score > -0.03
    return round(score, 4), is_solid


# ── Level 2: Color Surfaces ──────────────────────────────────────────────────

def _check_color_calm(base_item: dict, items_by_slot: dict) -> float:
    """Are the large surfaces settled or fighting?"""
    colors = []
    bc = (base_item.get("primary_color") or "").lower()
    if bc:
        colors.append(bc)
    for item in items_by_slot.values():
        if item:
            c = (item.get("primary_color") or "").lower()
            if c:
                colors.append(c)

    if len(colors) < 2:
        return 0.03

    families = [get_color_family(c) for c in colors]
    unique_families = set(families)
    non_neutral_families = unique_families - {"neutral"}

    score = 0.0

    if len(non_neutral_families) == 0:
        score += 0.03
    elif len(non_neutral_families) == 1:
        score += 0.02
    elif len(non_neutral_families) >= 3:
        score -= 0.04 * (len(non_neutral_families) - 2)

    if len(set(colors)) == 1 and len(colors) >= 3:
        score -= 0.03

    unique_non_neutrals = set(c for c in colors if c not in NEUTRALS)
    if len(unique_non_neutrals) >= 2:
        shades = len(unique_non_neutrals)
        if shades == 2:
            pass
        elif shades >= 3:
            score += 0.02

    return round(score, 4)


def _check_adjacent_color_clash(base_item: dict, items_by_slot: dict) -> float:
    """Penalize visually adjacent slots (top+layer) with clashing non-neutral colors."""
    layer = items_by_slot.get("layer")
    if not layer:
        return 0.0

    top_color = (base_item.get("primary_color") or "").lower()
    layer_color = (layer.get("primary_color") or "").lower()

    if not top_color or not layer_color:
        return 0.0
    if top_color in NEUTRALS or layer_color in NEUTRALS:
        return 0.0
    if top_color == layer_color:
        return 0.0

    top_family = get_color_family(top_color)
    layer_family = get_color_family(layer_color)
    if top_family == layer_family:
        return 0.0

    is_complement = (layer_color in COMPLEMENTS.get(top_color, set())
                     or top_color in COMPLEMENTS.get(layer_color, set()))
    is_analogous = (layer_color in ANALOGOUS.get(top_color, set())
                    or top_color in ANALOGOUS.get(layer_color, set()))

    if is_analogous:
        return 0.0
    if is_complement:
        return -0.03
    return -0.08


def score_color_surfaces(
    enriched: dict,
    items_by_slot: dict,
    base_item: dict,
    silhouette_solid: bool,
) -> tuple[float, bool]:
    """
    Level 2. Color composition + bookend + calm.
    Returns (score, is_calm).
    """
    composition = check_color_composition(base_item, items_by_slot)
    bookend = check_bookend_score(enriched, items_by_slot)
    calm = _check_color_calm(base_item, items_by_slot)
    adjacent = _check_adjacent_color_clash(base_item, items_by_slot)

    gate = 1.0 if silhouette_solid else 0.7
    score = (composition + bookend + calm + adjacent) * gate

    is_calm = score > -0.02
    return round(score, 4), is_calm


# ── Level 3: Texture & Narrative ─────────────────────────────────────────────

def _check_texture_variety(enriched: dict) -> float:
    """Reward variety, penalize flatness across the whole outfit."""
    structures = set()
    surfaces = set()
    for data in enriched.values():
        tex = data.get("texture", ("neutral", "matte"))
        if tex[0] != "neutral":
            structures.add(tex[0])
        surfaces.add(tex[1])

    score = 0.0
    if len(structures) >= 3:
        score += 0.03
    elif len(structures) == 0:
        score -= 0.02

    all_textures = [(d["texture"][0], d["texture"][1]) for d in enriched.values()
                    if d.get("texture", ("neutral", "matte"))[0] != "neutral"]
    if len(all_textures) >= 2 and len(set(all_textures)) == 1:
        score -= 0.03

    return round(score, 4)


def _score_narrative(
    enriched: dict,
    items_by_slot: dict,
    base_item: dict,
) -> dict:
    """
    Hero clarity, attention gradient, support deference.
    Returns dict with individual scores and hero_count.
    """
    loudness_map = {}
    loudness_map["base"] = compute_visual_loudness(base_item, enriched.get("base", {}))
    for slot, item in items_by_slot.items():
        if item and slot in enriched:
            loudness_map[slot] = compute_visual_loudness(item, enriched[slot])

    if not loudness_map:
        return {"hero_clarity": 0.0, "gradient": 0.0, "deference": 0.0, "hero_count": 0}

    sorted_items = sorted(loudness_map.items(), key=lambda x: x[1], reverse=True)
    loudest_key, loudest_val = sorted_items[0]
    second_val = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    hero_gap = loudest_val - second_val
    hero_count = sum(1 for _, v in sorted_items if v >= loudest_val - 0.05) if loudest_val > 0.2 else 0

    hero_clarity = 0.0
    if hero_count == 1 and hero_gap >= 0.2:
        hero_clarity = 0.05
    elif hero_count == 1 and hero_gap >= 0.1:
        hero_clarity = 0.03
    elif hero_count == 0:
        hero_clarity = -0.03
    elif hero_count == 2:
        hero_clarity = -0.06
    elif hero_count >= 3:
        hero_clarity = -0.12

    if hero_count == 1 and loudest_key in ("base", "top"):
        layer_item = items_by_slot.get("layer")
        if layer_item:
            layer_name = (layer_item.get("name") or "").lower()
            pullover_kws = {"pullover", "crewneck", "sweatshirt", "hoodie"}
            if any(kw in layer_name for kw in pullover_kws):
                hero_clarity -= 0.04

    gradient_score = 0.0
    if len(sorted_items) >= 3 and hero_count == 1:
        hero_loudness = sorted_items[0][1]
        violations = 0
        for key, val in sorted_items[1:]:
            role = enriched.get(key, {}).get("role", "supporting")
            if role == "supporting" and val > hero_loudness - 0.05:
                violations += 1
        if violations == 0:
            gradient_score = 0.02
        else:
            gradient_score = -0.02 * violations

    deference = 0.0
    if hero_count == 1:
        hero_loudness = sorted_items[0][1]
        supporting_loudness = [v for k, v in sorted_items[1:]]
        if supporting_loudness:
            avg_supporting = sum(supporting_loudness) / len(supporting_loudness)
            gap = hero_loudness - avg_supporting
            if gap >= 0.3:
                deference = 0.03
            elif gap >= 0.15:
                deference = 0.01
            elif gap < 0.05:
                deference = -0.02

    return {
        "hero_clarity": round(hero_clarity, 4),
        "gradient": round(gradient_score, 4),
        "deference": round(deference, 4),
        "hero_count": hero_count,
    }


def score_texture_and_narrative(
    enriched: dict,
    items_by_slot: dict,
    base_item: dict,
    color_calm: bool,
    weather_context: dict = None,
) -> tuple[float, bool]:
    """
    Level 3. Texture variety + contrast + narrative structure + layer justification.
    Returns (score, story_clear).
    """
    contrast = check_texture_contrast(enriched, items_by_slot)
    variety = _check_texture_variety(enriched)
    layer_adj = check_layer_justification(enriched, items_by_slot, weather_context)

    texture_gate = 1.2 if color_calm else 0.7
    texture_score = (contrast + variety) * texture_gate + layer_adj

    narrative = _score_narrative(enriched, items_by_slot, base_item)
    narrative_score = narrative["hero_clarity"] + narrative["gradient"] + narrative["deference"]

    score = texture_score + narrative_score
    story_clear = narrative["hero_count"] == 1 and narrative["hero_clarity"] >= 0.0
    return round(score, 4), story_clear, narrative


# ── Level 4: Finishing & Intent ──────────────────────────────────────────────

def score_finishing(
    enriched: dict,
    items_by_slot: dict,
    cohesion: dict,
    direction: str,
    base_item: dict,
    story_clear: bool,
) -> float:
    """
    Level 4. Intent alignment + direction match + formality coherence + occasion coherence.
    """
    intent_alignment = cohesion.get("intent_alignment", 0.5)

    direction_bonus = compute_direction_bonus(base_item, items_by_slot, direction)
    direction_gate = 1.2 if story_clear else 0.8
    gated_direction = direction_bonus * direction_gate

    formality_pen = check_formality_coherence(enriched)
    occasion_adj = check_occasion_coherence(base_item, items_by_slot)

    score = 0.35 * intent_alignment + gated_direction - formality_pen + occasion_adj
    return round(score, 4)


def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def check_hard_violations(
    base_item: dict, 
    items_by_slot: dict[str, dict], 
    direction: str
) -> list[str]:
    """
    Check for hard violations that should disqualify an outfit.
    Returns list of violation reasons (empty = no violations).
    """
    violations = []
    base_color = base_item.get("primary_color", "")
    
    for slot, item in items_by_slot.items():
        if not item:
            continue
        name_lower = item.get("name", "").lower()
        item_color = item.get("primary_color", "")
        
        # Banned keywords
        for kw in BANNED_KEYWORDS:
            if kw in name_lower:
                violations.append(f"banned: {kw} in {slot}")
        
        # Same color as base (unless intentional monochrome)
        if item_color and item_color == base_color and item_color not in NEUTRALS:
            violations.append(f"same color as base: {item_color} in {slot}")
    
    # Trendy same-neutral check
    if direction == "Trendy":
        bottom = items_by_slot.get("bottom")
        shoes = items_by_slot.get("shoes")
        if bottom and shoes:
            bc = bottom.get("primary_color", "")
            sc = shoes.get("primary_color", "")
            if bc == sc and bc in NEUTRALS and bc not in {"black", "white"}:
                violations.append(f"trendy same-neutral: {bc} bottom + shoes")
    
    return violations


def compute_direction_bonus(
    base_item: dict, 
    items_by_slot: dict[str, dict], 
    direction: str
) -> float:
    """Compute direction-specific bonus score."""
    bonus = 0.0
    
    colors = [base_item.get("primary_color", "")]
    for item in items_by_slot.values():
        if item:
            colors.append(item.get("primary_color", ""))
    
    neutral_count = sum(1 for c in colors if c in NEUTRALS)
    has_statement = False
    has_metallic = any("metallic" in c for c in colors if c)
    has_black_white = "black" in colors or "white" in colors
    
    for item in items_by_slot.values():
        if item:
            name_lower = item.get("name", "").lower()
            tags = item.get("style_tags") or []
            if "statement" in name_lower or any("statement" in t.lower() for t in tags):
                has_statement = True
    
    if direction == "Classic":
        # Reward neutral palette
        if neutral_count >= len(colors) - 1:
            bonus += 0.15
        # Reward classic/minimal tags
        for item in items_by_slot.values():
            if item:
                tags = item.get("style_tags") or []
                if any(t.lower() in {"classic", "minimal", "elegant", "timeless"} for t in tags):
                    bonus += 0.05
    
    elif direction == "Trendy":
        # Reward contrast and statement
        if has_metallic:
            bonus += 0.1
        if has_statement:
            bonus += 0.1
        # Reward variety (not all neutrals)
        if neutral_count < len(colors):
            bonus += 0.05
    
    elif direction == "Bold":
        # Reward contrast OR statement OR black/white anchor
        if has_statement:
            bonus += 0.15
        if has_black_white and neutral_count < len(colors):
            bonus += 0.1  # Good contrast
        # Reward non-neutral colors
        non_neutral = len(colors) - neutral_count
        if non_neutral >= 1:
            bonus += 0.05
    
    return min(bonus, 0.3)  # Cap bonus at 0.3


def score_outfit(
    base_item: dict,
    items_by_slot: dict[str, dict],
    direction: str,
    base_embedding: list = None,
    intent_vector: list = None,
    taste_vector: list = None,
    dislike_vector: list = None,
    weather_context: dict = None,
) -> dict:
    """
    Hierarchical outfit scoring mirroring how the eye reads an outfit:
    silhouette -> color surfaces -> texture/narrative -> finishing/intent.

    Each level gates the levels above it. A broken silhouette dampens
    everything; busy color surfaces dampen texture appreciation.

    Returns dict with total score, violations, and per-level breakdown.
    """
    violations = check_hard_violations(base_item, items_by_slot, direction)
    if violations:
        return {"total": -1.0, "violations": violations, "breakdown": {}}

    base_emb = base_embedding or []
    if not intent_vector and base_emb:
        intent_vector = compute_outfit_intent_vector(
            base_emb, direction, taste_vector, dislike_vector
        )

    enriched = enrich_items(base_item, items_by_slot)
    cohesion = compute_visual_cohesion(base_item, items_by_slot, intent_vector)

    # Level 1: Silhouette & Proportion (foundation)
    l1_score, silhouette_solid = score_silhouette(enriched, items_by_slot, cohesion)

    # Level 2: Color Surfaces (gated by L1)
    l2_score, color_calm = score_color_surfaces(
        enriched, items_by_slot, base_item, silhouette_solid
    )

    # Level 3: Texture & Narrative (gated by L2)
    l3_score, story_clear, narrative_detail = score_texture_and_narrative(
        enriched, items_by_slot, base_item, color_calm, weather_context
    )

    # Level 4: Finishing & Intent (gated by L3)
    l4_score = score_finishing(
        enriched, items_by_slot, cohesion, direction, base_item, story_clear
    )

    # Hierarchy dampening: broken foundation suppresses upper levels
    dampening = 1.0
    if not silhouette_solid:
        dampening *= 0.70
    if not color_calm:
        dampening *= 0.85

    # Bookend forgiveness: strong framing eases penalties in L2/L3
    bookend_raw = check_bookend_score(enriched, items_by_slot)
    if bookend_raw > 0.04:
        l3_score = l3_score * 1.0 + abs(min(l3_score, 0)) * 0.15

    total = (
        0.30 * l1_score +
        0.25 * l2_score +
        0.25 * l3_score * dampening +
        0.20 * l4_score * dampening
    )

    return {
        "total": round(total, 4),
        "violations": [],
        "breakdown": {
            "silhouette": round(l1_score, 4),
            "silhouette_solid": silhouette_solid,
            "color_surfaces": round(l2_score, 4),
            "color_calm": color_calm,
            "texture_narrative": round(l3_score, 4),
            "story_clear": story_clear,
            "finishing": round(l4_score, 4),
            "dampening": round(dampening, 3),
            "intent_alignment": round(cohesion.get("intent_alignment", 0.5), 4),
            "spread": round(cohesion.get("spread", 0.3), 4),
            "hero_clarity": narrative_detail.get("hero_clarity", 0.0),
            "hero_count": narrative_detail.get("hero_count", 0),
            "gradient": narrative_detail.get("gradient", 0.0),
            "deference": narrative_detail.get("deference", 0.0),
        }
    }


def pick_anchor_pair(
    base_item: dict,
    bottom_candidates: list[dict],
    shoes_candidates: list[dict],
    top_k: int = 3,
) -> tuple[dict | None, dict | None]:
    """
    Pick the most formality-coherent (bottom, shoes) pair to anchor the silhouette.
    Tries top_k of each, returns the pair with the tightest formality fit to the base.
    """
    if not bottom_candidates and not shoes_candidates:
        return None, None

    base_enriched = {}
    point, rmin, rmax = infer_formality_continuous(base_item)
    base_enriched["formality"] = point
    base_enriched["formality_range"] = (rmin, rmax)

    best_pair = (None, None)
    best_score = -999

    bottoms = bottom_candidates[:top_k] if bottom_candidates else [None]
    shoes = shoes_candidates[:top_k] if shoes_candidates else [None]

    for b in bottoms:
        for s in shoes:
            items = {"bottom": b, "shoes": s}
            enriched = enrich_items(base_item, items)

            silhouette_score = check_proportion_balance(enriched)
            shoe_bottom = check_shoe_bottom_harmony(enriched, items)
            formality_pen = check_formality_coherence(enriched)

            bookend = check_bookend_score(enriched, items)

            pair_score = silhouette_score + shoe_bottom + bookend - formality_pen
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (b, s)

    return best_pair


def generate_candidate_outfits(
    slots: list[str],
    candidates_by_slot: dict[str, list[dict]],
    max_candidates: int = 8,
    require_layer: bool = False
) -> list[dict[str, dict]]:
    """
    Generate multiple candidate outfit combinations from slot candidates.
    Uses simple combinatorial approach: vary each slot independently.
    
    Layer is optional by default (scoring decides), unless require_layer=True.
    
    Args:
        slots: List of slots to fill
        candidates_by_slot: Dict mapping slot -> list of candidate items
        max_candidates: Max number of outfit candidates to generate
        require_layer: If True, layer is required (e.g., cold weather)
    
    Returns:
        List of items_by_slot dicts (each is one candidate outfit)
    """
    import itertools
    
    # Get top candidates per slot (limit to avoid explosion)
    slot_options = []
    for slot in slots:
        options = candidates_by_slot.get(slot, [])[:3]  # Top 3 per slot
        if options:
            slot_candidates = [(slot, opt) for opt in options]
            # Layer is optional UNLESS require_layer is True
            if slot == "layer" and not require_layer:
                slot_candidates.append((slot, None))
            slot_options.append(slot_candidates)
        else:
            slot_options.append([(slot, None)])
    
    # Generate combinations
    combinations = list(itertools.product(*slot_options))[:max_candidates]
    
    # Convert to items_by_slot format
    outfits = []
    for combo in combinations:
        items_by_slot = {slot: item for slot, item in combo}
        outfits.append(items_by_slot)
    
    return outfits


def select_best_outfit(
    candidate_outfits: list[dict[str, dict]],
    base_item: dict,
    direction: str,
    base_embedding: list = None,
    taste_vector: list = None,
    dislike_vector: list = None,
    weather_context: dict = None,
) -> tuple[dict[str, dict], dict]:
    """
    Score all candidate outfits and return the best one.

    Returns:
        (best_items_by_slot, score_details)
    """
    best_outfit = None
    best_score = -999
    best_details = {}

    intent_vector = compute_outfit_intent_vector(
        base_embedding, direction, taste_vector, dislike_vector
    ) if base_embedding else None

    for items_by_slot in candidate_outfits:
        score_result = score_outfit(
            base_item, items_by_slot, direction,
            base_embedding=base_embedding,
            intent_vector=intent_vector,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector,
            weather_context=weather_context,
        )
        
        if score_result["total"] > best_score:
            best_score = score_result["total"]
            best_outfit = items_by_slot
            best_details = score_result
    
    return best_outfit or candidate_outfits[0], best_details


def assemble_outfit(
    direction: str,
    base_item: dict,
    items_by_slot: dict[str, dict],
    base_embedding: list = None,
    taste_vector: list = None,
    dislike_vector: list = None,
    weather_context: dict = None,
) -> dict:
    """
    Assemble a single outfit response with scoring.

    Returns:
        Outfit dict ready for API response
    """
    items = []
    for slot, item in items_by_slot.items():
        if item:
            items.append({
                "slot": slot,
                "id": item["id"],
                "name": item["name"],
                "image_url": item["image_url"],
                "shop_url": item.get("product_url"),
                "primary_color": item.get("primary_color"),
                "occasion_tags": item.get("occasion_tags", []),
                "style_tags": item.get("style_tags", []),
                "score": item.get("score"),  # Item's match score
            })
    
    intent_vector = compute_outfit_intent_vector(
        base_embedding, direction, taste_vector, dislike_vector
    ) if base_embedding else None

    score_result = score_outfit(
        base_item, items_by_slot, direction,
        base_embedding=base_embedding,
        intent_vector=intent_vector,
        taste_vector=taste_vector,
        dislike_vector=dislike_vector,
        weather_context=weather_context,
    )
    
    return {
        "direction": direction,
        "items": items,
        "explanation": generate_explanation(direction, base_item, items_by_slot),
        "score": score_result["total"],
        "score_breakdown": score_result["breakdown"],
        "violations": score_result["violations"]
    }


def apply_diversity_rule(outfits: list[dict]) -> list[dict]:
    """
    Ensure variety across outfits - avoid same item in multiple outfits.
    
    Simple V1 rule: if outfit 1 and 2 have same item in a slot,
    outfit 2 should use the backup candidate.
    
    This is called after initial selection with backup candidates available.
    """
    # Track used item IDs per slot
    used_by_slot = {}
    
    for outfit in outfits:
        for item in outfit.get("items", []):
            slot = item["slot"]
            item_id = item["item_id"]
            
            if slot not in used_by_slot:
                used_by_slot[slot] = set()
            
            # If already used, we'd need to swap with backup
            # For V1, we just flag it (actual swap happens in retrieval)
            used_by_slot[slot].add(item_id)
    
    return outfits

