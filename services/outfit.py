"""
Outfit structure definitions and assembly logic.
"""

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

# Formality levels (Fix 2) - higher = more formal
FORMALITY_LEVELS = {
    "casual": 0,       # sneakers, t-shirts, joggers
    "smart_casual": 1, # flats, blouses, jeans
    "dressy": 2,       # heels, clutch, blazer
}

# Keywords to infer formality level
FORMALITY_KEYWORDS = {
    0: {"sneaker", "flip flop", "jogger", "shorts", "t-shirt", "tank", "hoodie", "sandal", "sweatpant", "sweatshirt"},
    1: {"jeans", "flat", "loafer", "blouse", "shirt", "cardigan", "sweater", "skirt", "boot"},  
    2: {"heel", "pump", "stiletto", "clutch", "blazer", "trousers", "pencil", "gown", "dress shoe", "wedge"},
}

# Occasion tags for consistency (Fix 2)
OCCASION_KEYWORDS = {
    "casual": {"weekend", "casual", "everyday", "daytime", "errand", "relax", "lounge"},
    "work": {"office", "work", "professional", "business", "meeting"},
    "evening": {"evening", "dinner", "date", "night out", "party", "cocktail"},
    "formal": {"formal", "gala", "wedding", "event", "special occasion"},
}

# Slot-specific color preferences (what colors work best per slot)
SLOT_COLOR_PREFS = {
    "shoes": {"black", "white", "beige", "gray", "brown", "metallic", "navy"},
    "accessory": {"black", "beige", "brown", "white", "metallic", "navy"},
    "bottom": None,  # Use direction-based logic
    "top": None,
    "layer": {"black", "white", "gray", "beige", "navy", "brown"},
}

# What slots to fill based on input item category
OUTFIT_SLOTS = {
    "top": ["bottom", "shoes", "accessory"],
    "bottom": ["top", "shoes", "accessory"],
    "dress": ["shoes", "layer", "accessory"],
    "shoes": ["top", "bottom", "accessory"],
    "layer": ["top", "bottom", "shoes"],
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

# Which outfits include a layer (for tops/bottoms)
# Outfit 2 gets a layer for variety
LAYER_IN_OUTFIT = {1}  # 0-indexed: outfit index 1 (the second outfit)


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


def get_slots_for_outfit(base_category: str, outfit_index: int) -> list[str]:
    """
    Get slots for a specific outfit, considering layer rules.
    
    Args:
        base_category: Category of the user's input item
        outfit_index: 0, 1, or 2
    
    Returns:
        List of slots to fill for this outfit
    """
    base_slots = get_slots_for_category(base_category)
    
    # Add layer to outfit 1 (index 1) if not already included and category allows
    if outfit_index in LAYER_IN_OUTFIT:
        if "layer" not in base_slots and base_category in ["top", "bottom"]:
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
            return f"On-trend styling with modern proportions and a {' '.join(acc_name).lower()} to complete the look."
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

DRESSY_KEYWORDS = {"heel", "pump", "blazer", "trousers", "pencil", "clutch", "dress"}
CASUAL_KEYWORDS = {"sneaker", "flip flop", "jogger", "shorts", "t-shirt", "tank", "sandal", "hoodie"}


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


# ============== FIX 2: Formality & Occasion Consistency ==============

def infer_formality(item: dict) -> int:
    """
    Infer formality level from item name.
    Returns: 0 (casual), 1 (smart_casual), 2 (dressy)
    """
    if not item:
        return 1  # Default to smart_casual
    
    name_lower = item.get("name", "").lower()
    
    # Check dressy first (highest specificity)
    for kw in FORMALITY_KEYWORDS[2]:
        if kw in name_lower:
            return 2
    
    # Check casual
    for kw in FORMALITY_KEYWORDS[0]:
        if kw in name_lower:
            return 0
    
    # Default to smart_casual
    return 1


def infer_occasions(item: dict) -> set:
    """
    Infer plausible occasions from item name/tags.
    Returns: Set of occasion types
    """
    if not item:
        return {"casual", "work"}  # Default
    
    name_lower = item.get("name", "").lower()
    tags = " ".join(item.get("style_tags") or []).lower()
    combined = name_lower + " " + tags
    
    occasions = set()
    
    for occasion, keywords in OCCASION_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            occasions.add(occasion)
    
    # Infer from formality if no explicit occasions
    if not occasions:
        formality = infer_formality(item)
        if formality == 0:
            occasions = {"casual"}
        elif formality == 2:
            occasions = {"evening", "work"}
        else:
            occasions = {"casual", "work"}
    
    return occasions


def check_formality_consistency(items_by_slot: dict, base_item: dict) -> tuple[bool, float]:
    """
    Check if all items are within ±1 formality step of each other.
    
    Returns:
        (is_consistent: bool, penalty: float)
    """
    formalities = []
    
    # Include base item formality
    base_formality = infer_formality(base_item)
    formalities.append(base_formality)
    
    for item in items_by_slot.values():
        if item:
            formalities.append(infer_formality(item))
    
    if not formalities:
        return True, 0.0
    
    min_f = min(formalities)
    max_f = max(formalities)
    spread = max_f - min_f
    
    # Within ±1 step is OK
    if spread <= 1:
        return True, 0.0
    
    # Penalty scales with spread
    penalty = 0.2 * (spread - 1)
    return False, penalty


def check_occasion_consistency(items_by_slot: dict, base_item: dict) -> tuple[bool, float]:
    """
    Check if all items share at least one common occasion.
    
    Returns:
        (has_common_occasion: bool, penalty: float)
    """
    all_occasions = []
    
    # Include base item occasions
    base_occasions = infer_occasions(base_item)
    all_occasions.append(base_occasions)
    
    for item in items_by_slot.values():
        if item:
            all_occasions.append(infer_occasions(item))
    
    if len(all_occasions) < 2:
        return True, 0.0
    
    # Find intersection of all occasion sets
    common = all_occasions[0]
    for occasions in all_occasions[1:]:
        common = common & occasions
    
    if common:
        return True, 0.0
    
    # No common occasion - penalty
    return False, 0.15


# ============== FIX 3: Within-Outfit Diversity Control ==============

def get_color_family(color: str) -> str:
    """Get the color family (warm/cool/neutral) for a color."""
    if not color:
        return "neutral"
    
    color_lower = color.lower()
    
    for family, colors in COLOR_FAMILIES.items():
        if color_lower in colors:
            return family
    
    return "neutral"  # Unknown colors default to neutral


def check_within_outfit_diversity(items_by_slot: dict, base_item: dict) -> tuple[float, float]:
    """
    Check within-outfit diversity rules:
    - Penalty for multiple statement pieces
    - Bonus for color family harmony
    
    Returns:
        (penalty: float, bonus: float)
    """
    penalty = 0.0
    bonus = 0.0
    
    statement_count = 0
    colors = []
    
    # Include base item
    base_color = base_item.get("primary_color", "")
    if base_color:
        colors.append(base_color)
    
    base_name = base_item.get("name", "").lower()
    base_tags = base_item.get("style_tags") or []
    if "statement" in base_name or any("statement" in t.lower() for t in base_tags):
        statement_count += 1
    
    for item in items_by_slot.values():
        if not item:
            continue
        
        name_lower = item.get("name", "").lower()
        tags = item.get("style_tags") or []
        color = item.get("primary_color", "")
        
        # Count statements
        if "statement" in name_lower or any("statement" in t.lower() for t in tags):
            statement_count += 1
        
        # Collect colors
        if color:
            colors.append(color)
    
    # Penalty: More than 1 statement piece
    if statement_count > 1:
        penalty += 0.1 * (statement_count - 1)
    
    # Bonus: Color family harmony
    # Items in same family or complementary families get bonus
    if len(colors) >= 2:
        families = [get_color_family(c) for c in colors]
        neutral_count = families.count("neutral")
        non_neutral_families = set(f for f in families if f != "neutral")
        
        # Good: All neutrals or mostly neutrals + one accent family
        if neutral_count >= len(families) - 1:
            bonus += 0.05
        # Good: All same non-neutral family
        elif len(non_neutral_families) == 1:
            bonus += 0.03
        # Bad: Multiple non-neutral families competing
        elif len(non_neutral_families) > 1:
            penalty += 0.05 * (len(non_neutral_families) - 1)
    
    return penalty, bonus


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
    intent_vector: list = None,  # Fix 1: Unified outfit intent vector
    taste_vector: list = None,   # For personalization
    dislike_vector: list = None  # For avoiding dislikes
) -> dict:
    """
    Score a complete outfit using embedding similarity + direction bonuses - penalties.
    
    Enhanced with:
    - Fix 1: Intent vector - all items scored against unified outfit intent
    - Fix 2: Formality + occasion consistency
    - Fix 3: Within-outfit diversity control
    
    Returns dict with score breakdown and any violations.
    """
    # Check hard violations first
    violations = check_hard_violations(base_item, items_by_slot, direction)
    if violations:
        return {
            "total": -1.0,  # Disqualified
            "violations": violations,
            "breakdown": {}
        }
    
    # Fix 1: Compute intent vector if not provided
    base_emb = base_embedding or []
    if not intent_vector and base_emb:
        intent_vector = compute_outfit_intent_vector(
            base_emb, direction, taste_vector, dislike_vector
        )
    
    # Use intent vector for scoring (falls back to base_emb if no intent)
    scoring_vector = intent_vector if intent_vector else base_emb
    
    # Get embeddings (handle None items)
    bottom_item = items_by_slot.get("bottom")
    shoes_item = items_by_slot.get("shoes")
    acc_item = items_by_slot.get("accessory")
    layer_item = items_by_slot.get("layer")
    top_item = items_by_slot.get("top")
    
    bottom_emb = (bottom_item.get("embedding") if bottom_item else None) or []
    shoes_emb = (shoes_item.get("embedding") if shoes_item else None) or []
    acc_emb = (acc_item.get("embedding") if acc_item else None) or []
    layer_emb = (layer_item.get("embedding") if layer_item else None) or []
    top_emb = (top_item.get("embedding") if top_item else None) or []
    
    # Fix 1: Score all items against the INTENT vector (not just base)
    # This ensures thematic consistency across all pieces
    sim_intent_bottom = cosine_similarity(scoring_vector, bottom_emb) if bottom_emb else 0.5
    sim_intent_shoes = cosine_similarity(scoring_vector, shoes_emb) if shoes_emb else 0.5
    sim_intent_acc = cosine_similarity(scoring_vector, acc_emb) if acc_emb else 0.5
    sim_intent_layer = cosine_similarity(scoring_vector, layer_emb) if layer_emb else 0.0
    sim_intent_top = cosine_similarity(scoring_vector, top_emb) if top_emb else 0.0
    
    # Cross-item harmony (bottom-shoes still important)
    sim_bottom_shoes = cosine_similarity(bottom_emb, shoes_emb) if (bottom_emb and shoes_emb) else 0.5
    
    # Weighted similarity score - all items relative to intent
    sim_score = (
        0.35 * sim_intent_bottom +
        0.25 * sim_intent_shoes +
        0.15 * sim_intent_acc +
        0.10 * sim_intent_layer +
        0.10 * sim_intent_top +
        0.05 * sim_bottom_shoes  # Small bonus for bottom-shoes harmony
    )
    
    # Direction bonus
    direction_bonus = compute_direction_bonus(base_item, items_by_slot, direction)
    
    # Fix 2: Formality consistency
    formality_ok, formality_penalty = check_formality_consistency(items_by_slot, base_item)
    
    # Fix 2: Occasion consistency
    occasion_ok, occasion_penalty = check_occasion_consistency(items_by_slot, base_item)
    
    # Fix 3: Within-outfit diversity control
    diversity_penalty, harmony_bonus = check_within_outfit_diversity(items_by_slot, base_item)
    
    # Color harmony penalty (soft) - already existed but simplified
    colors = [base_item.get("primary_color", "")]
    for item in items_by_slot.values():
        if item:
            colors.append(item.get("primary_color", ""))
    
    non_neutrals = [c for c in colors if c and c not in NEUTRALS]
    color_penalty = 0.0
    if len(non_neutrals) > 2:
        color_penalty = 0.1 * (len(non_neutrals) - 2)
    
    # Total penalties
    total_penalty = color_penalty + formality_penalty + occasion_penalty + diversity_penalty
    total_bonus = direction_bonus + harmony_bonus
    
    total = sim_score + total_bonus - total_penalty
    
    return {
        "total": round(total, 3),
        "violations": [],
        "breakdown": {
            "sim_intent_weighted": round(sim_score, 3),
            "sim_bottom_shoes": round(sim_bottom_shoes, 3),
            "direction_bonus": round(direction_bonus, 3),
            "harmony_bonus": round(harmony_bonus, 3),
            "color_penalty": round(color_penalty, 3),
            "formality_penalty": round(formality_penalty, 3),
            "occasion_penalty": round(occasion_penalty, 3),
            "diversity_penalty": round(diversity_penalty, 3),
        }
    }


def generate_candidate_outfits(
    slots: list[str],
    candidates_by_slot: dict[str, list[dict]],
    max_candidates: int = 8
) -> list[dict[str, dict]]:
    """
    Generate multiple candidate outfit combinations from slot candidates.
    Uses simple combinatorial approach: vary each slot independently.
    
    Args:
        slots: List of slots to fill
        candidates_by_slot: Dict mapping slot -> list of candidate items
        max_candidates: Max number of outfit candidates to generate
    
    Returns:
        List of items_by_slot dicts (each is one candidate outfit)
    """
    import itertools
    
    # Get top candidates per slot (limit to avoid explosion)
    slot_options = []
    for slot in slots:
        options = candidates_by_slot.get(slot, [])[:3]  # Top 3 per slot
        if options:
            slot_options.append([(slot, opt) for opt in options])
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
    dislike_vector: list = None
) -> tuple[dict[str, dict], dict]:
    """
    Score all candidate outfits and return the best one.
    
    Args:
        candidate_outfits: List of outfit combinations to score
        base_item: User's input item
        direction: Style direction (Classic/Trendy/Bold)
        base_embedding: Embedding of base item
        taste_vector: User's taste preferences (Fix 1)
        dislike_vector: User's dislikes (Fix 1)
    
    Returns:
        (best_items_by_slot, score_details)
    """
    best_outfit = None
    best_score = -999
    best_details = {}
    
    # Fix 1: Compute intent vector once for all candidates
    intent_vector = compute_outfit_intent_vector(
        base_embedding, direction, taste_vector, dislike_vector
    ) if base_embedding else None
    
    for items_by_slot in candidate_outfits:
        score_result = score_outfit(
            base_item, items_by_slot, direction, 
            base_embedding=base_embedding,
            intent_vector=intent_vector,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
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
    dislike_vector: list = None
) -> dict:
    """
    Assemble a single outfit response with scoring.
    
    Args:
        direction: "Classic", "Trendy", or "Bold"
        base_item: The user's input item parsed tags
        items_by_slot: Dict mapping slot → selected catalog item
        base_embedding: Optional embedding for similarity scoring
        taste_vector: User's taste preferences (Fix 1)
        dislike_vector: User's dislikes (Fix 1)
    
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
            })
    
    # Score outfit with enhanced scoring (Fixes 1-3)
    intent_vector = compute_outfit_intent_vector(
        base_embedding, direction, taste_vector, dislike_vector
    ) if base_embedding else None
    
    score_result = score_outfit(
        base_item, items_by_slot, direction,
        base_embedding=base_embedding,
        intent_vector=intent_vector,
        taste_vector=taste_vector,
        dislike_vector=dislike_vector
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

