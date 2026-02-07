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
    base_embedding: list = None
) -> dict:
    """
    Score a complete outfit using embedding similarity + direction bonuses - penalties.
    
    Score formula:
      0.45 * sim(base, bottom)
    + 0.30 * sim(base, shoes)
    + 0.15 * sim(base, accessory)
    + 0.10 * sim(bottom, shoes)
    + direction_bonus
    - penalties
    
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
    
    # Get embeddings (handle None items)
    base_emb = base_embedding or []
    bottom_item = items_by_slot.get("bottom")
    shoes_item = items_by_slot.get("shoes")
    acc_item = items_by_slot.get("accessory")
    
    bottom_emb = (bottom_item.get("embedding") if bottom_item else None) or []
    shoes_emb = (shoes_item.get("embedding") if shoes_item else None) or []
    acc_emb = (acc_item.get("embedding") if acc_item else None) or []
    
    # Compute similarities
    sim_base_bottom = cosine_similarity(base_emb, bottom_emb) if bottom_emb else 0.5
    sim_base_shoes = cosine_similarity(base_emb, shoes_emb) if shoes_emb else 0.5
    sim_base_acc = cosine_similarity(base_emb, acc_emb) if acc_emb else 0.5
    sim_bottom_shoes = cosine_similarity(bottom_emb, shoes_emb) if (bottom_emb and shoes_emb) else 0.5
    
    # Weighted similarity score
    sim_score = (
        0.45 * sim_base_bottom +
        0.30 * sim_base_shoes +
        0.15 * sim_base_acc +
        0.10 * sim_bottom_shoes
    )
    
    # Direction bonus
    direction_bonus = compute_direction_bonus(base_item, items_by_slot, direction)
    
    # Color harmony penalty (soft)
    colors = [base_item.get("primary_color", "")]
    for item in items_by_slot.values():
        if item:
            colors.append(item.get("primary_color", ""))
    
    non_neutrals = [c for c in colors if c and c not in NEUTRALS]
    color_penalty = 0.0
    if len(non_neutrals) > 2:
        color_penalty = 0.1 * (len(non_neutrals) - 2)
    
    # Formality mismatch penalty (soft)
    formality_penalty = 0.0
    formalities = []
    for item in items_by_slot.values():
        if not item:
            continue
        name_lower = item.get("name", "").lower()
        if any(kw in name_lower for kw in DRESSY_KEYWORDS):
            formalities.append("dressy")
        elif any(kw in name_lower for kw in CASUAL_KEYWORDS):
            formalities.append("casual")
    
    if "dressy" in formalities and "casual" in formalities:
        formality_penalty = 0.15
    
    total = sim_score + direction_bonus - color_penalty - formality_penalty
    
    return {
        "total": round(total, 3),
        "violations": [],
        "breakdown": {
            "sim_base_bottom": round(sim_base_bottom, 3),
            "sim_base_shoes": round(sim_base_shoes, 3),
            "sim_base_acc": round(sim_base_acc, 3),
            "sim_bottom_shoes": round(sim_bottom_shoes, 3),
            "sim_weighted": round(sim_score, 3),
            "direction_bonus": round(direction_bonus, 3),
            "color_penalty": round(color_penalty, 3),
            "formality_penalty": round(formality_penalty, 3),
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
    base_embedding: list = None
) -> tuple[dict[str, dict], dict]:
    """
    Score all candidate outfits and return the best one.
    
    Returns:
        (best_items_by_slot, score_details)
    """
    best_outfit = None
    best_score = -999
    best_details = {}
    
    for items_by_slot in candidate_outfits:
        score_result = score_outfit(base_item, items_by_slot, direction, base_embedding)
        
        if score_result["total"] > best_score:
            best_score = score_result["total"]
            best_outfit = items_by_slot
            best_details = score_result
    
    return best_outfit or candidate_outfits[0], best_details


def assemble_outfit(
    direction: str, 
    base_item: dict, 
    items_by_slot: dict[str, dict],
    base_embedding: list = None
) -> dict:
    """
    Assemble a single outfit response with scoring.
    
    Args:
        direction: "Classic", "Trendy", or "Bold"
        base_item: The user's input item parsed tags
        items_by_slot: Dict mapping slot → selected catalog item
        base_embedding: Optional embedding for similarity scoring
    
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
    
    # Score outfit
    score_result = score_outfit(base_item, items_by_slot, direction, base_embedding)
    
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

