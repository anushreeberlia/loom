"""
Vector search and candidate retrieval logic.
"""

import os
import psycopg2
import httpx
from dotenv import load_dotenv

from services.outfit import (
    STYLE_DIRECTIONS, 
    NEUTRALS,
    build_base_item_text,
    get_preferred_colors,
    get_avoid_colors,
    is_layer_compatible
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"


# Slot-specific item type hints for better embedding search
SLOT_ITEM_HINTS = {
    "bottom": "women's skirt, jeans, trousers, or pants",
    "top": "women's blouse, shirt, t-shirt, or sweater",
    "shoes": "women's heels, flats, sneakers, or sandals",
    "layer": "women's jacket, cardigan, blazer, or coat",
    "accessory": "women's handbag, belt, scarf, or jewelry",
    "dress": "women's dress or gown",
}

# Item subtype keywords for diversity tracking
ITEM_SUBTYPE_KEYWORDS = {
    "bottom": ["skirt", "jeans", "trousers", "pants", "shorts", "capris", "leggings", "joggers"],
    "shoes": ["heels", "flats", "sneakers", "boots", "sandals", "flip flops", "loafers", "pumps", "wedges"],
    "accessory": ["bag", "handbag", "clutch", "belt", "watch", "earring", "necklace", "pendant", "bracelet", "scarf"],
    "layer": ["jacket", "blazer", "cardigan", "coat", "sweater", "hoodie", "vest"],
    "top": ["shirt", "blouse", "t-shirt", "top", "sweater", "tank", "tunic"],
}

# Exclusions - items to reject from certain slots
SLOT_EXCLUSIONS = {
    "bottom": ["stockings", "tights", "swimsuit", "bikini", "bra", "panty", "underwear", "lingerie"],
    "top": ["bra", "bikini", "swimsuit", "underwear", "lingerie"],
}

# Formality inference keywords
FORMALITY_DRESSY = {"heels", "pumps", "clutch", "blazer", "trousers", "pencil", "stilettos", "wedges"}
FORMALITY_CASUAL = {"sneakers", "flip flops", "joggers", "shorts", "t-shirt", "tank", "sandals", "flats"}

# Audience keywords
KIDS_KEYWORDS = {"girl's", "girls", "boy's", "boys", "kid", "kids", "baby", "toddler", "children"}

# Direction-specific accessory preferences
DIRECTION_ACCESSORY_PREFS = {
    "Classic": {"bag", "handbag", "watch", "belt"},
    "Trendy": {"bag", "handbag", "belt", "scarf"},
    "Bold": {"clutch", "jewelry", "necklace", "earring", "bracelet", "statement"},
}

# Semantic occasion contexts for intelligent filtering
# Each occasion has a "vibe" description and an "anti-vibe" description
OCCASION_SEMANTIC_CONTEXTS = {
    "work": {
        "vibe": "professional office business conservative modest appropriate polished refined tailored structured classic understated sophisticated practical comfortable sensible",
        "anti_vibe": "sexy revealing provocative clubbing nightlife glamorous flashy bold statement attention-grabbing mini skirt thigh high crop top low cut plunging neckline backless sheer see through animal print"
    },
    "casual": {
        "vibe": "relaxed comfortable everyday effortless laid-back weekend brunch daytime practical easy-going simple",
        "anti_vibe": "formal black-tie gala ballgown evening gown tuxedo ultra dressy"
    },
    "going-out": {
        "vibe": "glamorous sexy elegant party date night dinner chic statement bold eye-catching dressy stylish trendy",
        "anti_vibe": "athletic sporty gym workout sweatpants hoodie casual basic plain conservative modest office"
    },
    "smart-casual": {
        "vibe": "polished put-together elevated casual refined dinner date chic sophisticated understated elegant",
        "anti_vibe": "gym workout athletic sporty sweatpants loungewear pajamas ultra casual sloppy"
    },
    "workout": {
        "vibe": "athletic sporty gym fitness activewear performance breathable stretchy comfortable movement exercise training leggings sneakers",
        "anti_vibe": "formal dressy elegant heels business office work professional evening gown party blazer suit"
    },
    # "active" is returned by AI for workout moods - alias to workout
    "active": {
        "vibe": "athletic sporty gym fitness activewear performance breathable stretchy comfortable movement exercise training leggings sneakers",
        "anti_vibe": "formal dressy elegant heels business office work professional evening gown party blazer suit"
    }
}

# Cache for occasion embeddings (computed once per session)
_occasion_embedding_cache = {}


def get_occasion_embeddings(occasion: str) -> tuple[list[float], list[float]]:
    """
    Get or compute embeddings for an occasion's vibe and anti-vibe.
    Returns: (vibe_embedding, anti_vibe_embedding)
    """
    if occasion not in OCCASION_SEMANTIC_CONTEXTS:
        occasion = "casual"  # Default fallback
    
    cache_key = occasion
    if cache_key in _occasion_embedding_cache:
        return _occasion_embedding_cache[cache_key]
    
    context = OCCASION_SEMANTIC_CONTEXTS[occasion]
    embeddings = get_batch_embeddings([context["vibe"], context["anti_vibe"]])
    
    _occasion_embedding_cache[cache_key] = (embeddings[0], embeddings[1])
    return embeddings[0], embeddings[1]


def compute_occasion_score(item_embedding: list[float], occasion: str, item_tags: set = None) -> float:
    """
    Compute how well an item fits an occasion using semantic similarity + tag logic.
    
    Returns a score where:
    - Higher = better fit for the occasion
    - Items similar to anti-vibe get penalized
    - Items with mismatched occasion tags get penalized
    """
    import numpy as np
    
    vibe_emb, anti_vibe_emb = get_occasion_embeddings(occasion)
    
    item_emb = np.array(item_embedding)
    vibe = np.array(vibe_emb)
    anti_vibe = np.array(anti_vibe_emb)
    
    # Cosine similarities
    vibe_sim = np.dot(item_emb, vibe) / (np.linalg.norm(item_emb) * np.linalg.norm(vibe))
    anti_sim = np.dot(item_emb, anti_vibe) / (np.linalg.norm(item_emb) * np.linalg.norm(anti_vibe))
    
    # Base score from embeddings
    score = float(vibe_sim - anti_sim)
    
    # Tag-based adjustments (semantic backup)
    if item_tags:
        # Work occasion: penalize party/dinner/going-out items
        if occasion == "work":
            party_tags = {"party", "dinner", "date", "going-out", "night-out", "clubbing", "sexy", "glamorous", "statement"}
            if item_tags & party_tags:
                score -= 0.1  # Strong penalty
        
        # Going-out occasion: penalize work/office items
        elif occasion == "going-out":
            work_tags = {"work", "office", "business", "professional", "conservative"}
            if item_tags & work_tags:
                score -= 0.1
        
        # Workout occasion: only allow athletic items
        elif occasion == "workout":
            athletic_tags = {"sporty", "athletic", "activewear", "gym", "workout"}
            if not (item_tags & athletic_tags):
                score -= 0.15  # Very strong penalty for non-athletic
    
    return score


def filter_by_occasion_semantic(candidates: list[dict], occasion: str, threshold: float = -0.02) -> list[dict]:
    """
    Filter and rank candidates by semantic occasion fit.
    
    Args:
        candidates: List of items with embeddings
        occasion: The target occasion (work, casual, going-out, etc.)
        threshold: Minimum occasion score to include (items below this are filtered out)
    
    Returns:
        Filtered and sorted candidates (best fits first)
    """
    if occasion not in OCCASION_SEMANTIC_CONTEXTS:
        return candidates  # No filtering if unknown occasion
    
    if not candidates:
        return candidates
    
    # Score all candidates
    scored = []
    filtered_out = []
    for c in candidates:
        # Gather all item tags for semantic + tag-based scoring
        item_tags = set((c.get("occasion_tags") or []) + (c.get("style_tags") or []))
        
        if not c.get("embedding"):
            c["_occasion_score"] = 0  # No embedding = neutral
            scored.append(c)
        else:
            score = compute_occasion_score(c["embedding"], occasion, item_tags)
            c["_occasion_score"] = score
            if score >= threshold:
                scored.append(c)
            else:
                filtered_out.append(c)
    
    # Sort by occasion score (best fits first)
    scored.sort(key=lambda x: x.get("_occasion_score", 0), reverse=True)
    
    # If we filtered out everything, return at least some items (sorted by score)
    if not scored and filtered_out:
        filtered_out.sort(key=lambda x: x.get("_occasion_score", 0), reverse=True)
        return filtered_out[:3]  # Return top 3 even if below threshold
    
    return scored


def infer_product_type(item_name: str, slot: str) -> dict:
    """
    Infer product type attributes from item name.
    Returns: {"subtype": "skirt", "formality": "dressy", "audience": "women"}
    """
    name_lower = item_name.lower()
    result = {"subtype": None, "formality": None, "audience": "women"}
    
    # Get subtype
    for keyword in ITEM_SUBTYPE_KEYWORDS.get(slot, []):
        if keyword in name_lower:
            result["subtype"] = keyword
            break
    
    # Infer formality
    if any(kw in name_lower for kw in FORMALITY_DRESSY):
        result["formality"] = "dressy"
    elif any(kw in name_lower for kw in FORMALITY_CASUAL):
        result["formality"] = "casual"
    else:
        result["formality"] = "neutral"
    
    # Infer audience
    if any(kw in name_lower for kw in KIDS_KEYWORDS):
        result["audience"] = "kids"
    
    return result


def is_neutral_color(color: str) -> bool:
    """Check if a color is neutral."""
    return color in NEUTRALS if color else False


def extract_item_subtype(item_name: str, slot: str) -> str | None:
    """Extract the subtype of an item from its name."""
    name_lower = item_name.lower()
    keywords = ITEM_SUBTYPE_KEYWORDS.get(slot, [])
    
    for keyword in keywords:
        if keyword in name_lower:
            return keyword
    return None


def filter_by_subtype_diversity(
    candidates: list[dict], 
    slot: str, 
    used_subtypes: set[str]
) -> list[dict]:
    """
    Reorder candidates to prefer different subtypes than already used.
    Items with unused subtypes come first.
    """
    if not used_subtypes:
        return candidates
    
    # Split into preferred (different subtype) and fallback (same subtype)
    preferred = []
    fallback = []
    
    for c in candidates:
        subtype = extract_item_subtype(c.get("name", ""), slot)
        if subtype and subtype in used_subtypes:
            fallback.append(c)
        else:
            preferred.append(c)
    
    return preferred + fallback


def apply_direction_rerank(
    candidates: list[dict],
    direction: str,
    slot: str,
    base_item: dict = None,
    chosen_items: dict[str, dict] = None
) -> list[dict]:
    """
    Apply direction-specific, slot-aware reranking rules.
    
    - Trendy: avoid beige-on-beige, prefer variety
    - Classic: prefer neutrals (more for shoes/bags, less for bottoms)
    - Bold: prefer contrast or statement pieces (not just "more color")
    """
    if not candidates:
        return candidates
    
    chosen_items = chosen_items or {}
    base_item = base_item or {}
    
    # Get colors for contrast checking
    base_color = base_item.get("primary_color", "")
    bottom = chosen_items.get("bottom")
    bottom_color = bottom.get("primary_color") if bottom else None
    
    # Is base item already bright/colorful?
    base_is_bright = base_color and not is_neutral_color(base_color)
    
    def score_candidate(c: dict) -> float:
        score = 0.0
        color = c.get("primary_color", "")
        name_lower = c.get("name", "").lower()
        style_tags = c.get("style_tags") or []
        is_statement = "statement" in name_lower or any("statement" in t.lower() for t in style_tags)
        
        if direction == "Trendy":
            # Anti-beige-on-beige for shoes
            if slot == "shoes" and bottom_color:
                if is_neutral_color(bottom_color) and is_neutral_color(color):
                    if bottom_color == color:
                        score -= 0.5  # Heavy penalty for same neutral
                    elif color in {"white", "black", "metallic"}:
                        score += 0.2  # Prefer contrast neutrals/metallics
            
            # Prefer trendy items
            if is_statement:
                score += 0.1
                
        elif direction == "Classic":
            # Slot-aware neutral preference
            if is_neutral_color(color):
                if slot in {"shoes", "accessory"}:
                    score += 0.25  # Strong neutral preference for shoes/accessories
                elif slot == "bottom":
                    score += 0.15  # Moderate for bottoms
                    if "denim" in name_lower or color == "navy":
                        score += 0.05  # Bonus for classic bottom types
                else:
                    score += 0.2
            
            # Prefer classic item types
            if any(w in name_lower for w in ["classic", "structured", "tailored"]):
                score += 0.1
                
        elif direction == "Bold":
            # Bold = contrast OR statement, not just "more color"
            if base_is_bright:
                # Base is already colorful - prefer contrast (neutrals or complementary)
                if color in {"black", "white"}:
                    score += 0.2  # High contrast neutrals
                elif is_neutral_color(color):
                    score += 0.1  # Other neutrals for grounding
            else:
                # Base is neutral - bold items can add color
                if not is_neutral_color(color) and color != "unknown":
                    score += 0.2
            
            # Always prefer statement pieces for Bold
            if is_statement:
                score += 0.25
            
            # Dressier silhouettes for Bold
            if any(w in name_lower for w in ["structured", "tailored", "heel", "clutch"]):
                score += 0.1
        
        return score
    
    # Sort by score (higher is better), maintaining relative order for ties
    scored = [(c, score_candidate(c)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    
    return [c for c, _ in scored]


def fix_trendy_same_neutral(
    items_by_slot: dict[str, dict],
    all_candidates: dict[str, list[dict]]
) -> dict[str, dict]:
    """
    Fix Trendy outfit if bottom and shoes are same neutral color.
    Swap shoes for a different neutral, metallic, or white/black.
    
    Priority: metallic > white/black > different neutral > keep original
    """
    bottom = items_by_slot.get("bottom")
    shoes = items_by_slot.get("shoes")
    
    if not bottom or not shoes:
        return items_by_slot
    
    bottom_color = bottom.get("primary_color", "")
    shoes_color = shoes.get("primary_color", "")
    
    # Check if both are same neutral
    if not (is_neutral_color(bottom_color) and 
            is_neutral_color(shoes_color) and 
            bottom_color == shoes_color):
        return items_by_slot
    
    # Find alternative shoes - prioritize by preference
    shoe_candidates = all_candidates.get("shoes", [])
    current_shoe_id = shoes.get("id")
    
    best_alt = None
    best_priority = -1
    
    for alt_shoe in shoe_candidates:
        if alt_shoe.get("id") == current_shoe_id:
            continue
            
        alt_color = alt_shoe.get("primary_color", "")
        alt_name = alt_shoe.get("name", "").lower()
        
        # Skip if same color as bottom
        if alt_color == bottom_color:
            continue
        
        # Priority scoring
        priority = 0
        if "metallic" in alt_name or alt_color == "metallic":
            priority = 3  # Highest: metallic escape hatch
        elif alt_color in {"white", "black"}:
            priority = 2  # High contrast neutrals
        elif is_neutral_color(alt_color):
            priority = 1  # Different neutral
        else:
            priority = 0  # Non-neutral (still okay)
        
        if priority > best_priority:
            best_priority = priority
            best_alt = alt_shoe
    
    if best_alt:
        items_by_slot["shoes"] = best_alt
    
    return items_by_slot

# Keywords to exclude from results (sanity gate)
FORBIDDEN_KEYWORDS = {
    "swimsuit", "swimwear", "bikini", "swim", 
    "hosiery", "stockings", "tights", "socks",
    "girl's", "girls", "kid", "kids", "children", "boy",
    "dupatta", "innerwear", "underwear", "bra", "lingerie",
    "sleepwear", "nightwear", "pyjama", "pajama",
}


def build_query_text(
    base_item: dict, 
    direction: str, 
    slot: str,
    chosen_items: dict[str, dict] = None
) -> str:
    """
    Build a direction-aware query text for embedding.
    Uses sequential conditioning - includes already chosen items.
    
    Args:
        base_item: User's input item
        direction: Style direction
        slot: Slot to fill
        chosen_items: Items already chosen for this outfit (slot → item)
    """
    base_color = base_item.get("primary_color", "")
    base_category = base_item.get("category", "top")
    style = " ".join((base_item.get("style_tags") or [])[:2])
    
    direction_lower = direction.lower()
    dir_info = STYLE_DIRECTIONS.get(direction, {})
    color_policy = dir_info.get("color_policy", "neutrals")
    
    # Get slot-specific item type hint
    item_hint = SLOT_ITEM_HINTS.get(slot, f"women's {slot}")
    
    # Build color guidance based on policy
    if color_policy == "neutrals":
        color_hint = "in neutral colors like black, white, gray, or beige"
    elif color_policy == "contrast":
        color_hint = f"in contrasting colors that complement {base_color}"
    else:  # two_tone
        color_hint = "in neutral or one accent color"
    
    # Slot-specific color overrides
    if slot == "shoes":
        color_hint = "in neutral colors like black, white, or beige"
    elif slot == "accessory":
        color_hint = "in neutral or metallic tones"
    
    # Build base query
    base_query = f"{direction_lower} {item_hint} {color_hint}"
    
    # Add sequential conditioning - reference already chosen items
    chosen_items = chosen_items or {}
    context_parts = [f"{base_color} {style} {base_category}"]
    
    for chosen_slot, chosen_item in chosen_items.items():
        if chosen_item:
            chosen_color = chosen_item.get("primary_color", "")
            chosen_name = chosen_item.get("name", "").split()[-1]  # Last word usually describes item
            context_parts.append(f"{chosen_color} {chosen_slot}")
    
    context = " + ".join(context_parts)
    
    return f"{base_query} to complete an outfit with {context}"


def passes_sanity_check(item: dict, slot: str = None) -> bool:
    """
    Check if an item passes the sanity gate.
    
    Checks:
    1. No forbidden keywords (swimwear, underwear, etc.)
    2. No slot-specific exclusions (stockings in bottom slot)
    3. Audience must be women (not kids)
    """
    name = item.get("name", "").lower()
    
    # Global forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in name:
            return False
    
    # Slot-specific exclusions
    if slot and slot in SLOT_EXCLUSIONS:
        for keyword in SLOT_EXCLUSIONS[slot]:
            if keyword in name:
                return False
    
    # Audience check - reject kids items
    product_info = infer_product_type(item.get("name", ""), slot or "")
    if product_info.get("audience") == "kids":
        return False
    
    return True


def filter_candidates(candidates: list[dict], slot: str = None) -> list[dict]:
    """Filter out items that fail sanity check for the given slot."""
    return [c for c in candidates if passes_sanity_check(c, slot)]


def get_query_embedding(text: str) -> list[float]:
    """Embed query text using OpenAI."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")
    
    data = response.json()
    return data["data"][0]["embedding"]


def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call (much faster than individual calls)."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    if not texts:
        return []
    
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts  # Batch all texts in one call
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")
    
    data = response.json()
    # Sort by index to maintain order (API may return out of order)
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def retrieve_candidates(
    category: str, 
    query_embedding: list[float], 
    k: int = 20,
    exclude_ids: list[int] = None,
    avoid_colors: set[str] = None,
    prefer_colors: set[str] = None,
    source: str = None,
    use_closet: bool = False,
    user_id: str = "default"
) -> list[dict]:
    """
    Vector search for candidates in a category with color filtering.
    
    Args:
        category: Item category to filter by
        query_embedding: Query vector for similarity search
        k: Number of candidates to return
        exclude_ids: Item IDs to exclude
        avoid_colors: Colors to filter out (hard constraint)
        prefer_colors: Colors to prefer (soft boost in scoring)
        source: Catalog source to filter by (e.g., 'h_and_m', 'kaggle_fashion') - catalog only
        use_closet: If True, search user_closet_items instead of catalog_items
        user_id: User ID for closet filtering (only used if use_closet=True)
    
    Returns:
        List of items sorted by similarity + color preference
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    exclude_ids = exclude_ids or []
    avoid_colors = avoid_colors or set()
    
    # Choose table and columns based on mode
    if use_closet:
        table = "user_closet_items"
        select_cols = "id, name, image_url, NULL as product_url, primary_color, style_tags"
    else:
        table = "catalog_items"
        select_cols = "id, name, image_url, product_url, primary_color, style_tags"
    
    # Build dynamic WHERE clause
    where_conditions = ["category = %s", "embedding IS NOT NULL"]
    params = [query_embedding, category]
    
    # Closet mode: filter by user_id
    if use_closet:
        where_conditions.append("user_id = %s")
        params.append(user_id)
    # Catalog mode: filter by source
    elif source:
        where_conditions.append("source = %s")
        params.append(source)
    
    if exclude_ids:
        where_conditions.append("id != ALL(%s)")
        params.append(exclude_ids)
    
    if avoid_colors:
        where_conditions.append("(primary_color IS NULL OR primary_color != ALL(%s))")
        params.append(list(avoid_colors))
    
    params.append(k * 2)  # LIMIT
    
    query = f"""
        SELECT {select_cols},
               embedding::text as embedding_text,
               embedding <-> %s::vector as distance
        FROM {table}
        WHERE {' AND '.join(where_conditions)}
        ORDER BY distance
        LIMIT %s
    """
    
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    columns = ["id", "name", "image_url", "product_url", "primary_color", "style_tags", "embedding_text", "distance"]
    candidates = []
    for row in rows:
        item = dict(zip(columns, row))
        # Parse embedding from pgvector text format: "[0.1,0.2,...]"
        emb_text = item.pop("embedding_text", None)
        if emb_text:
            # Remove brackets and split by comma
            item["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")]
        else:
            item["embedding"] = []
        candidates.append(item)
    
    # Apply soft scoring for preferred colors
    if prefer_colors:
        for c in candidates:
            color = c.get("primary_color")
            # Boost score for preferred colors (lower distance = better)
            if color and color in prefer_colors:
                c["distance"] = c["distance"] * 0.85  # 15% boost
            elif color and color in NEUTRALS:
                c["distance"] = c["distance"] * 0.9   # 10% boost for neutrals
        
        # Re-sort by adjusted distance
        candidates.sort(key=lambda x: x["distance"])
    
    return candidates[:k]


# Slot to category mapping (for slots that don't have their own category)
SLOT_CATEGORY_MAP = {
    "layer": "top",  # Layers (jackets, cardigans) are stored under "top" category
}

# Keywords to filter layer items from tops
LAYER_KEYWORDS = {"jacket", "cardigan", "blazer", "coat", "vest", "shrug", "kimono", "poncho"}


def filter_layer_items(candidates: list[dict]) -> list[dict]:
    """Filter candidates to only include layer-like items (jackets, cardigans, etc.)."""
    filtered = []
    for c in candidates:
        name_lower = c.get("name", "").lower()
        if any(keyword in name_lower for keyword in LAYER_KEYWORDS):
            filtered.append(c)
    return filtered


def retrieve_for_slot(
    base_item: dict, 
    direction: str, 
    slot: str, 
    exclude_ids: list[int] = None,
    chosen_items: dict[str, dict] = None,
    used_subtypes: set[str] = None,
    k: int = 5,
    source: str = None,
    precomputed_embedding: list[float] = None,
    use_closet: bool = False,
    user_id: str = "default",
    occasion: str = None,  # Semantic occasion filtering (work, casual, going-out, etc.)
    prefer_occasions: list[str] = None,  # Legacy - kept for compatibility
    avoid_occasions: list[str] = None  # Legacy - kept for compatibility
) -> list[dict]:
    """
    Retrieve candidate items for a specific slot and direction.
    Uses sequential conditioning - considers already chosen items.
    Applies color diversity rules, sanity filtering, and subtype diversity.
    
    Args:
        base_item: User's input item
        direction: Style direction
        slot: Slot to fill
        exclude_ids: IDs to exclude (diversity across outfits)
        chosen_items: Items already chosen FOR THIS OUTFIT (sequential conditioning)
        used_subtypes: Item subtypes already used in previous outfits (e.g., {"skirt", "heels"})
        k: Number of candidates to return
        source: Catalog source to filter by (e.g., 'h_and_m', 'kaggle_fashion') - catalog only
        precomputed_embedding: Pre-computed query embedding (skips API call if provided)
        use_closet: If True, search user's closet instead of catalog
        user_id: User ID for closet filtering (only used if use_closet=True)
    """
    base_color = base_item.get("primary_color", "unknown")
    
    # Get color preferences for this direction and slot
    avoid_colors = get_avoid_colors(direction, base_color, slot)
    prefer_colors = get_preferred_colors(direction, base_color, slot)
    
    # For closet mode: don't hard-filter colors (small wardrobes are too restrictive)
    # Instead, make avoid_colors a soft preference (penalized in scoring, not excluded)
    if use_closet:
        # Move avoid_colors to soft penalty instead of hard filter
        avoid_colors_soft = avoid_colors
        avoid_colors = set()  # Don't hard-exclude any colors
    else:
        avoid_colors_soft = set()
    
    # Use precomputed embedding or compute new one
    if precomputed_embedding:
        query_embedding = precomputed_embedding
    else:
        query_text = build_query_text(base_item, direction, slot, chosen_items)
        query_embedding = get_query_embedding(query_text)
    
    # Map slot to category if needed (e.g., "layer" -> "top")
    # For closet, use exact category (user categorized their items)
    search_category = slot if use_closet else SLOT_CATEGORY_MAP.get(slot, slot)
    
    # Get more candidates than needed, then filter
    candidates = retrieve_candidates(
        category=search_category,
        query_embedding=query_embedding,
        k=k * 6 if slot == "layer" else k * 4,  # Get extra for layer filtering
        exclude_ids=exclude_ids,
        avoid_colors=avoid_colors,
        prefer_colors=prefer_colors,
        source=source,
        use_closet=use_closet,
        user_id=user_id
    )
    
    # For closet: apply soft color penalty (prefer non-matching, but don't exclude)
    if use_closet and avoid_colors_soft:
        for c in candidates:
            if c.get("primary_color") in avoid_colors_soft:
                c["_color_penalty"] = True  # Mark for soft scoring penalty
    
    # Apply SEMANTIC occasion filtering (intelligent, not keyword-based)
    if occasion and occasion in OCCASION_SEMANTIC_CONTEXTS:
        candidates = filter_by_occasion_semantic(candidates, occasion, threshold=-0.03)
    # Legacy keyword filtering (fallback if occasion not provided but prefer/avoid are)
    elif avoid_occasions or prefer_occasions:
        avoid_set = set(avoid_occasions or [])
        prefer_set = set(prefer_occasions or [])
        filtered = []
        for c in candidates:
            item_tags = set((c.get("occasion_tags") or []) + (c.get("style_tags") or []))
            if avoid_set and item_tags & avoid_set:
                continue
            if prefer_set and (item_tags & prefer_set or "everyday" in item_tags):
                c["_occasion_boost"] = True
            filtered.append(c)
        candidates = sorted(filtered, key=lambda x: (1 if x.get("_occasion_boost") else 0), reverse=True)
    
    # For layer slot in catalog mode, filter to only include layer-like items
    # (closet items are already properly categorized by user)
    if slot == "layer" and not use_closet:
        candidates = filter_layer_items(candidates)
    
    # For layers: filter by material weight compatibility with chosen top
    if slot == "layer" and chosen_items:
        top = chosen_items.get("top")
        if top:
            compatible = []
            for c in candidates:
                if is_layer_compatible(c, top):
                    compatible.append(c)
            if compatible:  # Only filter if we have options left
                candidates = compatible
    
    # For tops when base is a layer: top should be lighter than the layer
    # (you don't wear a chunky sweater under a light cardigan)
    if slot == "top" and base_item.get("category") == "layer":
        compatible = []
        for c in candidates:
            # Reverse check: layer (base) should be heavier than top (candidate)
            if is_layer_compatible(base_item, c):
                compatible.append(c)
        if compatible:
            candidates = compatible
    
    # Apply sanity gate (slot-aware)
    candidates = filter_candidates(candidates, slot)
    
    # Apply subtype diversity - prefer different item types across outfits
    if used_subtypes:
        candidates = filter_by_subtype_diversity(candidates, slot, used_subtypes)
    
    # Apply direction-specific reranking (slot-aware, base-aware for Bold)
    candidates = apply_direction_rerank(candidates, direction, slot, base_item, chosen_items)
    
    return candidates[:k]
