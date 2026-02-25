"""
Vector search and candidate retrieval logic.
"""

import os
import logging
import psycopg2
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from services.outfit import (
    STYLE_DIRECTIONS, 
    NEUTRALS,
    build_base_item_text,
    get_preferred_colors,
    get_avoid_colors,
    is_layer_compatible
)
from services.fashion_clip import embed_text as _clip_embed_text, embed_texts as _clip_embed_texts

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


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
        "vibe": "professional office business conservative modest polished refined tailored structured classic understated sophisticated blazer trousers slacks pencil skirt midi skirt knee-length button-down blouse loafers oxford pumps closed-toe cardigan tote bag collared shirt dress pants wool cotton structured handbag",
        "anti_vibe": "sexy revealing provocative clubbing nightlife party halter tank top cami camisole strappy spaghetti strap tube top bandeau sweatshirt hoodie athletic sporty gym workout mini skirt mini dress short skirt short shorts hot pants cargo shorts denim shorts cutoffs crop top graphic tee cropped low cut plunging neckline backless sheer see-through animal print leopard ripped distressed sneakers platform boots thigh-high boots stiletto heels chunky jewelry casual weekend lounge beach swimwear festival rave college university logo"
    },
    "casual": {
        "vibe": "relaxed comfortable everyday effortless laid-back weekend brunch daytime jeans t-shirt sneakers denim jacket hoodie tote bag flats sandals cotton linen shorts sundress canvas",
        "anti_vibe": "formal black-tie gala ballgown evening gown tuxedo sequin gown stiletto heels satin dress pearl necklace cufflinks bow tie blazer suit pencil skirt"
    },
    "going-out": {
        "vibe": "glamorous sexy elegant party date night dinner chic statement bold eye-catching dressy heels mini dress bodycon satin silk velvet lace sequins clutch bag strappy sandals statement earrings cocktail dress",
        "anti_vibe": "athletic sporty gym workout sweatpants hoodie sneakers running shoes t-shirt joggers leggings activewear baseball cap flip flops crocs backpack"
    },
    "smart-casual": {
        "vibe": "polished put-together elevated casual refined dinner date chic sophisticated understated elegant blazer chinos loafers ankle boots midi skirt knit sweater leather bag structured dress",
        "anti_vibe": "gym workout athletic sporty sweatpants joggers loungewear pajamas hoodie graphic tee flip flops crocs running shoes baseball cap ripped jeans"
    },
    "workout": {
        "vibe": "athletic sporty gym fitness activewear performance breathable stretchy leggings sports bra sneakers running shoes tank top shorts joggers headband water bottle duffel bag",
        "anti_vibe": "formal dressy elegant heels blazer suit pencil skirt silk blouse leather shoes loafers evening gown sequins satin clutch bag jewelry watch"
    }
}

# Cache for occasion embeddings (computed once per session)
_occasion_embedding_cache = {}
# Cache for mood text embeddings
_mood_embedding_cache = {}


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


def get_mood_embedding(mood_text: str) -> list[float]:
    """
    Embed raw mood text directly for semantic matching.
    This allows any mood description to work without mapping to predefined occasions.
    
    Args:
        mood_text: Free-form mood like "beach day", "funeral", "wedding guest"
        
    Returns:
        Embedding vector for the mood
    """
    if mood_text in _mood_embedding_cache:
        return _mood_embedding_cache[mood_text]
    
    # Create an outfit-focused description for better matching
    outfit_context = f"outfit for {mood_text} - clothing style appropriate for this occasion"
    embeddings = get_batch_embeddings([outfit_context])
    
    _mood_embedding_cache[mood_text] = embeddings[0]
    return embeddings[0]


_direct_mood_cache: dict[str, list[float]] = {}

def get_direct_mood_embedding(mood_text: str) -> list[float]:
    """Raw mood text embedding — used for tag-level scoring."""
    if mood_text in _direct_mood_cache:
        return _direct_mood_cache[mood_text]
    emb = get_batch_embeddings([mood_text])[0]
    _direct_mood_cache[mood_text] = emb
    return emb


_tag_embedding_cache: dict[str, list[float]] = {}

def _get_tag_embedding(tag: str) -> list[float]:
    if tag not in _tag_embedding_cache:
        _tag_embedding_cache[tag] = get_batch_embeddings([tag])[0]
    return _tag_embedding_cache[tag]


def compute_tag_mood_score(style_tags: list[str], mood_text: str) -> float:
    """
    Best individual style-tag similarity to the mood.
    
    Full item descriptions ("gray sporty fitted polyester top") all score the
    same against moods because CLIP averages the signal away. But individual
    tags separate cleanly: cosine("sporty","workout")=0.69 vs
    cosine("casual","workout")=0.64 — enough gap to rank correctly.
    
    Only uses style_tags (not occasion_tags like "everyday" which inflate
    casual scores).
    """
    import numpy as np

    if not style_tags:
        return 0.0

    mood_emb = np.array(get_direct_mood_embedding(mood_text))
    best_sim = 0.0
    for tag in style_tags:
        tag_emb = np.array(_get_tag_embedding(tag))
        sim = float(np.dot(tag_emb, mood_emb) / (np.linalg.norm(tag_emb) * np.linalg.norm(mood_emb)))
        if sim > best_sim:
            best_sim = sim
    return best_sim


def compute_occasion_score(item_embedding: list[float], occasion: str = None, 
                          mood_text: str = None, item_tags: set = None) -> float:
    """
    Compute how well an item fits an occasion using semantic similarity.
    
    Can use either:
    - occasion: A predefined occasion (work, casual, going-out, etc.)
    - mood_text: Raw mood text that gets embedded directly (more flexible)
    
    Returns a score where higher = better fit.
    """
    import numpy as np
    
    item_emb = np.array(item_embedding)
    
    # If mood_text provided, use direct embedding comparison (more flexible)
    if mood_text:
        mood_emb = np.array(get_mood_embedding(mood_text))
        # Simple cosine similarity - items similar to the mood score higher
        score = float(np.dot(item_emb, mood_emb) / (np.linalg.norm(item_emb) * np.linalg.norm(mood_emb)))
        return score
    
    # Otherwise use predefined occasion vibe/anti-vibe
    if not occasion:
        occasion = "casual"
        
    vibe_emb, anti_vibe_emb = get_occasion_embeddings(occasion)
    
    vibe = np.array(vibe_emb)
    anti_vibe = np.array(anti_vibe_emb)
    
    # Cosine similarities
    vibe_sim = np.dot(item_emb, vibe) / (np.linalg.norm(item_emb) * np.linalg.norm(vibe))
    anti_sim = np.dot(item_emb, anti_vibe) / (np.linalg.norm(item_emb) * np.linalg.norm(anti_vibe))
    
    # Score = vibe similarity - anti_vibe penalty
    # Work needs stronger filtering - casual/revealing items should be heavily penalized
    score = float(vibe_sim)
    
    if occasion == "work":
        # Work is strict - penalize anti-vibe items heavily
        # Even items somewhat similar to anti-vibe should be deprioritized
        score -= anti_sim * 2.0  # Strong penalty for any anti-vibe similarity
        # Extra penalty if more similar to anti than vibe
        if anti_sim > vibe_sim:
            score -= (anti_sim - vibe_sim) * 3.0
    elif occasion in ("going-out", "smart-casual"):
        excess_anti = max(0, anti_sim - vibe_sim)
        score -= excess_anti * 3
    else:
        excess_anti = max(0, anti_sim - vibe_sim)
        score -= excess_anti
    
    if item_tags and occasion == "workout":
        athletic_tags = {"sporty", "athletic", "activewear", "gym", "workout"}
        if not (item_tags & athletic_tags):
            score -= 0.15
    
    return score


def filter_by_occasion_semantic(candidates: list[dict], occasion: str = None, 
                                mood_text: str = None, threshold: float = -0.02) -> list[dict]:
    """
    Filter and rank candidates by semantic occasion fit.
    Uses relative filtering: keeps top half of scored candidates.
    """
    if not mood_text and occasion not in OCCASION_SEMANTIC_CONTEXTS:
        return candidates
    
    if not candidates:
        return candidates
    
    # Score all candidates
    all_scored = []
    for c in candidates:
        item_tags = set((c.get("occasion_tags") or []) + (c.get("style_tags") or []))
        
        if not c.get("embedding"):
            c["_occasion_score"] = 0
        else:
            score = compute_occasion_score(c["embedding"], occasion=occasion, 
                                          mood_text=mood_text, item_tags=item_tags)
            c["_occasion_score"] = score
        all_scored.append(c)
    
    all_scored.sort(key=lambda x: x.get("_occasion_score", 0), reverse=True)
    
    if len(all_scored) <= 3:
        return all_scored
    
    # Keep items within threshold of the best score
    # Work is stricter - need higher percentage of best score
    best = all_scored[0].get("_occasion_score", 0)
    if best > 0:
        # Work occasion requires stricter filtering (85% of best)
        # Other occasions use looser filtering (70% of best)
        cutoff_pct = 0.85 if occasion == "work" else 0.7
        cutoff = best * cutoff_pct
        filtered = [c for c in all_scored if c.get("_occasion_score", 0) >= cutoff]
        
        # Log for debugging
        logger.info(f"Occasion filter ({occasion}): best={best:.3f}, cutoff={cutoff:.3f}, kept {len(filtered)}/{len(all_scored)}")
        
        return filtered if len(filtered) >= 3 else all_scored[:3]
    
    return all_scored


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
    chosen_items: dict[str, dict] = None,
    occasion: str = None,
    mood_text: str = None
) -> str:
    """
    Build a direction-aware query text for embedding.
    Uses sequential conditioning - includes already chosen items.
    
    Args:
        base_item: User's input item
        direction: Style direction
        slot: Slot to fill
        chosen_items: Items already chosen for this outfit (slot → item)
        occasion: Optional predefined occasion (work, casual, going-out)
        mood_text: Raw mood text like "beach day", "funeral" - uses directly if provided
    """
    base_color = base_item.get("primary_color", "")
    base_category = base_item.get("category", "top")
    style = " ".join((base_item.get("style_tags") or [])[:2])
    
    direction_lower = direction.lower()
    dir_info = STYLE_DIRECTIONS.get(direction, {})
    color_policy = dir_info.get("color_policy", "neutrals")
    
    # Get slot-specific item type hint
    item_hint = SLOT_ITEM_HINTS.get(slot, f"women's {slot}")
    
    # For layers: customize the hint based on what's underneath
    # This helps embeddings find layers that work with the specific top
    if slot == "layer":
        top = None
        if chosen_items:
            top = chosen_items.get("top")
        if not top and base_category == "top":
            top = base_item
        
        if top:
            top_name = top.get("name", "").lower()
            top_style = " ".join((top.get("style_tags") or [])[:2])
            top_color = top.get("primary_color", "")
            
            # Detect top type to build better layer query
            is_strappy = any(kw in top_name for kw in ["cami", "tank", "halter", "strappy", "strap"])
            is_athletic = any(kw in top_name for kw in ["sweat", "hoodie", "athletic", "sport", "gym"])
            is_dressy = any(kw in top_name for kw in ["blouse", "silk", "satin", "elegant"])
            
            # Build layer hint based on top type
            if is_strappy:
                item_hint = f"women's open cardigan or lightweight jacket (not vest or pullover) that drapes nicely over a {top_color} {top_name}"
            elif is_athletic:
                item_hint = f"women's sporty zip-up jacket or athletic layer that matches a casual {top_color} {top_name}"
            elif is_dressy:
                item_hint = f"women's elegant blazer or refined cardigan that complements a dressy {top_color} {top_name}"
            else:
                item_hint = f"women's layer (cardigan, jacket, or blazer) that pairs well with a {top_style} {top_color} {top_name}"
    
    # Build color guidance based on policy
    if color_policy == "neutrals":
        color_hint = "in neutral colors like black, white, gray, or beige"
    elif color_policy == "contrast":
        color_hint = f"in contrasting colors that complement {base_color}"
    else:  # two_tone
        color_hint = "in neutral or one accent color"
    
    # Slot-specific overrides for better semantic matching
    if slot == "shoes":
        if occasion == "work":
            color_hint = "in professional neutral colors: black, navy, beige, or brown"
            item_hint = "women's professional closed-toe work shoes like loafers, pumps, or flats - office-appropriate"
        else:
            color_hint = "in neutral colors like black, white, or beige"
    elif slot == "bottom" and occasion == "work":
        item_hint = "women's professional bottoms like tailored trousers, dress pants, pencil skirt, or midi skirt - knee-length or longer, office-appropriate"
    elif slot == "accessory":
        color_hint = "in neutral or metallic tones"
    
    # Add occasion context for better retrieval
    # mood_text takes priority - allows ANY description like "beach day", "funeral", etc.
    occasion_hint = ""
    if mood_text:
        # Use raw mood text directly - much more flexible!
        occasion_hint = f"for {mood_text} "
    elif occasion:
        occasion_descriptions = {
            "work": "professional office-appropriate",
            "casual": "relaxed everyday",
            "going-out": "stylish evening night-out",
            "smart-casual": "polished yet relaxed",
            "workout": "athletic sporty activewear"
        }
        occasion_hint = occasion_descriptions.get(occasion, occasion) + " "
    
    # Build base query with occasion
    base_query = f"{occasion_hint}{direction_lower} {item_hint} {color_hint}"
    
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
    """Embed query text using FashionCLIP text encoder (512-dim, local, free)."""
    return _clip_embed_text(text)


def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch-embed multiple texts using FashionCLIP (local, free)."""
    if not texts:
        return []
    return _clip_embed_texts(texts)


def retrieve_candidates(
    category: str, 
    query_embedding: list[float], 
    k: int = 20,
    exclude_ids: list[int] = None,
    avoid_colors: set[str] = None,
    prefer_colors: set[str] = None,
    source: str = None,
    use_closet: bool = False,
    user_id: str = "default",
    shop_domain: str = None,  # If set, query shopify_catalog_items instead
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
        base_columns = ["id", "name", "image_url", "product_url", "primary_color", "style_tags"]
    elif shop_domain:
        table = "shopify_catalog_items"
        select_cols = "id, name, image_url, product_url, primary_color, style_tags, shopify_product_id, price"
        base_columns = ["id", "name", "image_url", "product_url", "primary_color", "style_tags", "shopify_product_id", "price"]
    else:
        table = "catalog_items"
        select_cols = "id, name, image_url, product_url, primary_color, style_tags"
        base_columns = ["id", "name", "image_url", "product_url", "primary_color", "style_tags"]
    
    # Build dynamic WHERE clause
    where_conditions = ["category = %s", "embedding IS NOT NULL"]
    params = [query_embedding, category]
    
    # Closet mode: filter by user_id
    if use_closet:
        where_conditions.append("user_id = %s")
        params.append(user_id)
    # Shopify mode: filter by shop_domain + only processed items
    elif shop_domain:
        where_conditions.append("shop_domain = %s")
        where_conditions.append("processed_at IS NOT NULL")
        params.append(shop_domain)
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
    
    columns = base_columns + ["embedding_text", "distance"]
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
    
    # Add small random noise to distances (5%) so similarly-scored items shuffle
    # This prevents the same items from always being selected
    import random
    for c in candidates:
        noise = random.uniform(0.95, 1.05)  # +/- 5% noise
        c["distance"] = c["distance"] * noise
    
    # Re-sort by adjusted distance (with noise)
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
    occasion: str = None,
    mood_text: str = None,
    prefer_occasions: list[str] = None,
    avoid_occasions: list[str] = None,
    shop_domain: str = None,  # If set, retrieve from shopify_catalog_items
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
        occasion: Predefined occasion (work, casual, going-out, etc.)
        mood_text: Raw mood text like "beach day", "funeral" - uses direct embedding comparison
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
    # For closet OR Shopify catalog, use exact category (items are already categorized correctly)
    # Main catalog stores layers under "top"; Shopify catalog has category="layer" directly
    search_category = slot if (use_closet or shop_domain) else SLOT_CATEGORY_MAP.get(slot, slot)
    
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
        user_id=user_id,
        shop_domain=shop_domain,
    )
    # Shopify: if no items with category='layer', try 'top' and filter to layer-like (vest, jacket, etc.)
    if slot == "layer" and shop_domain and not candidates:
        candidates = retrieve_candidates(
            category="top",
            query_embedding=query_embedding,
            k=k * 6,
            exclude_ids=exclude_ids,
            avoid_colors=avoid_colors,
            prefer_colors=prefer_colors,
            source=source,
            use_closet=False,
            user_id=user_id,
            shop_domain=shop_domain,
        )
        candidates = filter_layer_items(candidates)
    
    # For closet: apply soft color penalty (prefer non-matching, but don't exclude)
    if use_closet and avoid_colors_soft:
        for c in candidates:
            if c.get("primary_color") in avoid_colors_soft:
                c["_color_penalty"] = True  # Mark for soft scoring penalty
    
    
    # Apply SEMANTIC occasion filtering (intelligent, not keyword-based)
    # mood_text takes priority - it allows ANY mood description to work via direct embedding
    # occasion is used as fallback for predefined occasions
    if mood_text:
        candidates = filter_by_occasion_semantic(candidates, mood_text=mood_text)
    elif occasion and occasion in OCCASION_SEMANTIC_CONTEXTS:
        candidates = filter_by_occasion_semantic(candidates, occasion=occasion)
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
    # (closet and Shopify catalog items are already properly categorized by user)
    if slot == "layer" and not use_closet and not shop_domain:
        candidates = filter_layer_items(candidates)
    
    # For layers: filter by compatibility with the top underneath
    # The top could be in chosen_items OR be the base_item itself (daily outfit flow)
    if slot == "layer":
        top = None
        if chosen_items:
            top = chosen_items.get("top")
        if not top and base_item.get("category") == "top":
            top = base_item  # base_item IS the top in daily outfit flow
        
        if top:
            compatible = []
            for c in candidates:
                # Check material weight compatibility (layer should be heavier than top)
                if is_layer_compatible(c, top):
                    compatible.append(c)
            if compatible:  # Only filter if we have options left
                candidates = compatible
                logger.info(f"Layer compatibility: {len(candidates)} layers compatible with top")
    
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
