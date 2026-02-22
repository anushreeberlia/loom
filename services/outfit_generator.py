"""
Unified outfit generation: same retrieval + scoring + assembly as non-Shopify app.

Used by:
- Shopify app (shop_domain= set, then add collage in app)
- Non-Shopify app (use_closet=True, user_id= or source=)

Returns list of {direction, explanation, outfit_items} (no collage_url).
"""

import logging
import random

from services.outfit import (
    get_slots_for_outfit,
    generate_candidate_outfits,
    select_best_outfit,
    assemble_outfit,
)
from services.retrieval import retrieve_for_slot, build_query_text, get_batch_embeddings

logger = logging.getLogger(__name__)


def _get_slots(base_category: str, outfit_idx: int) -> list:
    """
    Slots for this outfit index.
    Include BOTH layer and accessory as candidates for top/bottom/shoes/dress so the
    scoring pipeline picks whichever scores higher - no slot is forced.
    """
    slots = get_slots_for_outfit(base_category, outfit_idx)
    # Ensure both layer and accessory are candidates so scoring picks the better one
    if base_category in ["top", "bottom", "shoes", "dress"]:
        if "layer" not in slots:
            slots = slots + ["layer"]
        if "accessory" not in slots:
            slots = slots + ["accessory"]
    return slots


def run_outfit_generation(
    item: dict,
    *,
    shop_domain: str = None,
    use_closet: bool = False,
    user_id: str = "default",
    source: str = None,
) -> list[dict]:
    """
    Build outfits for one anchor item using the same pipeline as the non-Shopify app.

    Args:
        item: Processed catalog item (id, name, category, image_url, embedding, primary_color,
              style_tags, occasion_tags, product_url?, price?, shopify_product_id?)
        shop_domain: If set, retrieve from shopify_catalog_items.
        use_closet: If True, retrieve from user_closet_items.
        user_id: For use_closet mode.
        source: For main catalog (e.g. "h_and_m").

    Returns:
        List of {direction, explanation, outfit_items}.
        outfit_items: list of {slot, id, name, image_url, product_url, price?, shopify_product_id?, is_anchor}.
    """
    base_item = {
        "category": item.get("category", "top"),
        "primary_color": item.get("primary_color"),
        "secondary_colors": item.get("secondary_colors", []),
        "style_tags": item.get("style_tags", []),
        "season_tags": item.get("season_tags", []),
        "occasion_tags": item.get("occasion_tags", []),
        "material": item.get("material"),
        "fit": item.get("fit"),
        "name": item.get("name", ""),
    }
    base_embedding = item.get("embedding") or []
    base_category = base_item["category"]
    directions = ["Classic", "Trendy", "Bold"]

    # Phase 1: batch query embeddings
    retrieval_tasks = []
    query_texts = []
    for outfit_idx, direction in enumerate(directions):
        for slot in _get_slots(base_category, outfit_idx):
            query_text = build_query_text(base_item, direction, slot, {})
            retrieval_tasks.append((outfit_idx, direction, slot))
            query_texts.append(query_text)

    if not query_texts:
        return []

    query_embeddings = get_batch_embeddings(query_texts)
    task_embeddings = {(t[0], t[1], t[2]): emb for t, emb in zip(retrieval_tasks, query_embeddings)}

    # Phase 2: retrieve candidates per slot
    all_candidates = {}
    for outfit_idx, direction in enumerate(directions):
        for slot in _get_slots(base_category, outfit_idx):
            emb = task_embeddings.get((outfit_idx, direction, slot))
            candidates = retrieve_for_slot(
                base_item=base_item,
                direction=direction,
                slot=slot,
                exclude_ids=[item["id"]],
                chosen_items={},
                k=15,
                precomputed_embedding=emb,
                shop_domain=shop_domain,
                use_closet=use_closet,
                user_id=user_id,
                source=source,
            )
            all_candidates[(outfit_idx, slot)] = candidates

    # Phase 3: score + select + assemble
    outfits_by_idx = {}
    used_ids_global = set()
    selection_order = list(range(len(directions)))
    random.shuffle(selection_order)

    for outfit_idx in selection_order:
        direction = directions[outfit_idx]
        slots = _get_slots(base_category, outfit_idx)
        candidates_by_slot = {}
        for slot in slots:
            raw = all_candidates.get((outfit_idx, slot), [])
            filtered = [c for c in raw if c["id"] not in used_ids_global]
            random.shuffle(filtered)
            candidates_by_slot[slot] = filtered

        has_layer_candidates = bool(candidates_by_slot.get("layer"))
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8,
            require_layer=has_layer_candidates,
        )
        if not candidate_outfits:
            continue

        best_items, score_details = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=base_embedding or None,
        )
        logger.info("  [%s] score=%.3f for %s", direction, score_details.get("total", 0), item.get("name"))

        for slot_item in best_items.values():
            if slot_item:
                used_ids_global.add(slot_item["id"])

        outfit_data = assemble_outfit(direction, base_item, best_items, base_embedding or None)

        # Build outfit_items: non-anchor from best_items (have product_url, price, shopify_product_id from retrieval)
        outfit_items = []
        for slot_name, slot_item in best_items.items():
            if not slot_item:
                continue
            outfit_items.append({
                "slot": slot_name,
                "id": slot_item["id"],
                "shopify_product_id": slot_item.get("shopify_product_id"),
                "name": slot_item["name"],
                "image_url": slot_item["image_url"],
                "product_url": slot_item.get("product_url"),
                "price": float(slot_item["price"]) if slot_item.get("price") is not None else None,
                "primary_color": slot_item.get("primary_color"),
                "is_anchor": False,
            })
        outfit_items.insert(0, {
            "slot": base_category,
            "id": item["id"],
            "shopify_product_id": item.get("shopify_product_id"),
            "name": item["name"],
            "image_url": item["image_url"],
            "product_url": item.get("product_url"),
            "price": float(item["price"]) if item.get("price") is not None else None,
            "is_anchor": True,
        })

        outfits_by_idx[outfit_idx] = {
            "direction": direction,
            "explanation": outfit_data.get("explanation", ""),
            "outfit_items": outfit_items,
        }

    return [outfits_by_idx[i] for i in range(len(directions)) if i in outfits_by_idx]
