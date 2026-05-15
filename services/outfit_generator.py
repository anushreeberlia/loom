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
    pick_anchor_pair,
    infer_outfit_occasion,
)
from services.retrieval import retrieve_for_slot, build_query_text, get_batch_embeddings

logger = logging.getLogger(__name__)

SILHOUETTE_SLOTS = {"bottom", "shoes"}
THIRD_PIECE_SLOTS = {"top", "layer"}
FINISHER_SLOTS = {"accessory"}


def _get_slots(base_category: str, outfit_idx: int) -> list:
    """
    Slots for this outfit index.
    Include BOTH layer and accessory as candidates for top/bottom/shoes/dress so the
    scoring pipeline picks whichever scores higher - no slot is forced.
    """
    slots = get_slots_for_outfit(base_category, outfit_idx)
    if base_category in ["top", "bottom", "shoes", "dress"]:
        if "layer" not in slots:
            slots = slots + ["layer"]
        if "accessory" not in slots:
            slots = slots + ["accessory"]
    return slots


def _retrieve_one(
    slot, base_item, direction, exclude_ids, chosen_items, *,
    shop_domain=None, use_closet=False, user_id="default", source=None,
    occasion=None, mood_text=None, k=15,
):
    """Retrieve candidates for a single slot with a fresh embedding."""
    query_text = build_query_text(
        base_item, direction, slot, chosen_items or {},
        occasion=occasion, mood_text=mood_text,
    )
    embs = get_batch_embeddings([query_text])
    emb = embs[0] if embs else None
    return retrieve_for_slot(
        base_item=base_item,
        direction=direction,
        slot=slot,
        exclude_ids=exclude_ids,
        chosen_items=chosen_items,
        k=k,
        precomputed_embedding=emb,
        shop_domain=shop_domain,
        use_closet=use_closet,
        user_id=user_id,
        source=source,
        occasion=occasion,
        mood_text=mood_text,
    )


def _cascade_one_outfit(
    item: dict,
    base_item: dict,
    base_embedding: list,
    direction: str,
    outfit_idx: int,
    used_ids: set,
    *,
    shop_domain=None,
    use_closet=False,
    user_id="default",
    source=None,
    occasion=None,
    mood_text=None,
    taste_vector=None,
    dislike_vector=None,
) -> dict | None:
    """
    Build a single outfit via three-round cascade retrieval:
      Round 1 - Silhouette anchor: bottom + shoes chosen for formality coherence.
      Round 2 - Third piece: layer/top retrieved with context from round 1.
      Round 3 - Finisher: accessory retrieved with context from rounds 1+2.
    Final scoring runs on the complete outfit.
    """
    all_slots = _get_slots(base_item["category"], outfit_idx)
    exclude_ids = [item["id"]] + list(used_ids)

    retrieval_kw = dict(
        shop_domain=shop_domain, use_closet=use_closet,
        user_id=user_id, source=source,
        occasion=occasion, mood_text=mood_text,
    )

    # ── Round 1: silhouette anchor (bottom + shoes) ──
    round1_slots = [s for s in all_slots if s in SILHOUETTE_SLOTS]

    round1_queries = []
    round1_slot_order = []
    for slot in round1_slots:
        qt = build_query_text(base_item, direction, slot, {}, occasion=occasion, mood_text=mood_text)
        round1_queries.append(qt)
        round1_slot_order.append(slot)

    round1_embs = get_batch_embeddings(round1_queries) if round1_queries else []
    round1_candidates = {}
    for i, slot in enumerate(round1_slot_order):
        emb = round1_embs[i] if i < len(round1_embs) else None
        cands = retrieve_for_slot(
            base_item=base_item, direction=direction, slot=slot,
            exclude_ids=exclude_ids, chosen_items={}, k=15,
            precomputed_embedding=emb,
            **retrieval_kw,
        )
        round1_candidates[slot] = [c for c in cands if c["id"] not in used_ids]

    chosen = {}
    if round1_slots:
        best_bottom, best_shoes = pick_anchor_pair(
            base_item,
            round1_candidates.get("bottom", []),
            round1_candidates.get("shoes", []),
            top_k=5,
        )
        if best_bottom:
            chosen["bottom"] = best_bottom
        if best_shoes:
            chosen["shoes"] = best_shoes

    # ── Round 2: third piece (top/layer) with silhouette context ──
    round2_slots = [s for s in all_slots if s in THIRD_PIECE_SLOTS]
    round2_candidates = {}
    for slot in round2_slots:
        cands = _retrieve_one(
            slot, base_item, direction, exclude_ids, chosen, k=15, **retrieval_kw,
        )
        round2_candidates[slot] = [c for c in cands if c["id"] not in used_ids]

    # ── Round 3: finisher (accessory) with full context ──
    round3_context = dict(chosen)
    for slot in round2_slots:
        if round2_candidates.get(slot):
            round3_context[slot] = round2_candidates[slot][0]

    round3_slots = [s for s in all_slots if s in FINISHER_SLOTS]
    round3_candidates = {}
    for slot in round3_slots:
        cands = _retrieve_one(
            slot, base_item, direction, exclude_ids, round3_context, k=15, **retrieval_kw,
        )
        round3_candidates[slot] = [c for c in cands if c["id"] not in used_ids]

    # ── Merge all candidates into per-slot lists for final combinatorial scoring ──
    candidates_by_slot = {}
    for slot in all_slots:
        pool = (
            round1_candidates.get(slot, [])
            + round2_candidates.get(slot, [])
            + round3_candidates.get(slot, [])
        )
        seen = set()
        deduped = []
        for c in pool:
            if c["id"] not in seen and c["id"] not in used_ids:
                seen.add(c["id"])
                deduped.append(c)
        if slot in chosen and chosen[slot]:
            anchor_id = chosen[slot]["id"]
            deduped = [chosen[slot]] + [c for c in deduped if c["id"] != anchor_id]

        candidates_by_slot[slot] = deduped

    has_layer_candidates = bool(candidates_by_slot.get("layer"))
    candidate_outfits = generate_candidate_outfits(
        slots=all_slots,
        candidates_by_slot=candidates_by_slot,
        max_candidates=8,
        require_layer=has_layer_candidates,
    )
    if not candidate_outfits:
        return None

    best_items, score_details = select_best_outfit(
        candidate_outfits=candidate_outfits,
        base_item=base_item,
        direction=direction,
        base_embedding=base_embedding or None,
        taste_vector=taste_vector,
        dislike_vector=dislike_vector,
    )
    logger.info("  [%s] score=%.3f for %s", direction, score_details.get("total", 0), item.get("name"))

    return best_items, score_details


def run_outfit_generation(
    item: dict,
    *,
    shop_domain: str = None,
    use_closet: bool = False,
    user_id: str = "default",
    source: str = None,
    occasion: str = None,
    mood_text: str = None,
    taste_vector: list = None,
    dislike_vector: list = None,
) -> list[dict]:
    """
    Build outfits for one anchor item using cascade retrieval.

    Three directions (Classic, Trendy, Bold), each built via:
      Round 1 - silhouette anchor (bottom + shoes)
      Round 2 - third piece with silhouette context
      Round 3 - finisher with full outfit context
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
    directions = ["Classic", "Trendy", "Bold"]

    effective_occasion = occasion
    if not effective_occasion and not mood_text:
        effective_occasion = infer_outfit_occasion(base_item)
        logger.info("Inferred occasion from anchor: %s", effective_occasion)

    outfits_by_idx = {}
    used_ids_global = set()
    selection_order = list(range(len(directions)))
    random.shuffle(selection_order)

    for outfit_idx in selection_order:
        direction = directions[outfit_idx]

        result = _cascade_one_outfit(
            item, base_item, base_embedding, direction, outfit_idx,
            used_ids_global,
            shop_domain=shop_domain,
            use_closet=use_closet,
            user_id=user_id,
            source=source,
            occasion=effective_occasion,
            mood_text=mood_text,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector,
        )
        if result is None:
            continue

        best_items, score_details = result

        for slot_item in best_items.values():
            if slot_item:
                used_ids_global.add(slot_item["id"])

        outfit_data = assemble_outfit(
            direction, base_item, best_items, base_embedding or None,
            taste_vector=taste_vector, dislike_vector=dislike_vector,
        )

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
            "slot": base_item["category"],
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
