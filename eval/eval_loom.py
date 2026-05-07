#!/usr/bin/env python3
"""
Loom ablation study: measure contribution of each scoring/retrieval component.

Runs outfit generation for N anchor items under the full system and six
ablated configurations, logging score breakdowns, violation rates,
diversity metrics, and latency.

Ablations:
  1. full          — complete system (baseline)
  2. random        — random retrieval from correct categories (trivial baseline)
  3. no_blend      — image-only embeddings (alpha=1.0, beta=0.0)
  4. no_occasion   — disable vibe/anti-vibe occasion filtering
  5. no_material   — disable semantic material weight checking
  6. no_noise      — deterministic retrieval (no distance perturbation)
  7. no_direction  — zero out direction reranking bonuses
  8. no_formality  — disable formality consistency penalty

Usage:
  # Full ablation on 100 anchors:
  python eval/eval_loom.py --n-anchors 100

  # Quick test (10 anchors, specific ablations):
  python eval/eval_loom.py --n-anchors 10 --ablations full,no_blend,no_material

  # Use Shopify catalog:
  python eval/eval_loom.py --shop-domain mystore.myshopify.com --n-anchors 50

Prerequisites:
  - PostgreSQL running with catalog data (catalog_items or shopify_catalog_items)
  - DATABASE_URL set in .env
  - FashionCLIP model downloaded (first run downloads ~605MB)

Output:
  - Prints ablation comparison table to stdout
  - Saves detailed results to eval/results/loom_ablation_TIMESTAMP.json
"""

import argparse
import json
import logging
import os
import sys
import time
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

ALL_ABLATIONS = ["full", "random", "no_blend", "no_occasion", "no_material", "no_noise", "no_direction", "no_formality"]


# ── Load anchor items from DB ─────────────────────────────────────────────────

def load_anchor_items(n: int, shop_domain: str = None, source: str = None, seed: int = 42) -> list[dict]:
    """Load N random processed items with embeddings from the catalog."""
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()

    try:
        if shop_domain:
            cur.execute(
                """
                SELECT id, shopify_product_id, name, category, image_url, product_url,
                       price, primary_color, secondary_colors, style_tags, season_tags,
                       occasion_tags, material, fit, embedding::text
                FROM shopify_catalog_items
                WHERE shop_domain = %s AND processed_at IS NOT NULL AND embedding IS NOT NULL
                ORDER BY random()
                LIMIT %s
                """,
                (shop_domain, n),
            )
        else:
            query = """
                SELECT id, name, category, image_url,
                       primary_color, secondary_colors, style_tags, season_tags,
                       occasion_tags, material, fit, embedding::text
                FROM catalog_items
                WHERE embedding IS NOT NULL
            """
            params = []
            if source:
                query += " AND source = %s"
                params.append(source)
            query += " ORDER BY random() LIMIT %s"
            params.append(n)
            cur.execute(query, params)

        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    items = []
    for row in rows:
        if shop_domain:
            cols = ["id", "shopify_product_id", "name", "category", "image_url", "product_url",
                    "price", "primary_color", "secondary_colors", "style_tags", "season_tags",
                    "occasion_tags", "material", "fit", "embedding_text"]
        else:
            cols = ["id", "name", "category", "image_url",
                    "primary_color", "secondary_colors", "style_tags", "season_tags",
                    "occasion_tags", "material", "fit", "embedding_text"]

        d = dict(zip(cols, row))
        emb_text = d.pop("embedding_text", None)
        d["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")] if emb_text else []
        if d["embedding"]:
            items.append(d)

    logger.info("Loaded %d anchor items", len(items))
    return items


# ── Ablation patches ──────────────────────────────────────────────────────────

def _make_ablation_patches(ablation: str) -> list:
    """Return a list of unittest.mock.patch context managers for the ablation."""
    patches = []

    if ablation == "random":
        import services.retrieval as ret_mod

        _orig_retrieve = ret_mod.retrieve_candidates

        def _random_retrieve(category, query_embedding, k=20, **kwargs):
            """Return random items from category instead of ANN-retrieved."""
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            cur = conn.cursor()
            shop_domain = kwargs.get("shop_domain")
            source = kwargs.get("source")
            exclude_ids = kwargs.get("exclude_ids") or []
            try:
                if shop_domain:
                    table = "shopify_catalog_items"
                    sel = "id, name, image_url, product_url, primary_color, style_tags, shopify_product_id, price, embedding::text"
                    cols = ["id", "name", "image_url", "product_url", "primary_color", "style_tags", "shopify_product_id", "price", "embedding_text"]
                    where = "category = %s AND embedding IS NOT NULL AND shop_domain = %s AND processed_at IS NOT NULL"
                    params = [category, shop_domain]
                else:
                    table = "catalog_items"
                    sel = "id, name, image_url, product_url, primary_color, style_tags, embedding::text"
                    cols = ["id", "name", "image_url", "product_url", "primary_color", "style_tags", "embedding_text"]
                    where = "category = %s AND embedding IS NOT NULL"
                    params = [category]
                    if source:
                        where += " AND source = %s"
                        params.append(source)
                if exclude_ids:
                    where += " AND id != ALL(%s)"
                    params.append(exclude_ids)
                params.append(k)
                cur.execute(f"SELECT {sel} FROM {table} WHERE {where} ORDER BY random() LIMIT %s", params)
                rows = cur.fetchall()
            finally:
                cur.close()
                conn.close()
            items = []
            for row in rows:
                item = dict(zip(cols, row))
                emb_text = item.pop("embedding_text", None)
                item["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")] if emb_text else []
                item["distance"] = random.random()
                items.append(item)
            return items

        patches.append(patch.object(ret_mod, "retrieve_candidates", _random_retrieve))

    elif ablation == "no_blend":
        import services.embedding as emb_mod
        original_blend = emb_mod.embed_item_blended

        def image_only_embed(image_bytes, base_item):
            return emb_mod.embed_item_image(image_bytes)

        patches.append(patch.object(emb_mod, "IMAGE_WEIGHT", 1.0))
        patches.append(patch.object(emb_mod, "TEXT_WEIGHT", 0.0))

    elif ablation == "no_occasion":
        import services.retrieval as ret_mod
        patches.append(patch.object(ret_mod, "filter_by_occasion_semantic",
                                    lambda candidates, *a, **kw: candidates))

    elif ablation == "no_material":
        import services.outfit as outfit_mod
        patches.append(patch.object(outfit_mod, "is_layer_compatible",
                                    lambda *a, **kw: True))

    elif ablation == "no_noise":
        _orig_random_uniform = random.uniform
        patches.append(patch.object(random, "uniform",
                                    lambda a, b: 1.0 if (a == 0.95 and b == 1.05) else _orig_random_uniform(a, b)))

    elif ablation == "no_direction":
        import services.outfit as outfit_mod
        patches.append(patch.object(outfit_mod, "compute_direction_bonus",
                                    lambda *a, **kw: 0.0))

    elif ablation == "no_formality":
        import services.outfit as outfit_mod
        patches.append(patch.object(outfit_mod, "check_formality_consistency",
                                    lambda *a, **kw: (True, 0.0)))

    return patches


# ── Run one outfit generation with optional ablation ──────────────────────────

_captured_scores = []


def _score_interceptor(original_fn):
    """Wrap select_best_outfit to capture score_details."""
    def wrapper(*args, **kwargs):
        best_items, score_details = original_fn(*args, **kwargs)
        _captured_scores.append(score_details)
        return best_items, score_details
    return wrapper


def generate_with_ablation(item: dict, ablation: str, shop_domain: str = None,
                           source: str = None) -> dict:
    """Generate outfits for one anchor item under the given ablation config."""
    import services.outfit_generator as gen_mod
    from services.outfit_generator import run_outfit_generation

    patches = _make_ablation_patches(ablation)
    _captured_scores.clear()

    orig_select = gen_mod.select_best_outfit
    gen_mod.select_best_outfit = _score_interceptor(orig_select)

    t0 = time.time()
    active_patches = [p.__enter__() for p in patches]
    try:
        outfits = run_outfit_generation(
            item,
            shop_domain=shop_domain,
            source=source,
        )
    except Exception as e:
        logger.warning("Generation failed for %s [%s]: %s", item.get("name"), ablation, e)
        outfits = []
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)
        gen_mod.select_best_outfit = orig_select

    elapsed_ms = (time.time() - t0) * 1000
    scores_copy = list(_captured_scores)

    return {
        "anchor_id": item["id"],
        "anchor_name": item.get("name", ""),
        "anchor_category": item.get("category", ""),
        "ablation": ablation,
        "n_outfits": len(outfits),
        "outfits": outfits,
        "score_details": scores_copy,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ── Compute metrics from generation results ──────────────────────────────────

def compute_metrics(all_results: list[dict]) -> dict:
    """Aggregate metrics across all anchor items for one ablation."""
    total_outfits = 0
    latencies = []
    outfit_scores = []
    sim_scores = []
    direction_bonuses = []
    hard_violations = 0
    total_scored = 0

    for result in all_results:
        latencies.append(result["elapsed_ms"])
        for outfit in result["outfits"]:
            total_outfits += 1

        for sd in result.get("score_details", []):
            total_scored += 1
            total_score = sd.get("total", 0)
            outfit_scores.append(total_score)
            bd = sd.get("breakdown", {})
            if bd.get("sim_intent_weighted"):
                sim_scores.append(bd["sim_intent_weighted"])
            if bd.get("direction_bonus") is not None:
                direction_bonuses.append(bd["direction_bonus"])
            if total_score <= -1.0:
                hard_violations += 1

    color_spreads = []
    subtype_diversities = []
    for result in all_results:
        colors_per_dir = []
        for outfit in result["outfits"]:
            outfit_colors = set()
            outfit_categories = set()
            for item in outfit.get("outfit_items", []):
                c = item.get("primary_color")
                if c:
                    outfit_colors.add(c)
                cat = item.get("slot") or item.get("category")
                if cat:
                    outfit_categories.add(cat)
            colors_per_dir.append(outfit_colors)
            subtype_diversities.append(len(outfit_categories))

        if len(colors_per_dir) >= 2:
            all_colors = set()
            for cs in colors_per_dir:
                all_colors |= cs
            color_spreads.append(len(all_colors))

    return {
        "n_anchors": len(all_results),
        "total_outfits": total_outfits,
        "outfits_per_anchor": round(total_outfits / max(1, len(all_results)), 2),
        "mean_score": round(sum(outfit_scores) / max(1, len(outfit_scores)), 3) if outfit_scores else 0,
        "mean_sim": round(sum(sim_scores) / max(1, len(sim_scores)), 3) if sim_scores else 0,
        "mean_dir_bonus": round(sum(direction_bonuses) / max(1, len(direction_bonuses)), 3) if direction_bonuses else 0,
        "hard_violation_pct": round(100 * hard_violations / max(1, total_scored), 1),
        "mean_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0, 1),
        "mean_color_spread": round(sum(color_spreads) / max(1, len(color_spreads)), 2) if color_spreads else 0,
        "mean_slot_diversity": round(sum(subtype_diversities) / max(1, len(subtype_diversities)), 2) if subtype_diversities else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_ablation_table(metrics_by_ablation: dict):
    header = (f"{'Ablation':<16} {'Score':>7} {'Sim':>6} {'DirB':>6} {'Viol%':>6} "
              f"{'Colors':>7} {'Slots':>6} {'ms':>7} {'P95':>7}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for ablation, m in metrics_by_ablation.items():
        print(f"{ablation:<16} {m['mean_score']:>7.3f} {m['mean_sim']:>6.3f} "
              f"{m['mean_dir_bonus']:>6.3f} {m['hard_violation_pct']:>5.1f}% "
              f"{m['mean_color_spread']:>7.2f} {m['mean_slot_diversity']:>6.2f} "
              f"{m['mean_latency_ms']:>6.0f} {m['p95_latency_ms']:>6.0f}")
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Loom ablation study")
    parser.add_argument("--n-anchors", type=int, default=100,
                        help="Number of anchor items to test")
    parser.add_argument("--ablations", default=",".join(ALL_ABLATIONS),
                        help="Comma-separated ablation configs")
    parser.add_argument("--shop-domain", default=None,
                        help="Shopify shop domain (uses shopify_catalog_items)")
    parser.add_argument("--source", default=None,
                        help="Catalog source filter (e.g., h_and_m)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args()

    ablations = [a.strip() for a in args.ablations.split(",")]
    random.seed(args.seed)

    anchor_items = load_anchor_items(args.n_anchors, shop_domain=args.shop_domain,
                                     source=args.source, seed=args.seed)

    if not anchor_items:
        logger.error("No anchor items loaded. Check your database and catalog.")
        sys.exit(1)

    metrics_by_ablation = {}
    full_results = {}

    for ablation in ablations:
        logger.info("Running ablation: %s (%d anchors)", ablation, len(anchor_items))
        results = []
        for i, item in enumerate(anchor_items):
            result = generate_with_ablation(item, ablation,
                                            shop_domain=args.shop_domain,
                                            source=args.source)
            results.append(result)

            if (i + 1) % 25 == 0:
                logger.info("  [%s] %d/%d anchors done", ablation, i + 1, len(anchor_items))

        metrics = compute_metrics(results)
        metrics["ablation"] = ablation
        metrics_by_ablation[ablation] = metrics
        full_results[ablation] = results

    print_ablation_table(metrics_by_ablation)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"loom_ablation_{ts}.json")

    serializable = {
        "timestamp": ts,
        "n_anchors": len(anchor_items),
        "ablations": ablations,
        "metrics": metrics_by_ablation,
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
