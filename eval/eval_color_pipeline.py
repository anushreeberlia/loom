#!/usr/bin/env python3
"""
Evaluate color prediction accuracy across three configurations:

  1. Florence-only: raw color prediction from the model
  2. FashionCLIP-only: zero-shot color classification (always)
  3. Florence + fallback: Florence predicts; if "unknown", FashionCLIP fills in
     (this is the production pipeline)

Usage:
  python eval/eval_color_pipeline.py
  python eval/eval_color_pipeline.py --limit 50
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLORENCE_REPO = Path("/Users/anushreeberlia/fashion-florence")
TEST_JSONL = FLORENCE_REPO / "data" / "processed" / "test.jsonl"


def exact_match_color(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    return pred.lower().strip() == gold.lower().strip()


def near_match_color(pred: str, gold: str) -> bool:
    """Allow near-synonyms as matches."""
    if exact_match_color(pred, gold):
        return True
    synonyms = {
        frozenset({"navy", "blue"}),
        frozenset({"beige", "brown"}),
        frozenset({"gray", "silver"}),
        frozenset({"metallic", "gold"}),
        frozenset({"metallic", "silver"}),
    }
    p, g = pred.lower().strip(), gold.lower().strip()
    return frozenset({p, g}) in synonyms


def load_test_set(limit: int = 0) -> list[dict]:
    items = []
    with open(TEST_JSONL) as f:
        for line in f:
            row = json.loads(line)
            image_path = FLORENCE_REPO / row["image_path"]
            if not image_path.exists():
                continue
            gt = json.loads(row["target"])
            items.append({"image_path": str(image_path), "ground_truth": gt})
            if limit and len(items) >= limit:
                break
    logger.info("Loaded %d test items", len(items))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args()

    from services.fashion_florence import _call_florence_api, _classify_color

    test_items = load_test_set(limit=args.limit)

    results = {
        "florence_only": {"exact": 0, "near": 0, "unknown_rate": 0, "n": 0},
        "fashionclip_only": {"exact": 0, "near": 0, "unknown_rate": 0, "n": 0},
        "florence_with_fallback": {"exact": 0, "near": 0, "unknown_rate": 0, "n": 0},
    }
    n_unknown = 0
    n_valid = 0

    for i, item in enumerate(test_items):
        gt = item["ground_truth"]
        g_color = gt.get("primary_color", "")
        if not g_color or g_color == "unknown":
            continue

        with open(item["image_path"], "rb") as f:
            image_bytes = f.read()

        try:
            raw = _call_florence_api(image_bytes)
            florence_color = (raw.get("primary_color") or "unknown").lower().strip()
        except Exception as e:
            if i < 3:
                logger.warning("Florence error on item %d: %s", i, str(e)[:100])
            continue

        try:
            clip_color = _classify_color(image_bytes)
        except Exception as e:
            clip_color = "unknown"

        n_valid += 1

        # Florence only
        if florence_color == "unknown":
            n_unknown += 1
        if exact_match_color(florence_color, g_color):
            results["florence_only"]["exact"] += 1
        if near_match_color(florence_color, g_color):
            results["florence_only"]["near"] += 1

        # FashionCLIP only
        if exact_match_color(clip_color, g_color):
            results["fashionclip_only"]["exact"] += 1
        if near_match_color(clip_color, g_color):
            results["fashionclip_only"]["near"] += 1

        # Combined pipeline (production)
        combined_color = florence_color if florence_color != "unknown" else clip_color
        if exact_match_color(combined_color, g_color):
            results["florence_with_fallback"]["exact"] += 1
        if near_match_color(combined_color, g_color):
            results["florence_with_fallback"]["near"] += 1

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d processed (%d valid)", i + 1, len(test_items), n_valid)

    # Compute percentages
    for method in results:
        results[method]["n"] = n_valid
        results[method]["exact_pct"] = round(100 * results[method]["exact"] / max(1, n_valid), 1)
        results[method]["near_pct"] = round(100 * results[method]["near"] / max(1, n_valid), 1)

    results["florence_only"]["unknown_rate"] = round(100 * n_unknown / max(1, n_valid), 1)

    print("\n" + "=" * 70)
    print("Color Prediction Pipeline Evaluation")
    print("=" * 70)
    print(f"{'Method':<25} {'Exact %':>8} {'Near %':>8} {'N':>6} {'Notes':>20}")
    print("-" * 70)
    for method, m in results.items():
        notes = ""
        if method == "florence_only":
            notes = f"({m['unknown_rate']}% unknown)"
        print(f"{method:<25} {m['exact_pct']:>7.1f}% {m['near_pct']:>7.1f}% {m['n']:>6} {notes:>20}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"color_pipeline_eval_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "results": results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
