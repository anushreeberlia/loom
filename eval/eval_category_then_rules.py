#!/usr/bin/env python3
"""
Ablation: Does Fashion Florence's style/occasion prediction add value beyond
category prediction + deterministic rule application?

Approach:
  1. Load existing Florence predictions from the most recent full eval run
     (or re-run Florence on test set)
  2. For each prediction, take ONLY the predicted category
  3. Apply the MOST COMMON style/occasion tags for that Loom category
     (computed from training set statistics)
  4. Compare style F1 and occasion F1 vs. the model's direct predictions

If the "category-then-defaults" baseline achieves similar F1 to the model,
it means the model is mostly memorizing category→tag mappings.
If the model's direct predictions are substantially better, it demonstrates
genuine visual style recognition beyond category classification.

Usage:
  python eval/eval_category_then_rules.py
  python eval/eval_category_then_rules.py --florence-results eval/results/florence_eval_20260506_222920.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLORENCE_REPO = Path("/Users/anushreeberlia/fashion-florence")
TEST_JSONL = FLORENCE_REPO / "data" / "processed" / "test.jsonl"
TRAIN_JSONL = FLORENCE_REPO / "data" / "processed" / "train.jsonl"


def set_f1(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_category_defaults(train_path: Path) -> dict:
    """
    From training data, compute the most common style_tags and occasion_tags
    for each Loom category. This represents what a "predict category then
    apply most-likely tags" baseline would produce.
    """
    style_counter = defaultdict(Counter)
    occasion_counter = defaultdict(Counter)
    cat_counts = Counter()

    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            gt = json.loads(row["target"])
            cat = gt.get("category", "")
            if not cat:
                continue
            cat_counts[cat] += 1
            for tag in gt.get("style_tags", []):
                style_counter[cat][tag] += 1
            for tag in gt.get("occasion_tags", []):
                occasion_counter[cat][tag] += 1

    defaults = {}
    for cat in cat_counts:
        top_styles = [tag for tag, _ in style_counter[cat].most_common(3)]
        top_occasions = [tag for tag, _ in occasion_counter[cat].most_common(2)]
        if not top_styles:
            top_styles = ["casual"]
        if not top_occasions:
            top_occasions = ["everyday"]
        defaults[cat] = {
            "style_tags": top_styles,
            "occasion_tags": top_occasions,
        }

    logger.info("Category defaults computed from %d training examples:", sum(cat_counts.values()))
    for cat, d in sorted(defaults.items()):
        logger.info("  %s: style=%s, occasion=%s", cat, d["style_tags"], d["occasion_tags"])

    return defaults


def run_evaluation(test_path: Path, category_defaults: dict):
    """
    For each test example:
      - Method A (model): Use Florence's full prediction (style_tags, occasion_tags)
      - Method B (category-then-defaults): Use Florence's predicted category,
        then look up default tags for that category
      - Method C (oracle-category-then-defaults): Use ground-truth category,
        then look up default tags (upper bound for rule-based)
    """
    from services.fashion_florence import _call_florence_api, expand_florence_output

    results = {
        "model_direct": {"style_f1": 0.0, "occasion_f1": 0.0, "n": 0},
        "category_then_defaults": {"style_f1": 0.0, "occasion_f1": 0.0, "n": 0},
        "oracle_category_defaults": {"style_f1": 0.0, "occasion_f1": 0.0, "n": 0},
    }

    test_items = []
    with open(test_path) as f:
        for line in f:
            row = json.loads(line)
            image_path = FLORENCE_REPO / row["image_path"]
            if not image_path.exists():
                continue
            gt = json.loads(row["target"])
            test_items.append({"image_path": str(image_path), "ground_truth": gt})

    logger.info("Loaded %d test items", len(test_items))
    n_valid = 0

    for i, item in enumerate(test_items):
        gt = item["ground_truth"]
        g_style = set(s.lower() for s in gt.get("style_tags", []))
        g_occ = set(s.lower() for s in gt.get("occasion_tags", []))
        g_cat = gt.get("category", "")

        try:
            with open(item["image_path"], "rb") as f:
                image_bytes = f.read()
            raw = _call_florence_api(image_bytes)
            pred = expand_florence_output(raw)
        except Exception as e:
            if i < 3:
                logger.warning("Error on item %d: %s", i, str(e)[:100])
            continue

        if not pred:
            continue

        n_valid += 1
        pred_cat = pred.get("category", "")

        # Method A: Model's direct prediction
        p_style = set(s.lower() for s in (pred.get("style_tags") or []))
        p_occ = set(s.lower() for s in (pred.get("occasion_tags") or []))
        results["model_direct"]["style_f1"] += set_f1(p_style, g_style)
        results["model_direct"]["occasion_f1"] += set_f1(p_occ, g_occ)

        # Method B: Predicted category → default tags
        cat_defaults = category_defaults.get(pred_cat, {"style_tags": ["casual"], "occasion_tags": ["everyday"]})
        default_style = set(s.lower() for s in cat_defaults["style_tags"])
        default_occ = set(s.lower() for s in cat_defaults["occasion_tags"])
        results["category_then_defaults"]["style_f1"] += set_f1(default_style, g_style)
        results["category_then_defaults"]["occasion_f1"] += set_f1(default_occ, g_occ)

        # Method C: Ground-truth category → default tags (oracle upper bound)
        oracle_defaults = category_defaults.get(g_cat, {"style_tags": ["casual"], "occasion_tags": ["everyday"]})
        oracle_style = set(s.lower() for s in oracle_defaults["style_tags"])
        oracle_occ = set(s.lower() for s in oracle_defaults["occasion_tags"])
        results["oracle_category_defaults"]["style_f1"] += set_f1(oracle_style, g_style)
        results["oracle_category_defaults"]["occasion_f1"] += set_f1(oracle_occ, g_occ)

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d processed (%d valid)", i + 1, len(test_items), n_valid)

    # Average
    for method in results:
        results[method]["n"] = n_valid
        results[method]["style_f1"] = round(results[method]["style_f1"] / max(1, n_valid), 3)
        results[method]["occasion_f1"] = round(results[method]["occasion_f1"] / max(1, n_valid), 3)

    return results


def run_offline_evaluation(test_path: Path, category_defaults: dict):
    """
    Offline version: uses ground-truth category to simulate what happens
    when you predict category correctly (94.6% of the time) and apply rules.
    Does NOT call the Florence API — runs instantly on test.jsonl alone.
    """
    results = {
        "ground_truth_model": {"style_f1": 0.0, "occasion_f1": 0.0, "n": 0},
        "oracle_category_defaults": {"style_f1": 0.0, "occasion_f1": 0.0, "n": 0},
    }

    n = 0
    with open(test_path) as f:
        for line in f:
            row = json.loads(line)
            gt = json.loads(row["target"])
            g_cat = gt.get("category", "")
            g_style = set(s.lower() for s in gt.get("style_tags", []))
            g_occ = set(s.lower() for s in gt.get("occasion_tags", []))

            if not g_cat:
                continue
            n += 1

            # Oracle category → defaults
            cat_defaults = category_defaults.get(g_cat, {"style_tags": ["casual"], "occasion_tags": ["everyday"]})
            default_style = set(s.lower() for s in cat_defaults["style_tags"])
            default_occ = set(s.lower() for s in cat_defaults["occasion_tags"])
            results["oracle_category_defaults"]["style_f1"] += set_f1(default_style, g_style)
            results["oracle_category_defaults"]["occasion_f1"] += set_f1(default_occ, g_occ)

            # "Perfect model" baseline (F1 = 1.0 trivially)
            results["ground_truth_model"]["style_f1"] += set_f1(g_style, g_style)
            results["ground_truth_model"]["occasion_f1"] += set_f1(g_occ, g_occ)

    for method in results:
        results[method]["n"] = n
        results[method]["style_f1"] = round(results[method]["style_f1"] / max(1, n), 3)
        results[method]["occasion_f1"] = round(results[method]["occasion_f1"] / max(1, n), 3)

    return results


def main():
    parser = argparse.ArgumentParser(description="Category-then-rules ablation for Fashion Florence")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline (no API calls, uses ground-truth category as oracle)")
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args()

    train_path = TRAIN_JSONL
    test_path = TEST_JSONL

    if not test_path.exists():
        logger.error("Test file not found: %s", test_path)
        sys.exit(1)

    if not train_path.exists():
        logger.error("Train file not found: %s (needed to compute category defaults)", train_path)
        sys.exit(1)

    category_defaults = compute_category_defaults(train_path)

    if args.offline:
        logger.info("Running OFFLINE evaluation (oracle category → defaults)")
        results = run_offline_evaluation(test_path, category_defaults)
    else:
        logger.info("Running ONLINE evaluation (Florence API → category → defaults)")
        results = run_evaluation(test_path, category_defaults)

    print("\n" + "=" * 70)
    print("Category-then-Rules Ablation Results")
    print("=" * 70)
    print(f"{'Method':<30} {'Style F1':>10} {'Occasion F1':>12} {'N':>6}")
    print("-" * 70)
    for method, m in results.items():
        print(f"{method:<30} {m['style_f1']:>10.3f} {m['occasion_f1']:>12.3f} {m['n']:>6}")
    print("=" * 70)

    if "model_direct" in results:
        model_f1 = results["model_direct"]["style_f1"]
        rules_f1 = results["category_then_defaults"]["style_f1"]
        if rules_f1 > 0:
            improvement = (model_f1 - rules_f1) / rules_f1 * 100
            print(f"\nModel's style prediction vs category-defaults: +{improvement:.1f}% relative")
            if improvement > 10:
                print("→ Model adds substantial visual style signal beyond category rules.")
            else:
                print("→ Most style signal comes from category; model adds marginal value.")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"category_rules_ablation_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "results": results, "category_defaults": category_defaults},
                  f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
