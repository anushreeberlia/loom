#!/usr/bin/env python3
"""
Evaluate Fashion Florence vs GPT-4o-mini on the held-out iMaterialist test set.

Uses pre-prepared test.jsonl from the fashion-florence repo (461 images).

Models:
  1. florence  — Fine-tuned Fashion Florence via HF Space API
  2. openai    — GPT-4o-mini via OpenAI API

Usage:
  # Both models on full test set:
  python eval/eval_florence.py --models florence,openai

  # Quick test (10 images):
  python eval/eval_florence.py --models florence --limit 10

  # Florence only:
  python eval/eval_florence.py --models florence
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLORENCE_REPO = Path("/Users/anushreeberlia/fashion-florence")
TEST_JSONL = FLORENCE_REPO / "data" / "processed" / "test.jsonl"


# ── Metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    return str(pred).lower().strip() == str(gold).lower().strip()


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_set(limit: int = 0) -> list[dict]:
    """Load test items from fashion-florence repo's pre-prepared test.jsonl."""
    items = []
    with open(TEST_JSONL, "r") as f:
        for line in f:
            row = json.loads(line)
            image_path = FLORENCE_REPO / row["image_path"]
            if not image_path.exists():
                continue
            gt = json.loads(row["target"])
            items.append({
                "image_path": str(image_path),
                "ground_truth": gt,
            })
            if limit and len(items) >= limit:
                break
    logger.info("Loaded %d test items from %s", len(items), TEST_JSONL)
    return items


# ── Model backends ────────────────────────────────────────────────────────────

def run_florence_raw(image_bytes: bytes) -> dict:
    """Call fine-tuned Florence via HF Space API and expand to full schema."""
    from services.fashion_florence import _call_florence_api, expand_florence_output
    raw = _call_florence_api(image_bytes)
    return expand_florence_output(raw)


def run_openai(image_bytes: bytes) -> dict:
    """Call GPT-4o-mini via OpenAI API (same as production fallback)."""
    from services.vision import _analyze_openai
    return _analyze_openai(image_bytes)


def run_gemini(image_bytes: bytes) -> dict:
    """Call Gemini 2.0 Flash via Google Generative AI API."""
    import google.generativeai as genai
    from PIL import Image
    import io

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))

    prompt = """Analyze the clothing item in this image. Return ONLY valid JSON with these keys:

- category: one of [top, bottom, dress, layer, shoes, accessory]
  * dress = ANY one-piece garment (mini dress, midi dress, bodycon dress, maxi, jumpsuit, romper)
  * top = SEPARATE upper body pieces only (blouses, t-shirts, sweaters, tanks, crop tops)
  * layer = outerwear worn OVER other clothes (jackets, coats, blazers, cardigans)
  * bottom = SEPARATE lower body pieces (pants, jeans, skirts, shorts)
  * shoes = footwear
  * accessory = bags, jewelry, scarves, hats, belts
- primary_color: one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]
- material: the fabric type (cotton, silk, knit, jersey, velvet, satin, leather, denim, linen, polyester, wool, chiffon, lace, etc.)
- style_tags: array from [minimalist, classic, edgy, romantic, sporty, athletic, activewear, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear, sexy, glamorous, trendy]
- occasion_tags: array from [everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest, going-out, clubbing, gym, workout, date, night-out, brunch]

JSON only, no markdown, no explanation."""

    response = model.generate_content(
        [prompt, image],
        generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=500),
    )

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    tags = json.loads(text)

    required_fields = ["category", "primary_color", "material", "style_tags", "occasion_tags"]
    for field in required_fields:
        if field not in tags:
            tags[field] = [] if field.endswith("_tags") else "unknown"

    return tags


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name: str, test_items: list) -> dict:
    json_valid = 0
    cat_correct = 0
    material_correct = 0
    style_f1_sum = 0.0
    occasion_f1_sum = 0.0
    per_cat = defaultdict(lambda: {"n": 0, "cat_ok": 0, "mat_ok": 0})
    latencies = []
    n_errors = 0

    for i, item in enumerate(test_items):
        gt = item["ground_truth"]
        pred = None
        is_valid = False

        with open(item["image_path"], "rb") as f:
            image_bytes = f.read()

        t0 = time.time()
        try:
            if model_name == "florence":
                pred = run_florence_raw(image_bytes)
            elif model_name == "openai":
                pred = run_openai(image_bytes)
            elif model_name == "gemini":
                pred = run_gemini(image_bytes)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            is_valid = True
        except json.JSONDecodeError:
            n_errors += 1
        except Exception as e:
            n_errors += 1
            if i < 3:
                logger.warning("[%s] Error on item %d: %s", model_name, i, str(e)[:150])
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)

        if is_valid and pred:
            json_valid += 1
            g_cat = gt.get("category", "")
            per_cat[g_cat]["n"] += 1

            if exact_match(pred.get("category"), g_cat):
                cat_correct += 1
                per_cat[g_cat]["cat_ok"] += 1

            if exact_match(pred.get("material"), gt.get("material")):
                material_correct += 1
                per_cat[g_cat]["mat_ok"] += 1

            p_style = set(s.lower() for s in (pred.get("style_tags") or []))
            g_style = set(s.lower() for s in (gt.get("style_tags") or []))
            style_f1_sum += set_f1(p_style, g_style)

            p_occ = set(s.lower() for s in (pred.get("occasion_tags") or []))
            g_occ = set(s.lower() for s in (gt.get("occasion_tags") or []))
            occasion_f1_sum += set_f1(p_occ, g_occ)

        if (i + 1) % 25 == 0 or i + 1 == len(test_items):
            logger.info("[%s] %d/%d | valid=%d err=%d | cat=%.1f%% mat=%.1f%% | %.0fms avg",
                        model_name, i + 1, len(test_items), json_valid, n_errors,
                        100 * cat_correct / max(1, json_valid),
                        100 * material_correct / max(1, json_valid),
                        sum(latencies) / len(latencies))

    n = len(test_items)
    nv = max(1, json_valid)

    per_cat_out = {}
    for cat, d in sorted(per_cat.items()):
        per_cat_out[cat] = {
            "n": d["n"],
            "cat_acc": round(100 * d["cat_ok"] / max(1, d["n"]), 1),
            "mat_acc": round(100 * d["mat_ok"] / max(1, d["n"]), 1),
        }

    return {
        "model": model_name,
        "n": n,
        "json_valid": json_valid,
        "json_valid_pct": round(100 * json_valid / n, 1),
        "category_acc": round(100 * cat_correct / nv, 1),
        "material_acc": round(100 * material_correct / nv, 1),
        "style_f1": round(style_f1_sum / nv, 3),
        "occasion_f1": round(occasion_f1_sum / nv, 3),
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
        "n_errors": n_errors,
        "per_category": per_cat_out,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]):
    hdr = f"{'Model':<16} {'JSON%':>6} {'Cat%':>6} {'Mat%':>6} {'StyF1':>6} {'OccF1':>6} {'Errs':>5} {'ms':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:<16} {r['json_valid_pct']:>5.1f}% {r['category_acc']:>5.1f}% "
              f"{r['material_acc']:>5.1f}% "
              f"{r['style_f1']:>5.3f} {r['occasion_f1']:>5.3f} {r['n_errors']:>5} {r['mean_ms']:>6.0f}")
    print("=" * len(hdr) + "\n")

    for r in results:
        if r.get("per_category"):
            print(f"  Per-category [{r['model']}]:")
            for cat, d in r["per_category"].items():
                print(f"    {cat:<12} n={d['n']:<4} cat={d['cat_acc']:>5.1f}%  mat={d['mat_acc']:>5.1f}%")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="florence",
                        help="Comma-separated: florence,openai,gemini")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 461")
    parser.add_argument("--output-dir", default="eval/results")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    test_items = load_test_set(limit=args.limit)
    all_results = []

    for model_name in models:
        logger.info("=== Evaluating: %s ===", model_name)
        r = evaluate(model_name, test_items)
        all_results.append(r)

    print_table(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"florence_eval_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "results": all_results}, f, indent=2, default=str)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
