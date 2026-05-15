#!/usr/bin/env python3
"""
Visual A/B comparison: material penalty ON vs OFF.

Generates N outfits for the same anchors under both configs,
then outputs an HTML file showing them side-by-side so you can
judge which produces better outfits.

Usage:
  python eval/visual_compare.py --n-anchors 10
  open eval/results/visual_compare.html
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


def load_anchor_items(n: int, seed: int = 42, categories: list = None) -> list[dict]:
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()
    try:
        if categories:
            cur.execute("""
                SELECT id, name, category, image_url,
                       primary_color, secondary_colors, style_tags, season_tags,
                       occasion_tags, material, fit, embedding::text
                FROM catalog_items
                WHERE embedding IS NOT NULL AND image_url IS NOT NULL
                  AND category = ANY(%s)
                ORDER BY md5(id::text || %s::text)
                LIMIT %s
            """, (categories, seed, n))
        else:
            cur.execute("""
                SELECT id, name, category, image_url,
                       primary_color, secondary_colors, style_tags, season_tags,
                       occasion_tags, material, fit, embedding::text
                FROM catalog_items
                WHERE embedding IS NOT NULL AND image_url IS NOT NULL
                ORDER BY md5(id::text || %s::text)
                LIMIT %s
            """, (seed, n))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    cols = ["id", "name", "category", "image_url",
            "primary_color", "secondary_colors", "style_tags", "season_tags",
            "occasion_tags", "material", "fit", "embedding_text"]
    items = []
    for row in rows:
        item = dict(zip(cols, row))
        emb_text = item.pop("embedding_text", None)
        if emb_text:
            item["embedding"] = [float(x) for x in emb_text.strip("[]").split(",")]
        else:
            item["embedding"] = []
        items.append(item)
    return items


def generate_outfit(item, disable_texture=False):
    """Generate outfits for one item, optionally disabling texture/layer checks."""
    from services.outfit_generator import run_outfit_generation
    import services.outfit as outfit_mod

    if disable_texture:
        with patch.object(outfit_mod, "check_texture_contrast", lambda *a, **kw: 0.0), \
             patch.object(outfit_mod, "check_layer_justification", lambda *a, **kw: 0.0):
            return run_outfit_generation(item)
    else:
        return run_outfit_generation(item)


def render_html(comparisons: list[dict], output_path: str):
    """Build an HTML page with side-by-side outfit comparisons."""
    rows_html = []

    for comp in comparisons:
        anchor = comp["anchor"]
        anchor_img = anchor.get("image_url", "")
        anchor_name = anchor.get("name", "Unknown")
        anchor_cat = anchor.get("category", "")
        anchor_material = anchor.get("material", "n/a")

        # Render outfit items as a row of images
        def outfit_to_html(outfits, label):
            if not outfits:
                return f'<div class="outfit-col"><h4>{label}</h4><p class="empty">No outfits generated</p></div>'

            html_parts = [f'<div class="outfit-col"><h4>{label}</h4>']
            for i, outfit in enumerate(outfits[:3]):
                direction = outfit.get("direction", "?")
                html_parts.append(f'<div class="outfit"><span class="dir-tag">{direction}</span>')
                html_parts.append('<div class="items-row">')
                for oi in outfit.get("outfit_items", []):
                    img = oi.get("image_url", "")
                    name = oi.get("name", "?")
                    slot = oi.get("slot", "")
                    is_anchor = oi.get("is_anchor", False)
                    border = "3px solid #2196F3" if is_anchor else "1px solid #ddd"
                    html_parts.append(
                        f'<div class="item" style="border:{border}">'
                        f'<img src="{img}" alt="{name}" title="{name} ({slot})">'
                        f'<span class="slot-label">{slot}</span>'
                        f'</div>'
                    )
                html_parts.append('</div></div>')
            html_parts.append('</div>')
            return '\n'.join(html_parts)

        with_material = outfit_to_html(comp["with_composition"], "WITH composition checks")
        without_material = outfit_to_html(comp["without_composition"], "WITHOUT composition checks")

        rows_html.append(f'''
        <div class="comparison">
            <div class="anchor-info">
                <img src="{anchor_img}" class="anchor-img">
                <div>
                    <strong>{anchor_name}</strong><br>
                    Category: {anchor_cat} | Material: {anchor_material}
                </div>
            </div>
            <div class="outfit-grid">
                {with_material}
                {without_material}
            </div>
        </div>
        ''')

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Material Penalty A/B Comparison</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.info {{ background: #e3f2fd; padding: 12px; border-radius: 6px; margin-bottom: 20px; }}
.comparison {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.anchor-info {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #eee; }}
.anchor-img {{ width: 80px; height: 80px; object-fit: cover; border-radius: 6px; }}
.outfit-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.outfit-col h4 {{ margin: 0 0 8px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
.outfit-col:first-child h4 {{ color: #2e7d32; }}
.outfit-col:last-child h4 {{ color: #c62828; }}
.outfit {{ margin-bottom: 12px; padding: 8px; background: #fafafa; border-radius: 4px; }}
.dir-tag {{ font-size: 11px; background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }}
.items-row {{ display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }}
.item {{ width: 90px; text-align: center; border-radius: 4px; overflow: hidden; }}
.item img {{ width: 85px; height: 100px; object-fit: cover; }}
.slot-label {{ font-size: 10px; color: #666; display: block; padding: 2px; }}
.empty {{ color: #999; font-style: italic; }}
.verdict {{ margin-top: 30px; background: #fff3e0; padding: 16px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Material Penalty: Visual A/B Comparison</h1>
<div class="info">
    <strong>What to look for:</strong> Do outfits on the LEFT (with material penalty) avoid
    pairing heavy layers over heavy tops? Are the items more coherent in terms of weight?
    <br>Blue-bordered items = anchor (input). Compare whether the non-anchor items differ between the two columns.
</div>
{"".join(rows_html)}
<div class="verdict">
    <h3>Your verdict</h3>
    <p>For each row, ask: "Which column produced outfits I'd actually wear?"</p>
    <ul>
        <li>If LEFT is consistently better or same: the material penalty is helping (or at least not hurting)</li>
        <li>If RIGHT is consistently better: the penalty is still over-constraining</li>
        <li>If they're identical: the penalty rarely fires (which is fine for a soft penalty)</li>
    </ul>
</div>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Visual comparison saved to: {output_path}")
    print(f"Open it with: open {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visual A/B: material penalty comparison")
    parser.add_argument("--n-anchors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/results/visual_compare.html")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Loading {args.n_anchors} anchor items...")
    anchors = load_anchor_items(args.n_anchors, seed=args.seed)
    print(f"Loaded {len(anchors)} anchors.")

    comparisons = []
    for i, item in enumerate(anchors):
        print(f"  [{i+1}/{len(anchors)}] {item.get('name', '?')} ({item.get('category')})...")

        with_composition = generate_outfit(item, disable_texture=False)
        without_composition = generate_outfit(item, disable_texture=True)

        comparisons.append({
            "anchor": item,
            "with_composition": with_composition,
            "without_composition": without_composition,
        })

    render_html(comparisons, args.output)


if __name__ == "__main__":
    main()
