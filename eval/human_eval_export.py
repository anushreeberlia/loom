#!/usr/bin/env python3
"""
Export v1 (rules-only) vs v2 (multihead-integrated) outfits for human A/B eval.

Generates a blind, randomized HTML page where each anchor shows two outfits
labeled "System A" and "System B" (order shuffled per sample).  Raters judge
coherence, occasion-fit, and wearability on 1-5 Likert scales and pick a
preference winner.

Usage:
    python eval/human_eval_export.py --n-samples 30
    python eval/human_eval_export.py --n-samples 50 --seed 123
"""

import argparse
import html
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")


def load_anchor_items(n: int, seed: int = 42, user_id: str = None) -> list[dict]:
    """Load N random items that have both legacy and multihead embeddings."""
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    cur = conn.cursor()
    try:
        if user_id:
            cur.execute("""
                SELECT id, name, category, image_url, primary_color, secondary_colors,
                       style_tags, season_tags, occasion_tags, material, fit,
                       embedding::text,
                       compat_embedding::text, style_embedding::text,
                       occasion_embedding::text, fit_embedding::text,
                       material_embedding::text
                FROM user_closet_items
                WHERE embedding IS NOT NULL
                  AND image_url IS NOT NULL
                  AND compat_embedding IS NOT NULL
                  AND user_id = %s AND status = 'ready'
                ORDER BY random()
                LIMIT %s
            """, (user_id, n * 2))
        else:
            cur.execute("""
                SELECT id, name, category, image_url, primary_color, secondary_colors,
                       style_tags, season_tags, occasion_tags, material, fit,
                       embedding::text,
                       compat_embedding::text, style_embedding::text,
                       occasion_embedding::text, fit_embedding::text,
                       material_embedding::text
                FROM catalog_items
                WHERE embedding IS NOT NULL
                  AND image_url IS NOT NULL
                  AND compat_embedding IS NOT NULL
                ORDER BY random()
                LIMIT %s
            """, (n * 2,))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    cols = ["id", "name", "category", "image_url", "primary_color",
            "secondary_colors", "style_tags", "season_tags", "occasion_tags",
            "material", "fit", "embedding_text",
            "compat_text", "style_text", "occasion_text", "fit_text", "material_text"]

    items = []
    for row in rows:
        item = dict(zip(cols, row))
        if item.get("embedding_text"):
            item["embedding"] = [float(x) for x in item["embedding_text"].strip("[]").split(",")]
        for hk, col in [("compat_embedding", "compat_text"),
                         ("style_embedding", "style_text"),
                         ("occasion_embedding", "occasion_text"),
                         ("fit_embedding", "fit_text"),
                         ("material_embedding", "material_text")]:
            raw = item.pop(col, None)
            if raw:
                item[hk] = [float(x) for x in raw.strip("[]").split(",")]
        item.pop("embedding_text", None)
        items.append(item)

    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def _run_generation(item: dict, use_multihead: bool, user_id: str = None) -> dict:
    """Run outfit generation for one anchor under rules-only or multihead mode."""
    import services.outfit_generator as gen_mod
    import services.retrieval as ret_mod
    from services.outfit_generator import run_outfit_generation

    with patch.object(gen_mod, "USE_MULTIHEAD", use_multihead), \
         patch.object(ret_mod, "USE_MULTIHEAD", use_multihead):
        try:
            kw = {}
            if user_id:
                kw["use_closet"] = True
                kw["user_id"] = user_id
            outfits = run_outfit_generation(item, **kw)
            return {
                "anchor_id": item["id"],
                "n_outfits": len(outfits),
                "outfits": outfits,
            }
        except Exception as e:
            logger.warning("Generation failed for %s (multihead=%s): %s",
                           item.get("name"), use_multihead, e)
            return {"anchor_id": item["id"], "n_outfits": 0, "outfits": [], "error": str(e)}


def generate_pairs(items: list[dict], user_id: str = None) -> list[dict]:
    """Generate (rules, multihead) outfit pairs for each anchor."""
    pairs = []
    for i, item in enumerate(items):
        logger.info("[%d/%d] Generating for: %s", i + 1, len(items), item.get("name", "?"))
        rules_result = _run_generation(item, use_multihead=False, user_id=user_id)
        mh_result = _run_generation(item, use_multihead=True, user_id=user_id)
        pairs.append({
            "anchor": item,
            "rules": rules_result,
            "multihead": mh_result,
        })
    return pairs


def _outfit_images_html(outfits: list[dict], max_directions: int = 3) -> str:
    """Render outfit items as image grids per direction."""
    if not outfits:
        return '<p style="color:#999;font-style:italic">No outfits generated</p>'

    parts = []
    for outfit in outfits[:max_directions]:
        direction = html.escape(outfit.get("direction", "?"))
        items = outfit.get("outfit_items", [])
        parts.append(f'<div class="direction-label">{direction}</div>')
        parts.append('<div class="outfit-row">')
        for oi in items[:6]:
            img = oi.get("image_url", "")
            name = html.escape(str(oi.get("name", ""))[:25])
            slot = oi.get("slot", "")
            anchor_cls = " anchor-badge" if oi.get("is_anchor") else ""
            parts.append(f'''<div class="item-cell{anchor_cls}">
<img src="{img}" alt="{name}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2260%22 height=%2280%22><rect fill=%22%23eee%22 width=%2260%22 height=%2280%22/><text x=%2230%22 y=%2240%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%228%22>No img</text></svg>'">
<div class="item-name">{name}</div>
<div class="item-slot">{slot}</div>
</div>''')
        parts.append('</div>')
    return "\n".join(parts)


def generate_html(pairs: list[dict], output_path: str, seed: int):
    """Generate a blind A/B comparison HTML form."""
    rng = random.Random(seed + 7)

    assignment = []
    for _ in pairs:
        a_is_rules = rng.random() < 0.5
        assignment.append(a_is_rules)

    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loom Human Evaluation — A/B</title>
<style>
:root { --blue: #2563eb; --green: #059669; --red: #dc2626; --bg: #f8fafc; --card: #fff; --border: #e2e8f0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: #1e293b; line-height: 1.5; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
h1 { text-align: center; font-size: 1.5rem; margin-bottom: 8px; }
.subtitle { text-align: center; color: #64748b; margin-bottom: 24px; font-size: 0.9rem; }
.instructions { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 32px; }
.instructions h3 { margin-bottom: 8px; }
.instructions ul { margin-left: 20px; margin-top: 6px; }
.instructions li { margin-bottom: 4px; font-size: 0.9rem; }
.progress-bar { position: sticky; top: 0; z-index: 100; background: var(--card); border-bottom: 1px solid var(--border); padding: 8px 16px; text-align: center; font-size: 0.85rem; color: #64748b; }
.progress-fill { height: 3px; background: var(--blue); transition: width 0.3s; }

.sample { background: var(--card); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 28px; overflow: hidden; }
.sample-header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 14px; background: #f1f5f9; }
.sample-header img { width: 72px; height: 72px; object-fit: cover; border-radius: 8px; border: 1px solid var(--border); }
.sample-header .info { flex: 1; }
.sample-header .info h3 { font-size: 1rem; margin-bottom: 2px; }
.sample-header .info .meta { font-size: 0.8rem; color: #64748b; }
.sample-number { font-size: 0.75rem; color: #94a3b8; font-weight: 600; }

.systems { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.system { padding: 16px 20px; }
.system:first-child { border-right: 1px solid var(--border); }
.system-label { font-weight: 700; font-size: 1rem; margin-bottom: 10px; padding: 4px 12px; border-radius: 6px; display: inline-block; }
.system-a .system-label { background: #dbeafe; color: #1d4ed8; }
.system-b .system-label { background: #fce7f3; color: #be185d; }

.direction-label { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin: 10px 0 4px; }
.outfit-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
.item-cell { text-align: center; width: 64px; }
.item-cell img { width: 60px; height: 80px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
.item-cell.anchor-badge img { border: 2px solid var(--blue); }
.item-name { font-size: 0.6rem; color: #64748b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 64px; }
.item-slot { font-size: 0.55rem; color: #94a3b8; }

.ratings { padding: 12px 20px 16px; border-top: 1px solid var(--border); background: #fafbfc; }
.ratings h4 { font-size: 0.85rem; margin-bottom: 8px; color: #475569; }
.rating-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 24px; margin-bottom: 12px; }
.rating-row { display: flex; align-items: center; gap: 8px; }
.rating-row label { font-size: 0.8rem; min-width: 80px; color: #475569; }
.rating-row select { font-size: 0.8rem; padding: 3px 6px; border-radius: 4px; border: 1px solid var(--border); }
.pref-row { display: flex; align-items: center; gap: 12px; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
.pref-row label { font-size: 0.85rem; font-weight: 600; color: #334155; }
.pref-btn { padding: 6px 16px; border-radius: 6px; border: 2px solid var(--border); background: var(--card); cursor: pointer; font-size: 0.8rem; font-weight: 600; transition: all 0.15s; }
.pref-btn:hover { border-color: var(--blue); }
.pref-btn.selected { background: var(--blue); color: #fff; border-color: var(--blue); }
.pref-btn.selected-tie { background: #f59e0b; color: #fff; border-color: #f59e0b; }

.export-section { text-align: center; margin: 32px 0; }
.export-btn { padding: 14px 48px; font-size: 1rem; font-weight: 600; background: var(--blue); color: #fff; border: none; border-radius: 10px; cursor: pointer; transition: background 0.15s; }
.export-btn:hover { background: #1d4ed8; }
.stats { margin-top: 12px; font-size: 0.85rem; color: #64748b; }
</style>
</head>
<body>
<div class="container">
<h1>Loom Outfit Evaluation</h1>
<p class="subtitle">Blind A/B comparison &mdash; which system produces better outfits?</p>
<div class="instructions">
<h3>Instructions</h3>
<p>For each anchor item you see two systems (A and B) that generated outfits.
The assignment is randomized — you don't know which is which.</p>
<ul>
<li><strong>Coherence</strong> (1-5): Do the items go well together visually? (1 = clashing, 5 = perfect)</li>
<li><strong>Occasion</strong> (1-5): Would you wear this to the inferred occasion? (1 = wrong, 5 = perfect)</li>
<li><strong>Wearability</strong> (1-5): Would you actually wear this? (1 = never, 5 = definitely)</li>
<li><strong>Preference</strong>: Overall, which system's outfits do you prefer?</li>
</ul>
</div>
<div class="progress-bar" id="progressBar">0 / """ + str(len(pairs)) + """ rated<div class="progress-fill" id="progressFill" style="width:0%"></div></div>
<form id="evalForm">
"""]

    for idx, pair in enumerate(pairs):
        anchor = pair["anchor"]
        a_is_rules = assignment[idx]

        system_a = pair["rules"] if a_is_rules else pair["multihead"]
        system_b = pair["multihead"] if a_is_rules else pair["rules"]

        item_name = html.escape(str(anchor.get("name", f"Item {anchor['id']}")))
        item_cat = html.escape(str(anchor.get("category", "")))
        item_img = anchor.get("image_url", "")
        item_color = anchor.get("primary_color", "?")
        item_material = anchor.get("material", "?")

        a_html = _outfit_images_html(system_a.get("outfits", []))
        b_html = _outfit_images_html(system_b.get("outfits", []))

        rating_options = '<option value="">-</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option>'

        html_parts.append(f"""
<div class="sample" id="sample_{idx}">
<div class="sample-header">
<img src="{item_img}" alt="{item_name}" onerror="this.style.display='none'">
<div class="info">
<h3>{item_name}</h3>
<div class="meta">{item_cat} &middot; {item_color} &middot; {item_material}</div>
</div>
<div class="sample-number">#{idx + 1}</div>
</div>

<div class="systems">
<div class="system system-a">
<div class="system-label">System A</div>
{a_html}
</div>
<div class="system system-b">
<div class="system-label">System B</div>
{b_html}
</div>
</div>

<div class="ratings">
<h4>Rate each system</h4>
<div class="rating-grid">
<div class="rating-row"><label>A Coherence:</label><select name="q{idx}_a_coherence" onchange="updateProgress()">{rating_options}</select></div>
<div class="rating-row"><label>B Coherence:</label><select name="q{idx}_b_coherence" onchange="updateProgress()">{rating_options}</select></div>
<div class="rating-row"><label>A Occasion:</label><select name="q{idx}_a_occasion" onchange="updateProgress()">{rating_options}</select></div>
<div class="rating-row"><label>B Occasion:</label><select name="q{idx}_b_occasion" onchange="updateProgress()">{rating_options}</select></div>
<div class="rating-row"><label>A Wearability:</label><select name="q{idx}_a_wearability" onchange="updateProgress()">{rating_options}</select></div>
<div class="rating-row"><label>B Wearability:</label><select name="q{idx}_b_wearability" onchange="updateProgress()">{rating_options}</select></div>
</div>
<div class="pref-row">
<label>Prefer:</label>
<button type="button" class="pref-btn" data-q="q{idx}_pref" data-val="A" onclick="setPref(this)">A</button>
<button type="button" class="pref-btn" data-q="q{idx}_pref" data-val="tie" onclick="setPref(this)">Tie</button>
<button type="button" class="pref-btn" data-q="q{idx}_pref" data-val="B" onclick="setPref(this)">B</button>
<input type="hidden" name="q{idx}_pref" value="">
</div>
</div>
</div>
""")

    assignment_json = json.dumps(["rules" if a else "multihead" for a in assignment])

    html_parts.append(f"""
</form>
<div class="export-section">
<button type="button" class="export-btn" onclick="exportResults()">Export Results as JSON</button>
<div class="stats" id="statsBox"></div>
</div>
</div>

<script>
const ASSIGNMENT = {assignment_json};
const N = {len(pairs)};

function setPref(btn) {{
    const q = btn.dataset.q;
    const val = btn.dataset.val;
    document.querySelector('input[name="' + q + '"]').value = val;
    const siblings = btn.parentElement.querySelectorAll('.pref-btn');
    siblings.forEach(s => {{ s.classList.remove('selected', 'selected-tie'); }});
    btn.classList.add(val === 'tie' ? 'selected-tie' : 'selected');
    updateProgress();
}}

function updateProgress() {{
    let rated = 0;
    for (let i = 0; i < N; i++) {{
        const pref = document.querySelector('input[name="q' + i + '_pref"]').value;
        if (pref) rated++;
    }}
    document.getElementById('progressBar').childNodes[0].textContent = rated + ' / ' + N + ' rated';
    document.getElementById('progressFill').style.width = (rated / N * 100) + '%';
}}

function exportResults() {{
    const form = document.getElementById('evalForm');
    const data = {{ meta: {{ n_samples: N, timestamp: new Date().toISOString() }}, ratings: [] }};

    for (let i = 0; i < N; i++) {{
        const entry = {{ sample: i, a_system: ASSIGNMENT[i] }};
        ['a_coherence', 'a_occasion', 'a_wearability',
         'b_coherence', 'b_occasion', 'b_wearability'].forEach(k => {{
            const el = form.querySelector('[name="q' + i + '_' + k + '"]');
            entry[k] = el && el.value ? parseInt(el.value) : null;
        }});
        const pref = form.querySelector('input[name="q' + i + '_pref"]');
        entry.preference = pref ? pref.value || null : null;
        if (entry.preference && entry.preference !== 'tie') {{
            entry.preferred_system = (entry.preference === 'A') ? ASSIGNMENT[i]
                : (ASSIGNMENT[i] === 'rules' ? 'multihead' : 'rules');
        }} else {{
            entry.preferred_system = entry.preference === 'tie' ? 'tie' : null;
        }}
        data.ratings.push(entry);
    }}

    // Summary stats
    let rules_wins = 0, mh_wins = 0, ties = 0, unanswered = 0;
    const rules_scores = [], mh_scores = [];
    data.ratings.forEach(r => {{
        if (r.preferred_system === 'rules') rules_wins++;
        else if (r.preferred_system === 'multihead') mh_wins++;
        else if (r.preferred_system === 'tie') ties++;
        else unanswered++;

        const a_avg = [r.a_coherence, r.a_occasion, r.a_wearability].filter(v => v).reduce((s,v) => s+v, 0) / 3;
        const b_avg = [r.b_coherence, r.b_occasion, r.b_wearability].filter(v => v).reduce((s,v) => s+v, 0) / 3;
        if (r.a_system === 'rules') {{ rules_scores.push(a_avg); mh_scores.push(b_avg); }}
        else {{ mh_scores.push(a_avg); rules_scores.push(b_avg); }}
    }});

    const avg = arr => arr.length ? (arr.reduce((s,v)=>s+v,0)/arr.length).toFixed(2) : '?';
    data.summary = {{
        rules_wins, multihead_wins: mh_wins, ties, unanswered,
        rules_avg_score: parseFloat(avg(rules_scores)),
        multihead_avg_score: parseFloat(avg(mh_scores)),
    }};

    document.getElementById('statsBox').innerHTML =
        '<strong>Results:</strong> Rules wins: ' + rules_wins +
        ' | Multihead wins: ' + mh_wins +
        ' | Ties: ' + ties +
        ' | Avg rules: ' + avg(rules_scores) +
        ' | Avg multihead: ' + avg(mh_scores);

    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'human_eval_results.json'; a.click();
}}
</script>
</body></html>""")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))
    logger.info("Human eval HTML saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Export blind A/B human eval: rules vs multihead")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of anchor items to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--user-id", default=None,
                        help="User ID to load closet items from (omit for catalog)")
    args = parser.parse_args()

    random.seed(args.seed)

    src = f"user {args.user_id} closet" if args.user_id else "catalog"
    logger.info("Loading %d anchor items from %s (with multihead embeddings)...", args.n_samples, src)
    anchor_items = load_anchor_items(args.n_samples, seed=args.seed, user_id=args.user_id)
    if not anchor_items:
        logger.error("No items with multihead embeddings found. Run backfill first.")
        sys.exit(1)
    logger.info("Loaded %d anchors", len(anchor_items))

    logger.info("Generating outfit pairs (rules vs multihead) ...")
    pairs = generate_pairs(anchor_items, user_id=args.user_id)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(args.output_dir, f"human_eval_ab_{ts}.html")
    generate_html(pairs, html_path, args.seed)

    json_path = os.path.join(args.output_dir, f"human_eval_ab_data_{ts}.json")
    serializable = {
        "timestamp": ts,
        "seed": args.seed,
        "n_samples": len(anchor_items),
        "pairs": [
            {
                "anchor_id": p["anchor"]["id"],
                "anchor_name": p["anchor"].get("name"),
                "rules_n_outfits": p["rules"]["n_outfits"],
                "multihead_n_outfits": p["multihead"]["n_outfits"],
            }
            for p in pairs
        ],
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Pair metadata saved to %s", json_path)

    print(f"\nDone! Open in a browser to rate:")
    print(f"  {html_path}")
    print(f"\nAfter rating, click 'Export Results as JSON' to save.")


if __name__ == "__main__":
    main()
