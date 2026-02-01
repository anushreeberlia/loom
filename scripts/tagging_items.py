import os
import sys
import psycopg2
import httpx
import json
import time
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tagging import (
    ALLOWED_COLORS, ALLOWED_FIT, ALLOWED_STYLE, 
    ALLOWED_OCCASION, ALLOWED_SEASON, COLOR_MAP,
    normalize_color, validate_tags
)

load_dotenv()

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def build_prompt(name: str, category: str, hints: dict) -> str:
    hints_str = ""
    if hints.get("color"):
        hints_str += f"\n- color hint: {hints['color']}"
    
    return f"""Tag this fashion item for outfit styling.

Return ONLY valid JSON with these keys:
- primary_color: MUST be one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]. If the color is a specific shade (e.g. magenta, teal, burgundy), map to the closest base color.
- secondary_colors: array of strings from the same color palette (can be empty)
- material: string or null. If material is unknown or not evident from the name, return null.
- fit: one of [fitted, slim, straight, relaxed, oversized, wide, cropped, loose, unknown]
- style_tags: array (choose from: minimalist, classic, edgy, romantic, sporty, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear)
- occasion_tags: array (choose from: everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest)
- season_tags: array - IMPORTANT: use EITHER all_season OR a subset of [spring, summer, fall, winter], NEVER both

Item:
- name: "{name}"
- category: {category}{hints_str}

JSON only, no markdown."""


def build_fix_prompt(invalid_json: str) -> str:
    return f"""Fix this JSON to match the schema. Return ONLY valid JSON.

Required keys:
- primary_color: one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]
- secondary_colors: array from same palette
- material: string or null (use null if unknown)
- fit: one of [fitted, slim, straight, relaxed, oversized, wide, cropped, loose, unknown]
- style_tags: array (from: minimalist, classic, edgy, romantic, sporty, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear)
- occasion_tags: array (from: everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest)
- season_tags: array - use EITHER all_season OR subset of [spring, summer, fall, winter], NEVER both

Invalid JSON:
{invalid_json}

Return corrected JSON only."""


def call_gemini(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"
    
    response = httpx.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": "You are a fashion catalog tagger. Return ONLY valid JSON."}]},
            "generationConfig": {"temperature": 0.3}
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def tag_item(name: str, category: str, hints: dict) -> dict:
    prompt = build_prompt(name, category, hints)
    response_text = call_gemini(prompt)
    
    try:
        tags = parse_json(response_text)
    except json.JSONDecodeError:
        # Retry once with fix prompt
        fix_prompt = build_fix_prompt(response_text)
        response_text = call_gemini(fix_prompt)
        tags = parse_json(response_text)
    
    # Validate and fix tags (using shared function)
    tags = validate_tags(tags, include_category=False)
    
    return tags


def main():
    if not GEMINI_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Run: export GOOGLE_API_KEY=your_key_here")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Get ALL untagged items
    cursor.execute("""
        SELECT id, name, category, secondary_colors, season_tags, occasion_tags
        FROM catalog_items 
        WHERE tagged_at IS NULL AND tagging_error IS NULL
    """)
    items = cursor.fetchall()
    
    total = len(items)
    print(f"Found {total} untagged items")
    
    if total == 0:
        print("Nothing to tag!")
        cursor.close()
        conn.close()
        return
    
    tagged = 0
    failed = 0
    
    for i, (item_id, name, category, colors, seasons, occasions) in enumerate(items):
        # Build hints from existing data
        hints = {
            "color": colors[0] if colors else None,
            "season": seasons[0] if seasons else None,
            "occasion": occasions[0] if occasions else None,
        }
        
        print(f"[{i+1}/{total}] [{item_id}] {name[:50]}...", end=" ")
        
        try:
            tags = tag_item(name, category, hints)
            
            cursor.execute("""
                UPDATE catalog_items SET
                    primary_color = %s,
                    secondary_colors = %s,
                    material = %s,
                    fit = %s,
                    style_tags = %s,
                    season_tags = %s,
                    occasion_tags = %s,
                    tagged_at = NOW(),
                    tagging_error = NULL
                WHERE id = %s
            """, (
                tags.get("primary_color"),
                tags.get("secondary_colors", []),
                tags.get("material"),
                tags.get("fit"),
                tags.get("style_tags", []),
                tags.get("season_tags", []),
                tags.get("occasion_tags", []),
                item_id
            ))
            conn.commit()
            print("✓")
            tagged += 1
            
        except Exception as e:
            cursor.execute("""
                UPDATE catalog_items SET tagging_error = %s WHERE id = %s
            """, (str(e)[:500], item_id))
            conn.commit()
            print(f"✗ {str(e)[:60]}")
            failed += 1
        
        time.sleep(4)  # Rate limit: 15 requests/minute
    
    cursor.close()
    conn.close()
    print(f"\n{'='*50}")
    print(f"DONE! Tagged: {tagged}, Failed: {failed}")


if __name__ == "__main__":
    main()
