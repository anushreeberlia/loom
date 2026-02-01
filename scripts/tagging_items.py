import os
import psycopg2
import httpx
import json
import time
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Allowed values for validation
ALLOWED_COLORS = {"black", "white", "gray", "beige", "brown", "blue", "navy", "green", "yellow", "orange", "red", "pink", "purple", "metallic", "multi", "unknown"}
ALLOWED_FIT = {"fitted", "slim", "straight", "relaxed", "oversized", "wide", "cropped", "loose", "unknown"}
ALLOWED_STYLE = {"minimalist", "classic", "edgy", "romantic", "sporty", "bohemian", "streetwear", "preppy", "elegant", "casual", "chic", "vintage", "statement", "workwear"}
ALLOWED_OCCASION = {"everyday", "casual", "work", "dinner", "party", "formal", "vacation", "lounge", "wedding_guest"}
ALLOWED_SEASON = {"spring", "summer", "fall", "winter", "all_season"}

# Color mapping for shades → base colors
COLOR_MAP = {
    "magenta": "pink", "fuchsia": "pink", "rose": "pink", "coral": "pink", "salmon": "pink",
    "violet": "purple", "lavender": "purple", "plum": "purple", "mauve": "purple",
    "teal": "green", "olive": "green", "mint": "green", "emerald": "green", "khaki": "green",
    "burgundy": "red", "maroon": "red", "crimson": "red", "wine": "red",
    "tan": "beige", "cream": "beige", "ivory": "beige", "sand": "beige", "nude": "beige", "camel": "beige",
    "charcoal": "gray", "silver": "gray", "grey": "gray",
    "gold": "metallic", "bronze": "metallic", "copper": "metallic",
    "indigo": "blue", "cobalt": "blue", "turquoise": "blue", "aqua": "blue", "sky": "blue", "denim": "blue",
    "mustard": "yellow", "lemon": "yellow",
    "rust": "orange", "peach": "orange", "terracotta": "orange",
    "coffee": "brown", "chocolate": "brown", "espresso": "brown", "mocha": "brown",
    "off-white": "white", "offwhite": "white",
}


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


def normalize_color(color: str) -> str:
    """Map shade to base color, or return unknown if not recognized."""
    if not color:
        return "unknown"
    color = color.lower().strip()
    if color in ALLOWED_COLORS:
        return color
    if color in COLOR_MAP:
        return COLOR_MAP[color]
    return "unknown"


def validate_and_fix_tags(tags: dict) -> dict:
    """Validate tags against allowed values, fix what we can."""
    # Primary color: normalize to allowed palette
    primary = tags.get("primary_color", "unknown")
    tags["primary_color"] = normalize_color(primary)
    
    # Secondary colors: normalize each, filter out unknowns
    secondary = tags.get("secondary_colors", [])
    normalized_secondary = [normalize_color(c) for c in secondary]
    tags["secondary_colors"] = [c for c in normalized_secondary if c != "unknown"]
    
    # Fit: must be in allowed set, default to unknown
    fit = tags.get("fit", "unknown")
    if fit not in ALLOWED_FIT:
        fit = "unknown"
    tags["fit"] = fit
    
    # Style tags: filter to allowed values only
    style = tags.get("style_tags", [])
    tags["style_tags"] = [s for s in style if s in ALLOWED_STYLE]
    
    # Occasion tags: filter to allowed values only
    occasion = tags.get("occasion_tags", [])
    tags["occasion_tags"] = [o for o in occasion if o in ALLOWED_OCCASION]
    
    # Season tags: fix all-season → all_season, enforce exclusivity
    season = tags.get("season_tags", [])
    # Normalize: replace hyphens with underscores
    season = [s.replace("-", "_") for s in season]
    # Filter to allowed values
    season = [s for s in season if s in ALLOWED_SEASON]
    # If all_season is present, use only that
    if "all_season" in season:
        season = ["all_season"]
    tags["season_tags"] = season
    
    # Material: convert empty string to null
    material = tags.get("material")
    if material == "" or material == "unknown":
        tags["material"] = None
    
    return tags


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
    
    # Validate and fix tags
    tags = validate_and_fix_tags(tags)
    
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
