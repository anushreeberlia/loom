import os
import json
import httpx
from dotenv import load_dotenv

from services.tagging import (
    ALLOWED_COLORS, ALLOWED_FIT, ALLOWED_STYLE, 
    ALLOWED_OCCASION, ALLOWED_SEASON, ALLOWED_CATEGORY,
    validate_tags
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def parse_description(description: str) -> dict:
    """
    Convert plain text description to structured BaseItem JSON.
    
    Args:
        description: Plain text description from vision model
        
    Returns:
        Structured dict with category, colors, fit, style_tags, etc.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    
    prompt = f"""Convert this clothing description into structured fashion tags.

Return ONLY valid JSON with these keys:
- category: one of [top, bottom, dress, layer, shoes, accessory]
- primary_color: one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]
- secondary_colors: array from same palette (can be empty)
- material: string or null
- fit: one of [fitted, slim, straight, relaxed, oversized, wide, cropped, loose, unknown]
- style_tags: array (from: minimalist, classic, edgy, romantic, sporty, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear)
- occasion_tags: array (from: everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest)
- season_tags: array - use EITHER all_season OR subset of [spring, summer, fall, winter], NEVER both

Description:
"{description}"

JSON only, no markdown."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"
    
    response = httpx.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Parser API error: {response.text}")
    
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    
    # Parse JSON (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    tags = json.loads(text)
    tags = validate_tags(tags, include_category=True)
    
    return tags
