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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def parse_description(description: str) -> dict:
    """
    Convert plain text description to structured BaseItem JSON using OpenAI.
    
    Args:
        description: Plain text description from vision model
        
    Returns:
        Structured dict with category, colors, fit, style_tags, etc.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    prompt = f"""Convert this clothing description into structured fashion tags.

Return ONLY valid JSON with these keys:
- category: one of [top, bottom, dress, layer, shoes, accessory]
  * top = blouses, t-shirts, shirts, tanks, crop tops (worn directly on upper body, lightweight)
  * layer = sweaters, cardigans, jackets, coats, blazers, vests, hoodies, knit sweaters (worn OVER tops, heavier)
  * bottom = pants, jeans, skirts, shorts, trousers
  * dress = one-piece dresses, jumpsuits, rompers
  * shoes = any footwear
  * accessory = bags, jewelry, scarves, hats, belts
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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a fashion expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Parser API error: {response.text}")
    
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    
    # Parse JSON (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    tags = json.loads(text)
    tags = validate_tags(tags, include_category=True)
    
    return tags
