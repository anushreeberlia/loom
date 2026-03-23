"""
Vision service — structured fashion metadata extraction from clothing images.

Backends:
  - "florence" (default): Fashion Florence model, zero-cost local inference
  - "openai": GPT-4o-mini via API (fallback)

Set VISION_BACKEND env var to choose. Defaults to "florence".
"""

import os
import json
import base64
import logging
import httpx
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

from services.tagging import validate_tags

load_dotenv()

logger = logging.getLogger(__name__)

VISION_BACKEND = os.getenv("VISION_BACKEND", "florence").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_IMAGE_SIZE = 512


def resize_image_for_vision(image_bytes: bytes) -> bytes:
    """Resize image to reduce API latency while keeping enough detail for clothing recognition."""
    try:
        img = Image.open(BytesIO(image_bytes))

        if max(img.size) > MAX_IMAGE_SIZE:
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()

        return image_bytes
    except Exception:
        return image_bytes


def analyze_image(image_bytes: bytes, backend: str = None) -> dict:
    """
    Extract structured fashion metadata from a clothing photo.

    Args:
        backend: "florence", "openai", or None (uses VISION_BACKEND env var).
    """
    chosen = (backend or VISION_BACKEND).lower()

    if chosen == "florence":
        try:
            from services.fashion_florence import analyze_image as florence_analyze
            return florence_analyze(image_bytes)
        except Exception as e:
            if not OPENAI_API_KEY:
                raise
            logger.warning("Florence failed (%s), falling back to OpenAI", e)
            return _analyze_openai(image_bytes)

    return _analyze_openai(image_bytes)


def _analyze_openai(image_bytes: bytes) -> dict:
    """GPT-4o-mini vision analysis — original implementation, used as fallback."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    resized_bytes = resize_image_for_vision(image_bytes)
    base64_image = base64.b64encode(resized_bytes).decode("utf-8")

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fashion expert. Analyze clothing images and return structured JSON only."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze the clothing item in this image. Return ONLY valid JSON with these keys:

- category: one of [top, bottom, dress, layer, shoes, accessory]
  * dress = ANY one-piece garment (mini dress, midi dress, bodycon dress, maxi, jumpsuit, romper)
  * top = SEPARATE upper body pieces only (blouses, t-shirts, sweaters, tanks, crop tops)
  * layer = outerwear worn OVER other clothes (jackets, coats, blazers, cardigans)
  * bottom = SEPARATE lower body pieces (pants, jeans, skirts, shorts)
  * shoes = footwear
  * accessory = bags, jewelry, scarves, hats, belts
- primary_color: one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]
- secondary_colors: array from same palette (can be empty)
- material: REQUIRED string - the fabric type (cotton, silk, knit, jersey, velvet, satin, leather, denim, linen, polyester, wool, chiffon, lace, etc.)
- fit: one of [fitted, bodycon, slim, straight, relaxed, oversized, wide, cropped, loose, unknown]
- style_tags: array from [minimalist, classic, edgy, romantic, sporty, athletic, activewear, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear, sexy, glamorous, trendy]
  * sexy = revealing, low-cut, bodycon, mini, backless, cutouts
  * athletic/activewear = sports bras, leggings, workout gear
  * elegant = dressy, refined, sophisticated
- occasion_tags: array from [everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest, going-out, clubbing, gym, workout, date, night-out, brunch]
  * If sexy/revealing/mini/bodycon → going-out, clubbing, date, party (NOT work, NOT casual)
  * If sporty/athletic → gym, workout (NOT work)
  * work = conservative, professional pieces only
- season_tags: array - use EITHER ["all_season"] OR subset of [spring, summer, fall, winter], NEVER both

JSON only, no markdown."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500
        },
        timeout=30.0
    )

    if response.status_code != 200:
        raise Exception(f"Vision API error: {response.text}")

    data = response.json()
    text = data["choices"][0]["message"]["content"]

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    tags = json.loads(text)
    tags = validate_tags(tags, include_category=True)

    return tags


def describe_image(image_bytes: bytes) -> str:
    """
    Legacy: plain-text description from GPT-4o vision.
    Kept for backward compatibility with existing code paths.
    Prefer analyze_image() for new code.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    resized_bytes = resize_image_for_vision(image_bytes)
    base64_image = base64.b64encode(resized_bytes).decode("utf-8")

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fashion expert. Describe the clothing item in the image clearly and concisely."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Describe the main clothing item visible in this image. Focus on:
- category (top, bottom, dress, layer, shoes, accessory)
- material/fabric
- colors
- fit
- vibe/occasion"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200
        },
        timeout=30.0
    )

    if response.status_code != 200:
        raise Exception(f"Vision API error: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]
