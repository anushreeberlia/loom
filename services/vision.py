import os
import base64
import httpx
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Max image dimension for Vision API (smaller = faster, 512-768 is plenty for clothing)
MAX_IMAGE_SIZE = 512


def resize_image_for_vision(image_bytes: bytes) -> bytes:
    """Resize image to reduce API latency while keeping enough detail for clothing recognition."""
    try:
        img = Image.open(BytesIO(image_bytes))
        
        # Only resize if larger than max size
        if max(img.size) > MAX_IMAGE_SIZE:
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (handles PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
        
        return image_bytes
    except Exception:
        # If resize fails, return original
        return image_bytes


def describe_image(image_bytes: bytes) -> str:
    """
    Send image to GPT-4o and get a plain text description.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Plain text description of the clothing item
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Resize image for faster API response
    resized_bytes = resize_image_for_vision(image_bytes)
    
    # Encode image to base64
    base64_image = base64.b64encode(resized_bytes).decode("utf-8")
    
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
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
- category (top, bottom, dress, shoes, layer, accessory)
- colors (primary and any secondary)
- fit (fitted, relaxed, oversized, etc.)
- style (casual, formal, sporty, elegant, etc.)
- any notable details (patterns, textures, features)

Do not guess brand. Do not add opinions. Be factual and concise."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"  # Faster processing, sufficient for clothing
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200  # Reduced from 300, description is short
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Vision API error: {response.text}")
    
    data = response.json()
    return data["choices"][0]["message"]["content"]

