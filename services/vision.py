import os
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


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
    
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        },
        timeout=30.0
    )
    
    if response.status_code != 200:
        raise Exception(f"Vision API error: {response.text}")
    
    data = response.json()
    return data["choices"][0]["message"]["content"]

