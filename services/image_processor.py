"""
Simple image processing - resize and handle transparency.
Background removal done client-side in browser.
"""

import io
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Max dimension for reasonable file size
MAX_DIMENSION = 1200


def process_clothing_image(image_bytes: bytes) -> bytes:
    """
    Process image: handle transparency, resize, add white bg.
    Background removal is done client-side before upload.
    """
    logger.info("Processing image...")
    
    # 1. Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Fix EXIF rotation
    try:
        img = ImageOps.exif_transpose(img)
    except:
        pass
    
    logger.info(f"  Original: {img.size}, mode: {img.mode}")
    
    # 2. Resize if too large
    max_dim = max(img.size)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"  Resized to: {img.size}")
    
    # 3. Handle transparency - add white background
    if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
        # Has transparency - composite onto white
        img = img.convert('RGBA')
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
        logger.info("  Added white background")
    else:
        img = img.convert("RGB")
    
    # 4. Save as JPEG
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=90)
    output.seek(0)
    
    logger.info("  Done!")
    return output.getvalue()
