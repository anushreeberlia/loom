"""
Simple image processing - resize only.
No background removal (causes OOM on Railway).
"""

import io
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Max dimension for reasonable file size
MAX_DIMENSION = 1200


def process_clothing_image(image_bytes: bytes) -> bytes:
    """
    Simple processing: just resize if needed.
    Background removal skipped (causes OOM).
    """
    logger.info("Processing image...")
    
    # 1. Open and fix EXIF rotation
    img = Image.open(io.BytesIO(image_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except:
        pass
    img = img.convert("RGB")
    logger.info(f"  Original: {img.size}")
    
    # 2. Resize if too large
    max_dim = max(img.size)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"  Resized to: {img.size}")
    
    # 3. Save as JPEG
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    output.seek(0)
    
    logger.info("  Done!")
    return output.getvalue()
