"""
Simple image processing - just background removal.
User handles rotation manually.
"""

import io
import logging
from PIL import Image, ImageOps
from rembg import remove, new_session

logger = logging.getLogger(__name__)

# Max dimension to prevent OOM (very conservative)
MAX_DIMENSION = 800

# Pre-load the smallest rembg model
logger.info("Loading rembg model (isnet-general-use)...")
try:
    REMBG_SESSION = new_session("isnet-general-use")
    logger.info("rembg model loaded!")
except Exception as e:
    logger.warning(f"Could not load rembg: {e}")
    REMBG_SESSION = None


def process_clothing_image(image_bytes: bytes) -> bytes:
    """
    Simple processing: resize + background removal + white bg.
    User can rotate manually via UI.
    """
    logger.info("Processing image...")
    
    # 1. Open and fix EXIF rotation
    img = Image.open(io.BytesIO(image_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except:
        pass
    img = img.convert("RGB")
    original_size = img.size
    logger.info(f"  Original: {original_size}")
    
    # 2. Resize if too large (prevent OOM)
    max_dim = max(img.size)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"  Resized to: {img.size}")
    
    # 3. Background removal
    if REMBG_SESSION:
        try:
            img_bytes = pil_to_bytes(img, "PNG")
            result_bytes = remove(img_bytes, session=REMBG_SESSION)
            img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
            logger.info("  Background removed")
        except Exception as e:
            logger.error(f"  Background removal failed: {e}")
            # Continue without bg removal
    
    # 4. Add white background
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    
    # 5. Save as JPEG
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    output.seek(0)
    
    logger.info("  Done!")
    return output.getvalue()


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    output = io.BytesIO()
    img.save(output, format=format)
    output.seek(0)
    return output.getvalue()
