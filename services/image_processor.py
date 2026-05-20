"""
Image processing: background removal (rembg) + resize + white bg.
"""

import io
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1200

_rembg_session = None

def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
        logger.info("rembg U2-Net session initialized")
    return _rembg_session


def process_clothing_image(image_bytes: bytes) -> bytes:
    """
    Process image: remove background with rembg, resize, add white bg.
    """
    logger.info("Processing image...")
    
    img = Image.open(io.BytesIO(image_bytes))
    
    try:
        img = ImageOps.exif_transpose(img)
    except:
        pass
    
    logger.info(f"  Original: {img.size}, mode: {img.mode}")
    
    # Resize before rembg for speed
    max_dim = max(img.size)
    if max_dim > MAX_DIMENSION:
        scale = MAX_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"  Resized to: {img.size}")

    # Background removal with rembg
    try:
        from rembg import remove
        session = _get_rembg_session()
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        result_bytes = remove(img_bytes.getvalue(), session=session)
        img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        logger.info("  Background removed")
    except Exception as e:
        logger.warning(f"  rembg failed, using original: {e}")
        img = img.convert("RGBA") if img.mode != "RGBA" else img

    # Composite onto white background
    background = Image.new('RGB', img.size, (255, 255, 255))
    if img.mode == 'RGBA':
        background.paste(img, mask=img.split()[3])
    else:
        background.paste(img)
    img = background
    logger.info("  Composited on white bg")
    
    # Save as JPEG
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=90)
    output.seek(0)
    
    logger.info("  Done!")
    return output.getvalue()
