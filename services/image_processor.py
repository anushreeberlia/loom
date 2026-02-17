"""
Local image processing - background removal, trimming, padding.
Uses rembg (local AI model) instead of Cloudinary AI.
"""

import io
import logging
from PIL import Image
from rembg import remove

logger = logging.getLogger(__name__)


def process_clothing_image(image_bytes: bytes, target_ratio: tuple = (3, 4)) -> bytes:
    """
    Process a clothing image:
    1. Remove background using rembg (local AI)
    2. Trim empty space
    3. Pad to target aspect ratio with white background
    
    Args:
        image_bytes: Raw image bytes
        target_ratio: Target aspect ratio as (width, height), default 3:4
        
    Returns:
        Processed image as PNG bytes
    """
    logger.info("Processing image: removing background...")
    
    # 1. Remove background
    output_bytes = remove(image_bytes)
    
    # 2. Open as PIL Image
    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    
    # 3. Trim empty space (get bounding box of non-transparent pixels)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        logger.info(f"Trimmed to {img.size}")
    
    # 4. Pad to target aspect ratio with white background
    target_w_ratio, target_h_ratio = target_ratio
    current_w, current_h = img.size
    current_ratio = current_w / current_h
    target_ratio_val = target_w_ratio / target_h_ratio
    
    if current_ratio > target_ratio_val:
        # Too wide - add height
        new_w = current_w
        new_h = int(current_w / target_ratio_val)
    else:
        # Too tall - add width
        new_h = current_h
        new_w = int(current_h * target_ratio_val)
    
    # Add some padding (10% margin)
    padding = int(min(new_w, new_h) * 0.1)
    new_w += padding * 2
    new_h += padding * 2
    
    # Create white background
    background = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    
    # Center the image
    x = (new_w - current_w) // 2
    y = (new_h - current_h) // 2
    background.paste(img, (x, y), img)  # Use img as mask for transparency
    
    # Convert to RGB (no transparency) for JPEG compatibility
    final = background.convert("RGB")
    
    # Save to bytes
    output = io.BytesIO()
    final.save(output, format="JPEG", quality=90)
    output.seek(0)
    
    logger.info(f"Processed image: {final.size}")
    return output.getvalue()


def auto_rotate_image(image_bytes: bytes) -> bytes:
    """
    Auto-rotate image based on EXIF orientation.
    """
    from PIL import ExifTags
    
    img = Image.open(io.BytesIO(image_bytes))
    
    try:
        # Find orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = img._getexif()
        if exif:
            orientation_val = exif.get(orientation)
            if orientation_val == 3:
                img = img.rotate(180, expand=True)
            elif orientation_val == 6:
                img = img.rotate(270, expand=True)
            elif orientation_val == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # No EXIF data
    
    output = io.BytesIO()
    img.save(output, format=img.format or "JPEG", quality=95)
    output.seek(0)
    return output.getvalue()

