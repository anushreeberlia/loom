"""
Clothing Image Processing Pipeline

Upload → Cleanup → Background Removal → Orientation Correction → Cropping → Save

Modular design for independent improvements.
"""

import io
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ExifTags
from rembg import remove, new_session

logger = logging.getLogger(__name__)

# Standard output dimensions (3:4 aspect ratio, retail standard)
OUTPUT_WIDTH = 768
OUTPUT_HEIGHT = 1024
PADDING_PERCENT = 0.08  # 8% padding around garment
MAX_INPUT_DIMENSION = 1024  # Smaller to prevent OOM on Railway

# Pre-load lightweight rembg model at import time
# u2netp is 4MB vs u2net's 176MB - much less memory
logger.info("Pre-loading rembg model (u2netp)...")
REMBG_SESSION = new_session("u2netp")
logger.info("rembg model loaded!")


def process_clothing_image(image_bytes: bytes) -> bytes:
    """
    Full processing pipeline for clothing images.
    
    Pipeline:
    1. EXIF orientation fix
    2. Resize if too large (prevent OOM)
    3. Image cleanup (white balance, contrast)
    4. Background removal (using pre-loaded model)
    5. Orientation correction (ensure upright)
    6. Smart cropping & framing
    7. Normalize to standard size
    
    Returns: Processed JPEG bytes
    """
    logger.info("Starting image processing pipeline...")
    
    # Step 1: Fix EXIF orientation
    img = fix_exif_orientation(image_bytes)
    logger.info(f"  1. EXIF fixed, size: {img.size}")
    
    # Step 2: Resize large images to prevent OOM
    img = resize_if_large(img)
    logger.info(f"  2. Resized if needed: {img.size}")
    
    # Step 3: Image cleanup
    img = cleanup_image(img)
    logger.info(f"  3. Image cleaned up")
    
    # Step 4: Background removal (using pre-loaded session)
    img_bytes = pil_to_bytes(img, format="PNG")
    removed_bg = remove(img_bytes, session=REMBG_SESSION)
    img = Image.open(io.BytesIO(removed_bg)).convert("RGBA")
    logger.info(f"  4. Background removed")
    
    # Step 5: Orientation correction
    img = correct_orientation(img)
    logger.info(f"  5. Orientation corrected")
    
    # Step 6: Smart crop and frame
    img = smart_crop_and_frame(img)
    logger.info(f"  6. Cropped and framed: {img.size}")
    
    # Step 7: Normalize to standard size
    img = normalize_size(img)
    logger.info(f"  7. Normalized to {img.size}")
    
    # Convert to JPEG
    output = io.BytesIO()
    img.convert("RGB").save(output, format="JPEG", quality=85)  # Lower quality = smaller memory
    output.seek(0)
    result = output.getvalue()
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    logger.info("Pipeline complete!")
    return result


def fix_exif_orientation(image_bytes: bytes) -> Image.Image:
    """Fix image orientation based on EXIF data."""
    img = Image.open(io.BytesIO(image_bytes))
    
    try:
        # Use PIL's built-in EXIF transpose
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logger.warning(f"EXIF transpose failed: {e}")
    
    return img.convert("RGB")


def resize_if_large(img: Image.Image) -> Image.Image:
    """
    Resize image if larger than MAX_INPUT_DIMENSION to prevent OOM.
    Maintains aspect ratio.
    """
    width, height = img.size
    max_dim = max(width, height)
    
    if max_dim > MAX_INPUT_DIMENSION:
        scale = MAX_INPUT_DIMENSION / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        logger.info(f"    Resized from {width}x{height} to {new_width}x{new_height}")
    
    return img


def cleanup_image(img: Image.Image) -> Image.Image:
    """
    Image cleanup layer:
    - Auto white balance (via color enhancement)
    - Brightness/contrast normalization
    - Mild sharpening
    """
    # Convert to numpy for analysis
    arr = np.array(img)
    
    # Auto white balance: stretch each channel
    result = auto_white_balance(arr)
    img = Image.fromarray(result)
    
    # Auto contrast
    img = ImageOps.autocontrast(img, cutoff=0.5)
    
    # Mild brightness boost if too dark
    brightness = get_brightness(img)
    if brightness < 100:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
    
    # Mild sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    return img


def auto_white_balance(img_array: np.ndarray) -> np.ndarray:
    """Simple white balance by stretching each color channel."""
    result = np.zeros_like(img_array)
    for i in range(3):
        channel = img_array[:, :, i]
        p2, p98 = np.percentile(channel, (2, 98))
        if p98 > p2:
            result[:, :, i] = np.clip((channel - p2) * 255.0 / (p98 - p2), 0, 255)
        else:
            result[:, :, i] = channel
    return result.astype(np.uint8)


def get_brightness(img: Image.Image) -> float:
    """Get average brightness of image."""
    grayscale = img.convert("L")
    return np.mean(np.array(grayscale))


def correct_orientation(img: Image.Image) -> Image.Image:
    """
    Correct garment orientation to ensure it's upright.
    
    Strategy:
    1. Find bounding box of non-transparent pixels
    2. Analyze shape - clothing is usually taller than wide
    3. If wider than tall, rotate 90°
    4. Check if upside down using simple heuristics
    """
    # Get alpha channel
    if img.mode != "RGBA":
        return img
    
    alpha = np.array(img.split()[3])
    
    # Find non-transparent pixels
    coords = np.where(alpha > 10)
    if len(coords[0]) == 0:
        return img
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    height = y_max - y_min
    width = x_max - x_min
    
    # If significantly wider than tall, probably needs rotation
    if width > height * 1.3:
        logger.info("    Rotating 90° (wider than tall)")
        img = img.rotate(90, expand=True, resample=Image.BICUBIC)
    
    # Simple upside-down check: clothing usually has more detail at top
    # (neckline, collar, shoulders) - top should have more edges
    img_array = np.array(img.convert("L"))
    alpha = np.array(img.split()[3]) if img.mode == "RGBA" else np.ones_like(img_array) * 255
    
    # Compare top third vs bottom third edge density
    h = img_array.shape[0]
    top_region = img_array[:h//3] * (alpha[:h//3] > 10)
    bottom_region = img_array[2*h//3:] * (alpha[2*h//3:] > 10)
    
    # Simple edge detection via gradient
    top_edges = np.abs(np.diff(top_region.astype(float))).sum() if top_region.size > 0 else 0
    bottom_edges = np.abs(np.diff(bottom_region.astype(float))).sum() if bottom_region.size > 0 else 0
    
    # Normalize by area
    top_area = np.sum(alpha[:h//3] > 10)
    bottom_area = np.sum(alpha[2*h//3:] > 10)
    
    top_density = top_edges / max(top_area, 1)
    bottom_density = bottom_edges / max(bottom_area, 1)
    
    # If bottom has significantly more detail, might be upside down
    if bottom_density > top_density * 1.5:
        logger.info("    Rotating 180° (upside down detection)")
        img = img.rotate(180, expand=True, resample=Image.BICUBIC)
    
    return img


def smart_crop_and_frame(img: Image.Image) -> Image.Image:
    """
    Smart cropping:
    1. Find garment bounding box
    2. Add consistent padding
    3. Center on white background
    4. Maintain 3:4 aspect ratio
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Get bounding box of non-transparent pixels
    bbox = img.getbbox()
    if not bbox:
        return img
    
    # Crop to content
    cropped = img.crop(bbox)
    c_width, c_height = cropped.size
    
    # Calculate target dimensions with padding
    padding = int(max(c_width, c_height) * PADDING_PERCENT)
    
    # Determine final size maintaining 3:4 ratio
    target_ratio = 3 / 4
    current_ratio = c_width / c_height
    
    if current_ratio > target_ratio:
        # Too wide - height determines size
        final_width = c_width + padding * 2
        final_height = int(final_width / target_ratio)
    else:
        # Too tall - width determines size  
        final_height = c_height + padding * 2
        final_width = int(final_height * target_ratio)
    
    # Create white background
    background = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
    
    # Center the garment
    x = (final_width - c_width) // 2
    y = (final_height - c_height) // 2
    background.paste(cropped, (x, y), cropped)
    
    return background


def normalize_size(img: Image.Image) -> Image.Image:
    """Resize to standard output dimensions."""
    return img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    output = io.BytesIO()
    img.save(output, format=format)
    output.seek(0)
    return output.getvalue()


# Legacy function for compatibility
def auto_rotate_image(image_bytes: bytes) -> bytes:
    """Auto-rotate image based on EXIF orientation."""
    img = fix_exif_orientation(image_bytes)
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output.getvalue()
