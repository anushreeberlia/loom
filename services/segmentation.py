"""
Garment segmentation service.

Removes background from clothing images to produce clean cutouts before embedding.
Clean cutouts improve embedding quality by eliminating background noise (room, body,
furniture) that would otherwise pollute the fashion representation.

Uses rembg (wraps U2-Net) as the primary segmentation backend.
Falls back to returning the original image if segmentation fails.

Integration point: called BEFORE embedding, not before Cloudinary upload.
The full image with background is stored in Cloudinary for display;
the segmented cutout is used only for generating embeddings.
"""

import io
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_rembg_session = None


def _get_rembg_session():
    """Lazy-load rembg session (downloads U2-Net model on first call, ~170MB)."""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session

    try:
        import rembg
        _rembg_session = rembg.new_session(model_name="isnet-general-use")
        logger.info("rembg session loaded (isnet-general-use)")
    except ImportError:
        logger.error("rembg not installed")
        _rembg_session = None
    except Exception as e:
        logger.error(f"rembg session creation failed: {e}")
        _rembg_session = None

    return _rembg_session


def segment_garment(image_bytes: bytes, output_format: str = "PNG") -> bytes:
    """
    Remove background from a clothing image.

    Returns RGBA PNG bytes with transparent background (for clean embedding),
    or the original image bytes if segmentation is unavailable/fails.

    Args:
        image_bytes: Input JPEG/PNG image bytes
        output_format: "PNG" for RGBA with transparency, "JPEG" for white background

    Returns:
        Segmented image bytes (RGBA PNG or RGB JPEG with white BG)
    """
    session = _get_rembg_session()
    if session is None:
        logger.debug("Segmentation unavailable, returning original image")
        return image_bytes

    try:
        import rembg
        result_bytes = rembg.remove(
            image_bytes,
            session=session,
            alpha_matting=False,
        )

        if output_format == "JPEG":
            return _rgba_to_white_bg_jpeg(result_bytes)

        return result_bytes

    except Exception as e:
        logger.warning(f"Segmentation failed, returning original: {e}")
        return image_bytes


MAX_SEGMENT_DIM = 1024


def _resize_for_segmentation(image_bytes: bytes) -> bytes:
    """
    Resize large images before segmentation for speed, keeping natural background.
    U2-Net runtime scales with resolution; 1024px keeps quality while staying fast.
    """
    img = Image.open(io.BytesIO(image_bytes))

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    max_dim = max(img.size)
    if max_dim <= MAX_SEGMENT_DIM:
        return image_bytes

    scale = MAX_SEGMENT_DIM / max_dim
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size, Image.LANCZOS)

    if img.mode == "RGBA":
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    else:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)

    return buf.getvalue()


def segment_for_embedding(image_bytes: bytes) -> bytes:
    """
    Segment garment specifically for embedding input.

    Pipeline: resize (for speed) → segment (with natural background for better
    edge detection) → composite on white (what FashionCLIP expects).
    """
    resized = _resize_for_segmentation(image_bytes)
    segmented = segment_garment(resized, output_format="PNG")

    if segmented == resized:
        return image_bytes

    return _rgba_to_white_bg_jpeg(segmented)


def _rgba_to_white_bg_jpeg(png_bytes: bytes) -> bytes:
    """Convert RGBA PNG to RGB JPEG with white background."""
    img = Image.open(io.BytesIO(png_bytes))

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def is_segmentation_available() -> bool:
    """Check if segmentation is available without triggering model download."""
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False
