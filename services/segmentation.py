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
        _rembg_session = rembg.new_session(model_name="u2net_cloth_seg")
        logger.info("rembg session loaded (u2net_cloth_seg)")
    except ImportError:
        logger.warning("rembg not installed, trying isnet-general-use fallback")
        try:
            import rembg
            _rembg_session = rembg.new_session(model_name="isnet-general-use")
            logger.info("rembg session loaded (isnet-general-use)")
        except Exception as e:
            logger.error(f"Failed to load any rembg model: {e}")
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
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
        )

        if output_format == "JPEG":
            return _rgba_to_white_bg_jpeg(result_bytes)

        return result_bytes

    except Exception as e:
        logger.warning(f"Segmentation failed, returning original: {e}")
        return image_bytes


def segment_for_embedding(image_bytes: bytes) -> bytes:
    """
    Segment garment specifically for embedding input.

    Returns a clean JPEG with white background (what FashionCLIP/CLIP
    expects -- trained on product images with white/neutral backgrounds).
    """
    segmented = segment_garment(image_bytes, output_format="PNG")

    if segmented == image_bytes:
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
