"""
Collage generation service.
Creates grid-layout collages from outfit items.
"""
import os
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Collage settings
COLLAGE_DIR = Path("collages")
CELL_SIZE = 300  # Each item image size
GRID_SIZE = 2    # 2x2 grid
CANVAS_SIZE = CELL_SIZE * GRID_SIZE  # 600x600
BACKGROUND_COLOR = (245, 245, 245)  # Light gray
PADDING = 10


def get_collage_path(generation_id: int, direction: str) -> Path:
    """Get the path where a collage should be stored."""
    return COLLAGE_DIR / str(generation_id) / f"{direction.lower()}.jpg"


def collage_exists(generation_id: int, direction: str) -> bool:
    """Check if collage already exists (cache check)."""
    return get_collage_path(generation_id, direction).exists()


def load_and_resize_image(image_path: str, size: tuple[int, int]) -> Image.Image | None:
    """Load an image and resize it to fit in the given size while maintaining aspect ratio."""
    try:
        # Handle remote URLs
        if image_path.startswith("http"):
            import httpx
            from io import BytesIO
            response = httpx.get(image_path, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Image not found: {image_path} (status {response.status_code})")
                return None
            img = Image.open(BytesIO(response.content))
        else:
            # Local file
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            img = Image.open(image_path)
        
        img = img.convert("RGB")  # Ensure RGB mode
        
        # Resize maintaining aspect ratio
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create a white background and paste the image centered
        background = Image.new("RGB", size, (255, 255, 255))
        offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
        background.paste(img, offset)
        
        return background
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def create_placeholder(size: tuple[int, int], text: str = "?") -> Image.Image:
    """Create a placeholder image for missing items."""
    img = Image.new("RGB", size, (220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # Draw centered text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), text, fill=(150, 150, 150), font=font)
    
    return img


# Fixed grid positions by category
# Default layout (for tops/bottoms):
# ┌──────────┬──────────┐
# │   TOP    │  BOTTOM  │
# ├──────────┼──────────┤
# │  SHOES   │ ACCESSOR │
# └──────────┴──────────┘
#
# Dress layout:
# ┌──────────┬──────────┐
# │  DRESS   │  LAYER   │
# ├──────────┼──────────┤
# │  SHOES   │ ACCESSOR │
# └──────────┴──────────┘

SLOT_POSITIONS = {
    "top": (0, 0),                    # Top-left
    "bottom": (CELL_SIZE, 0),         # Top-right
    "shoes": (0, CELL_SIZE),          # Bottom-left
    "accessory": (CELL_SIZE, CELL_SIZE),  # Bottom-right
    "dress": (0, 0),                  # Top-left (full body)
    "layer": (CELL_SIZE, 0),          # Top-right
    "bag": (CELL_SIZE, CELL_SIZE),    # Bottom-right (with accessories)
}

SLOT_LABELS = {
    "top": "TOP",
    "bottom": "BOTTOM", 
    "shoes": "SHOES",
    "accessory": "ACC",
    "dress": "DRESS",
    "layer": "LAYER",
    "bag": "BAG",
}

# Slots to display based on base item category
LAYOUT_SLOTS = {
    "dress": ["dress", "layer", "shoes", "accessory"],
    "top": ["top", "bottom", "shoes", "accessory"],
    "bottom": ["top", "bottom", "shoes", "accessory"],
    "default": ["top", "bottom", "shoes", "accessory"],
}


def create_grid_collage(
    items: list[dict],
    output_path: Path,
    base_item: dict = None,
    title: str = None
) -> Path:
    """
    Create a 2x2 grid collage with category-aware positions.
    
    Default layout (top/bottom input):
        TOP     | BOTTOM
        --------|--------
        SHOES   | ACCESSORY
    
    Dress layout:
        DRESS   | LAYER
        --------|--------
        SHOES   | ACCESSORY
    
    Args:
        items: List of item dicts with 'image_url' and 'slot'
        output_path: Where to save the collage
        base_item: The user's input item (placed in its category slot)
        title: Optional title (direction name)
    
    Returns:
        Path to the created collage
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create canvas
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND_COLOR)
    
    # Determine layout based on base item category
    base_category = base_item.get("category", "top") if base_item else "top"
    layout_slots = list(LAYOUT_SLOTS.get(base_category, LAYOUT_SLOTS["default"]))
    
    # Check if items include a layer - if so, use custom positions
    has_layer = any(item.get("slot") == "layer" for item in items)
    custom_positions = None
    if has_layer and "layer" not in layout_slots and base_category in ["top", "bottom"]:
        # Custom layout for top/bottom with layer:
        # INPUT        | LAYER
        # COMPLEMENT   | SHOES
        # Where COMPLEMENT is the opposite of input (top->bottom, bottom->top)
        complement_slot = "bottom" if base_category == "top" else "top"
        layout_slots = [base_category, "layer", complement_slot, "shoes"]
        custom_positions = {
            base_category: (0, 0),              # Top-left (input)
            complement_slot: (0, CELL_SIZE),    # Bottom-left (complementary piece)
            "layer": (CELL_SIZE, 0),            # Top-right
            "shoes": (CELL_SIZE, CELL_SIZE),    # Bottom-right
        }
    
    # Build slot -> item mapping
    slot_items = {}
    
    # Add base item first (user's input)
    if base_item:
        base_slot = base_category
        slot_items[base_slot] = {
            "image_url": base_item.get("image_url"),
            "slot": base_slot,
            "is_input": True
        }
    
    # Add retrieved items
    for item in items:
        slot = item.get("slot")
        if slot and slot not in slot_items:  # Don't override base item
            slot_items[slot] = item
    
    # Draw each slot based on the layout
    for slot in layout_slots:
        # Use custom positions if available, otherwise default
        positions = custom_positions if custom_positions else SLOT_POSITIONS
        pos = positions.get(slot, (0, 0))
        item = slot_items.get(slot)
        
        cell_size = (CELL_SIZE - PADDING * 2, CELL_SIZE - PADDING * 2)
        
        if item:
            image_path = item.get("image_url", "")
            
            # Handle relative paths (but not URLs)
            if image_path and not image_path.startswith("/") and not image_path.startswith("http"):
                image_path = str(Path(image_path))
            
            img = load_and_resize_image(image_path, cell_size)
            
            if img:
                canvas.paste(img, (pos[0] + PADDING, pos[1] + PADDING))
                
                # Add border for input item
                if item.get("is_input"):
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle(
                        [pos[0] + PADDING, pos[1] + PADDING, 
                         pos[0] + CELL_SIZE - PADDING, pos[1] + CELL_SIZE - PADDING],
                        outline=(100, 149, 237),  # Cornflower blue
                        width=3
                    )
            else:
                placeholder = create_placeholder(cell_size, SLOT_LABELS.get(slot, "?"))
                canvas.paste(placeholder, (pos[0] + PADDING, pos[1] + PADDING))
        else:
            # Empty slot placeholder
            placeholder = create_placeholder(cell_size, SLOT_LABELS.get(slot, "?"))
            canvas.paste(placeholder, (pos[0] + PADDING, pos[1] + PADDING))
    
    # Add subtle grid lines
    draw = ImageDraw.Draw(canvas)
    line_color = (230, 230, 230)
    draw.line([(CELL_SIZE, 0), (CELL_SIZE, CANVAS_SIZE)], fill=line_color, width=2)
    draw.line([(0, CELL_SIZE), (CANVAS_SIZE, CELL_SIZE)], fill=line_color, width=2)
    
    # Save collage
    canvas.save(output_path, "JPEG", quality=90)
    logger.info(f"Collage created: {output_path}")
    
    return output_path


def generate_outfit_collage(
    generation_id: int,
    direction: str,
    items: list[dict],
    base_item: dict = None,
    force: bool = False
) -> str:
    """
    Generate a collage for an outfit, with caching.
    
    Args:
        generation_id: The generation ID
        direction: "Classic", "Trendy", or "Bold"
        items: List of item dicts from the outfit
        base_item: Dict with 'image_url' and 'category' for user's input
        force: If True, regenerate even if cached
    
    Returns:
        Relative URL path to the collage (e.g., "collages/42/classic.jpg")
    """
    output_path = get_collage_path(generation_id, direction)
    
    # Check cache
    if not force and output_path.exists():
        logger.info(f"Collage cache hit: {output_path}")
        return str(output_path)
    
    # Create collage with base item in its category slot
    create_grid_collage(items, output_path, base_item=base_item, title=direction)
    
    return str(output_path)
