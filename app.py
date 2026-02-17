from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import random
import logging
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import Json

from services.vision import describe_image
from services.parser import parse_description
from services.embedding import embed_base_item
from services.outfit import (
    STYLE_DIRECTIONS, 
    get_slots_for_outfit, 
    assemble_outfit,
    generate_candidate_outfits,
    select_best_outfit
)
from services.retrieval import retrieve_for_slot, build_query_text, get_batch_embeddings
from services.collage import generate_outfit_collage
from services.weather import fetch_weather, get_weather_outfit_adjustments, WeatherData
from services.image_processor import process_clothing_image

import os
from dotenv import load_dotenv
load_dotenv()


class FeedbackRequest(BaseModel):
    generation_id: int
    outfit_index: int  # 0, 1, or 2
    liked: bool
    session_id: str = None  # For taste vector tracking

app = FastAPI(title="AI Outfit Styler")

# Ensure directories exist
Path("collages").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/outfit_styler")

# Catalog source to use for outfit generation
# Options: 'h_and_m', 'kaggle_fashion', or None (use all)
CATALOG_SOURCE = "h_and_m"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_absolute_url(base_url: str, relative_path: str) -> str:
    """
    Convert a relative path to an absolute URL.
    
    Handles:
    - catalog/images/xxx.jpg -> {base_url}/static/catalog/xxx.jpg
    - collages/42/classic.jpg -> {base_url}/static/generated/42/classic.jpg
    - /static/collages/... -> {base_url}/static/generated/...
    """
    if not relative_path:
        return ""
    
    # Already absolute
    if relative_path.startswith("http://") or relative_path.startswith("https://"):
        return relative_path
    
    # Remove leading slash if present
    path = relative_path.lstrip("/")
    
    # Catalog images: catalog/images/xxx.jpg -> /static/catalog/xxx.jpg
    if path.startswith("catalog/images/"):
        filename = path.replace("catalog/images/", "")
        return f"{base_url}/static/catalog/{filename}"
    
    # Collages: collages/42/classic.jpg -> /static/generated/42/classic.jpg
    if path.startswith("collages/"):
        subpath = path.replace("collages/", "")
        return f"{base_url}/static/generated/{subpath}"
    
    # Legacy: /static/collages/... -> /static/generated/...
    if path.startswith("static/collages/"):
        subpath = path.replace("static/collages/", "")
        return f"{base_url}/static/generated/{subpath}"
    
    # Fallback: just prepend base URL
    return f"{base_url}/{path}"


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def get_cached_image_analysis(image_hash: str) -> dict | None:
    """Check if we've already analyzed this image (by hash)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT input_description, parsed_tags, base_item_embedding::text
               FROM outfit_generations 
               WHERE input_image_hash = %s 
               ORDER BY created_at DESC 
               LIMIT 1""",
            (image_hash,)
        )
        row = cursor.fetchone()
        if row and row[0] and row[1]:
            # Parse embedding from pgvector text format
            embedding = None
            if row[2]:
                embedding = [float(x) for x in row[2].strip("[]").split(",")]
            return {
                "description": row[0],
                "base_item": row[1],
                "embedding": embedding
            }
        return None
    except Exception as e:
        logger.error(f"Cache lookup error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_disliked_item_ids(session_id: str) -> set[int]:
    """Get IDs of items the user has disliked (to exclude from future results)."""
    if not session_id:
        return set()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get all outfit items from disliked outfits for this session
        cursor.execute(
            """SELECT og.output_outfits, fe.outfit_index
               FROM feedback_events fe
               JOIN outfit_generations og ON og.id = fe.generation_id
               WHERE fe.session_id = %s AND fe.liked = false""",
            (session_id,)
        )
        
        disliked_ids = set()
        for row in cursor.fetchall():
            outfits = row[0]
            outfit_idx = row[1]
            if outfits and outfit_idx < len(outfits):
                outfit = outfits[outfit_idx]
                for item in outfit.get("items", []):
                    if item and item.get("id"):
                        disliked_ids.add(item["id"])
        
        return disliked_ids
    except Exception as e:
        logger.error(f"Error fetching disliked items: {e}")
        return set()
    finally:
        cursor.close()
        conn.close()


def get_taste_vector(session_id: str) -> tuple[list | None, list | None]:
    """Retrieve taste and dislike vectors for a session, if exists."""
    if not session_id:
        return None, None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT taste_embedding::text, dislike_embedding::text, like_count, dislike_count
               FROM taste_vectors 
               WHERE session_id = %s""",
            (session_id,)
        )
        row = cursor.fetchone()
        taste_vec = None
        dislike_vec = None
        
        if row:
            # Parse like embedding
            if row[0] and row[2] and row[2] > 0:
                taste_vec = [float(x) for x in row[0].strip("[]").split(",")]
            # Parse dislike embedding
            if row[1] and row[3] and row[3] > 0:
                dislike_vec = [float(x) for x in row[1].strip("[]").split(",")]
        
        return taste_vec, dislike_vec
    except Exception as e:
        logger.error(f"Error fetching taste vector: {e}")
        return None, None
    finally:
        cursor.close()
        conn.close()


def update_taste_vector(session_id: str, item_embeddings: list[list], liked: bool):
    """
    Update taste vector based on feedback.
    
    For likes: blend item embeddings into taste_embedding (moving average)
    For dislikes: blend item embeddings into dislike_embedding (to penalize)
    """
    if not session_id or not item_embeddings:
        return
    
    # Filter out empty embeddings
    valid_embeddings = [e for e in item_embeddings if e and len(e) == 1536]
    if not valid_embeddings:
        return
    
    # Average the item embeddings from this outfit
    outfit_embedding = [
        sum(e[i] for e in valid_embeddings) / len(valid_embeddings)
        for i in range(1536)
    ]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get current vectors
        cursor.execute(
            """SELECT taste_embedding::text, dislike_embedding::text, like_count, dislike_count
               FROM taste_vectors WHERE session_id = %s""",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if liked:
            if row and row[0]:
                # Blend with existing taste vector
                current_emb = [float(x) for x in row[0].strip("[]").split(",")]
                count = row[2] or 0
                alpha = 1.0 / (count + 1)
                new_embedding = [
                    (1 - alpha) * current_emb[i] + alpha * outfit_embedding[i]
                    for i in range(1536)
                ]
                cursor.execute(
                    """UPDATE taste_vectors 
                       SET taste_embedding = %s::vector, like_count = like_count + 1, updated_at = NOW()
                       WHERE session_id = %s""",
                    (new_embedding, session_id)
                )
            else:
                # First like
                cursor.execute(
                    """INSERT INTO taste_vectors (session_id, taste_embedding, like_count)
                       VALUES (%s, %s::vector, 1)
                       ON CONFLICT (session_id) 
                       DO UPDATE SET taste_embedding = EXCLUDED.taste_embedding,
                                     like_count = taste_vectors.like_count + 1, updated_at = NOW()""",
                    (session_id, outfit_embedding)
                )
        else:
            # Dislike - blend into dislike_embedding
            if row and row[1]:
                # Blend with existing dislike vector
                current_emb = [float(x) for x in row[1].strip("[]").split(",")]
                count = row[3] or 0
                alpha = 1.0 / (count + 1)
                new_embedding = [
                    (1 - alpha) * current_emb[i] + alpha * outfit_embedding[i]
                    for i in range(1536)
                ]
                cursor.execute(
                    """UPDATE taste_vectors 
                       SET dislike_embedding = %s::vector, dislike_count = dislike_count + 1, updated_at = NOW()
                       WHERE session_id = %s""",
                    (new_embedding, session_id)
                )
            else:
                # First dislike
                cursor.execute(
                    """INSERT INTO taste_vectors (session_id, dislike_embedding, dislike_count)
                       VALUES (%s, %s::vector, 1)
                       ON CONFLICT (session_id) 
                       DO UPDATE SET dislike_embedding = EXCLUDED.dislike_embedding,
                                     dislike_count = taste_vectors.dislike_count + 1, updated_at = NOW()""",
                    (session_id, outfit_embedding)
                )
        
        conn.commit()
        logger.info(f"Taste vector updated for session {session_id[:8]}... (liked={liked})")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating taste vector: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/")
async def serve_landing():
    """Serve the landing page."""
    return FileResponse("static/landing.html")


@app.get("/demo")
async def serve_demo():
    """Serve the demo page (catalog-based, no login)."""
    return FileResponse("static/index.html")


@app.get("/closet")
async def serve_closet():
    """Serve the closet page for outfit generation from personal wardrobe."""
    return FileResponse("static/closet.html")


@app.get("/inventory")
async def serve_inventory():
    """Serve the inventory page for managing closet items."""
    return FileResponse("static/inventory.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount static files AFTER route definitions
Path("collages").mkdir(exist_ok=True)  # Ensure collages dir exists
app.mount("/static/generated", StaticFiles(directory="collages"), name="generated")
# Note: catalog images are now served from Cloudinary, no local static mount needed


@app.post("/v1/outfits:generate")
async def generate_outfits(request: Request, file: UploadFile = File(...), session_id: str = Form(None)):
    logger.info(f"Request received: POST /v1/outfits:generate (session={session_id[:8] if session_id else 'none'}...)")
    
    # Get base URL for absolute URLs
    base_url = str(request.base_url).rstrip("/")
    
    # Get taste vectors for personalization
    taste_vector, dislike_vector = get_taste_vector(session_id) if session_id else (None, None)
    if taste_vector or dislike_vector:
        logger.info(f"Taste vectors found (likes={taste_vector is not None}, dislikes={dislike_vector is not None})")
    
    # Get disliked item IDs to exclude entirely
    disliked_item_ids = get_disliked_item_ids(session_id) if session_id else set()
    if disliked_item_ids:
        logger.info(f"Excluding {len(disliked_item_ids)} previously disliked items")

    # Validate file
    contents = await file.read()
    if not contents:
        logger.warning("No image uploaded or file is empty")
        raise HTTPException(status_code=400, detail="No image uploaded or file is empty")

    # 1. Save uploaded file (always needed for collage)
    ext = file.filename.split(".")[-1] if file.filename else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    upload_path = Path("uploads") / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(contents)

    # Compute image hash for caching
    image_hash = hashlib.sha256(contents).hexdigest()
    
    # Check cache first
    cached = get_cached_image_analysis(image_hash)
    if cached and cached.get("embedding"):
        logger.info(f"⚡ Cache HIT for image {image_hash[:12]}... - skipping Vision/Parser/Embedding")
        description = cached["description"]
        base_item = cached["base_item"]
        embedding = cached["embedding"]
    else:
        logger.info(f"Cache MISS for image {image_hash[:12]}... - calling APIs")

        # 2. Vision: Get description from image
        logger.info("Calling vision API...")
        try:
            description = describe_image(contents)
            logger.info(f"Description: {description[:100]}...")
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            raise HTTPException(status_code=500, detail=f"Vision API error: {str(e)}")

        # 3. Parser: Convert description to structured JSON
        logger.info("Parsing description to BaseItem...")
        try:
            base_item = parse_description(description)
            logger.info(f"BaseItem: {base_item}")
        except Exception as e:
            logger.error(f"Parser error: {e}")
            raise HTTPException(status_code=500, detail=f"Parser error: {str(e)}")

        # 4. Embedding: Generate embedding for BaseItem
        logger.info("Generating embedding...")
        try:
            embedding = embed_base_item(base_item)
            logger.info(f"Embedding generated (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    # 5. Retrieve candidates and assemble outfits
    # Phase 1: BATCH all query embeddings in ONE API call
    # Phase 2: PARALLEL database queries (no more API calls)
    # Phase 3: SEQUENTIAL selection to avoid duplicate items
    logger.info("Building query embeddings (batched)...")
    directions = ["Classic", "Trendy", "Bold"]
    base_category = base_item.get("category", "top")
    
    # Build list of all (direction, slot) pairs and their query texts
    # Layer candidates are retrieved for all outfits - scoring decides if used
    retrieval_tasks = []
    query_texts = []
    for outfit_idx, direction in enumerate(directions):
        slots = get_slots_for_outfit(base_category, outfit_idx)
        for slot in slots:
            query_text = build_query_text(base_item, direction, slot, {})
            retrieval_tasks.append((outfit_idx, direction, slot))
            query_texts.append(query_text)
    
    # SINGLE API call for ALL embeddings (replaces 9-12 individual calls)
    logger.info(f"Batching {len(query_texts)} embeddings in one API call...")
    query_embeddings = get_batch_embeddings(query_texts)
    
    # Blend taste/dislike vectors if available (personalization)
    if taste_vector or dislike_vector:
        TASTE_WEIGHT = 0.25   # How much likes boost retrieval
        DISLIKE_WEIGHT = 0.15  # How much dislikes penalize (slightly less aggressive)
        blended_embeddings = []
        for emb in query_embeddings:
            blended = list(emb)  # Start with original
            
            # Add taste vector (boost liked styles)
            if taste_vector:
                blended = [
                    (1 - TASTE_WEIGHT) * blended[i] + TASTE_WEIGHT * taste_vector[i]
                    for i in range(len(blended))
                ]
            
            # Subtract dislike vector (penalize disliked styles)
            if dislike_vector:
                blended = [
                    blended[i] - DISLIKE_WEIGHT * dislike_vector[i]
                    for i in range(len(blended))
                ]
            
            blended_embeddings.append(blended)
        query_embeddings = blended_embeddings
        logger.info(f"Query embeddings personalized (taste={taste_vector is not None}, dislike={dislike_vector is not None})")
    
    task_embeddings = dict(zip([(t[0], t[1], t[2]) for t in retrieval_tasks], query_embeddings))
    logger.info("Embeddings ready, retrieving candidates...")
    
    def retrieve_task(task):
        """Retrieve candidates for one (direction, slot) pair using precomputed embedding."""
        outfit_idx, direction, slot = task
        precomputed_emb = task_embeddings.get((outfit_idx, direction, slot))
        try:
            candidates = retrieve_for_slot(
                base_item=base_item,
                direction=direction,
                slot=slot,
                exclude_ids=list(disliked_item_ids),  # Exclude previously disliked items
                chosen_items={},
                used_subtypes=set(),
                k=15,
                source=CATALOG_SOURCE,
                precomputed_embedding=precomputed_emb
            )
            logger.info(f"  [{direction}] {slot}: {len(candidates)} candidates")
            return (outfit_idx, direction, slot, candidates)
        except Exception as e:
            logger.error(f"  [{direction}] {slot}: Retrieval error - {e}")
            return (outfit_idx, direction, slot, [])
    
    # Phase 2: Run ALL database retrievals in parallel (no API calls now!)
    all_candidates = {}  # (outfit_idx, slot) -> candidates
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(retrieve_task, task) for task in retrieval_tasks]
        for future in as_completed(futures):
            outfit_idx, direction, slot, candidates = future.result()
            all_candidates[(outfit_idx, slot)] = candidates
    
    logger.info("All candidates retrieved, selecting items...")
    
    # Phase 3: Sequential selection with diversity tracking
    outfits_by_idx = {}  # Store by index to maintain correct output order
    used_ids_global = set()  # Track used item IDs across ALL outfits
    
    # Randomize selection order so different directions get first pick each time
    # This prevents the same outfit from just swapping categories on regeneration
    selection_order = list(range(len(directions)))
    random.shuffle(selection_order)
    logger.info(f"Selection order: {[directions[i] for i in selection_order]}")
    
    for outfit_idx in selection_order:
        direction = directions[outfit_idx]
        logger.info(f"Selecting {direction} outfit...")
        slots = get_slots_for_outfit(base_category, outfit_idx)
        
        # Filter out already-used items from candidates
        candidates_by_slot = {}
        for slot in slots:
            raw_candidates = all_candidates.get((outfit_idx, slot), [])
            # Remove items already used in previous outfits
            filtered = [c for c in raw_candidates if c["id"] not in used_ids_global]
            
            # Add strong randomness: fully shuffle all candidates
            # This ensures different items get considered each regeneration
            random.shuffle(filtered)
            
            candidates_by_slot[slot] = filtered
            if len(filtered) < len(raw_candidates):
                logger.info(f"    {slot}: {len(raw_candidates)} → {len(filtered)} after dedup")
        
        # Generate candidate outfits (combinations)
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8
        )
        logger.info(f"  [{direction}] Generated {len(candidate_outfits)} candidate outfits")
        
        # Score and select best outfit (with Fixes 1-3: intent vector, formality, diversity)
        best_items, score_details = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=embedding,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
        )
        
        # Log selection and track used IDs
        logger.info(f"  [{direction}] Best score: {score_details.get('total', 0):.3f}")
        for slot, item in best_items.items():
            if item:
                logger.info(f"    [{direction}] {slot}: #{item['id']} - {item['name'][:35]}")
                used_ids_global.add(item["id"])
        
        # Assemble final outfit (with enhanced scoring)
        outfit = assemble_outfit(
            direction, base_item, best_items, embedding,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
        )
        outfits_by_idx[outfit_idx] = outfit
    
    # Convert to list in correct order (Classic, Trendy, Bold)
    outfits = [outfits_by_idx[i] for i in range(len(directions))]
    logger.info("All outfits built with unique items")

    # 6. Store in database (get generation_id first for collages)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO outfit_generations 
               (input_image_url, input_image_hash, input_description, parsed_tags, base_item_embedding, output_outfits, input_type) 
               VALUES (%s, %s, %s, %s, %s, %s, %s) 
               RETURNING id""",
            (str(upload_path), image_hash, description, Json(base_item), embedding, Json(outfits), "image")
        )
        generation_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Generation created with id: {generation_id}")
        
        # 7. Generate collages for each outfit (include input image)
        logger.info("Generating outfit collages...")
        base_item_for_collage = {
            "image_url": str(upload_path),
            "category": base_item.get("category", "top")
        }
        
        for outfit in outfits:
            direction = outfit["direction"]
            items = outfit.get("items", [])
            
            try:
                collage_path = generate_outfit_collage(
                    generation_id, 
                    direction, 
                    items,
                    base_item=base_item_for_collage
                )
                outfit["collage_url"] = make_absolute_url(base_url, collage_path)
                logger.info(f"  {direction} collage: {outfit['collage_url']}")
            except Exception as e:
                logger.error(f"  {direction} collage failed: {e}")
                outfit["collage_url"] = None
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

    # 8. Convert all image URLs to absolute
    for outfit in outfits:
        for item in outfit.get("items", []):
            if item.get("image_url"):
                item["image_url"] = make_absolute_url(base_url, item["image_url"])

    # 9. Return response
    return {
        "generation_id": generation_id,
        "base_item": base_item,
        "description": description,
        "outfits": outfits
    }


@app.post("/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit like/dislike feedback for a generated outfit (upsert - one per generation+outfit)."""
    logger.info(f"Feedback received: gen={req.generation_id}, outfit={req.outfit_index}, liked={req.liked}, session={req.session_id[:8] if req.session_id else 'none'}...")
    
    if req.outfit_index not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="outfit_index must be 0, 1, or 2")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get generation with outfit data for taste vector
        cursor.execute(
            "SELECT output_outfits FROM outfit_generations WHERE id = %s", 
            (req.generation_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Generation not found")
        
        # Upsert feedback (insert or update if exists)
        cursor.execute(
            """INSERT INTO feedback_events (generation_id, outfit_index, liked, session_id, updated_at) 
               VALUES (%s, %s, %s, %s, NOW()) 
               ON CONFLICT (generation_id, outfit_index) 
               DO UPDATE SET liked = EXCLUDED.liked, session_id = EXCLUDED.session_id, updated_at = NOW()
               RETURNING id""",
            (req.generation_id, req.outfit_index, req.liked, req.session_id)
        )
        feedback_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Feedback stored/updated: id={feedback_id}")
        
        # Update taste vector if session provided
        if req.session_id and row[0]:
            outfits = row[0]
            if req.outfit_index < len(outfits):
                outfit = outfits[req.outfit_index]
                # Extract item embeddings from the outfit
                item_embeddings = []
                for item in outfit.get("items", []):
                    if item and item.get("embedding"):
                        item_embeddings.append(item["embedding"])
                
                if item_embeddings:
                    update_taste_vector(req.session_id, item_embeddings, req.liked)
        
        return {"feedback_id": feedback_id, "status": "recorded"}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Feedback DB error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


@app.get("/v1/feedback/stats")
async def get_feedback_stats():
    """Get aggregated feedback stats per direction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                outfit_index,
                COUNT(*) FILTER (WHERE liked = true) as likes,
                COUNT(*) FILTER (WHERE liked = false) as dislikes,
                COUNT(*) as total
            FROM feedback_events
            GROUP BY outfit_index
            ORDER BY outfit_index
        """)
        rows = cursor.fetchall()
        
        directions = ["Classic", "Trendy", "Bold"]
        stats = {}
        for idx, likes, dislikes, total in rows:
            if idx < len(directions):
                stats[directions[idx]] = {
                    "likes": likes,
                    "dislikes": dislikes,
                    "total": total,
                    "like_rate": round(likes / total * 100, 1) if total > 0 else 0
                }
        return stats
    finally:
        cursor.close()
        conn.close()


# =====================
# WEATHER API ENDPOINT
# =====================

@app.get("/v1/weather")
async def get_weather(lat: float, lon: float):
    """
    Get current weather and outfit recommendations.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Weather data with outfit adjustments
    """
    weather = await fetch_weather(lat, lon)
    
    if not weather:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    
    adjustments = get_weather_outfit_adjustments(weather)
    
    return {
        "weather": weather.to_dict(),
        "outfit_adjustments": adjustments
    }


# =====================
# CLOSET API ENDPOINTS
# =====================

import cloudinary
import cloudinary.uploader

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


@app.get("/v1/closet/items")
async def list_closet_items(user_id: str = "default"):
    """List all items in user's closet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color, secondary_colors,
                      style_tags, season_tags, occasion_tags, material, fit, created_at
               FROM user_closet_items 
               WHERE user_id = %s
               ORDER BY created_at DESC""",
            (user_id,)
        )
        rows = cursor.fetchall()
        
        items = []
        for row in rows:
            items.append({
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "image_url": row[3],
                "primary_color": row[4],
                "secondary_colors": row[5],
                "style_tags": row[6],
                "season_tags": row[7],
                "occasion_tags": row[8],
                "material": row[9],
                "fit": row[10],
                "created_at": row[11].isoformat() if row[11] else None
            })
        
        return {"items": items, "count": len(items)}
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/closet/items")
async def add_closet_item(file: UploadFile = File(...), user_id: str = Form("default")):
    """
    Upload a new item to user's closet.
    Processes image through vision/parser pipeline and stores with embedding.
    """
    logger.info(f"Adding closet item for user: {user_id}")
    
    # Read and validate file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image uploaded")
    
    # 1. Process image locally (background removal, trim, pad)
    try:
        logger.info("Processing image locally...")
        processed_bytes = process_clothing_image(contents)
        logger.info("Image processed, uploading to Cloudinary...")
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        # Fall back to uploading original if processing fails
        processed_bytes = contents
    
    # 2. Upload processed image to Cloudinary (just storage, no transformations)
    try:
        upload_result = cloudinary.uploader.upload(
            processed_bytes,
            folder="closet",
            resource_type="image"
        )
        image_url = upload_result["secure_url"]
        logger.info(f"Uploaded to Cloudinary: {image_url}")
    except Exception as e:
        logger.error(f"Cloudinary upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    
    # 2. Vision: Get description from image
    try:
        description = describe_image(contents)
        logger.info(f"Vision description: {description[:100]}...")
    except Exception as e:
        logger.error(f"Vision API error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision API error: {str(e)}")
    
    # 3. Parser: Convert description to structured JSON
    try:
        parsed = parse_description(description)
        logger.info(f"Parsed item: {parsed}")
    except Exception as e:
        logger.error(f"Parser error: {e}")
        raise HTTPException(status_code=500, detail=f"Parser error: {str(e)}")
    
    # 4. Embedding: Generate embedding
    try:
        embedding = embed_base_item(parsed)
        logger.info(f"Embedding generated (dim={len(embedding)})")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
    
    # 5. Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Generate name from description
        name = f"{parsed.get('primary_color', '')} {parsed.get('category', 'item')}".strip().title()
        
        cursor.execute(
            """INSERT INTO user_closet_items 
               (user_id, name, category, image_url, primary_color, secondary_colors,
                style_tags, season_tags, occasion_tags, material, fit, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                user_id,
                name,
                parsed.get("category", "top"),
                image_url,
                parsed.get("primary_color"),
                parsed.get("secondary_colors"),
                parsed.get("style_tags"),
                parsed.get("season_tags"),
                parsed.get("occasion_tags"),
                parsed.get("material"),
                parsed.get("fit"),
                embedding
            )
        )
        item_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Closet item created: id={item_id}")
        
        return {
            "id": item_id,
            "name": name,
            "category": parsed.get("category"),
            "image_url": image_url,
            "parsed": parsed
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


@app.delete("/v1/closet/items/{item_id}")
async def delete_closet_item(item_id: int, user_id: str = "default"):
    """Delete an item from user's closet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM user_closet_items WHERE id = %s AND user_id = %s RETURNING id",
            (item_id, user_id)
        )
        deleted = cursor.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        
        conn.commit()
        logger.info(f"Closet item deleted: id={item_id}")
        return {"deleted": item_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


class RotateRequest(BaseModel):
    image_url: str


@app.post("/v1/closet/items/{item_id}/rotate")
async def rotate_closet_item(item_id: int, req: RotateRequest, user_id: str = "default"):
    """Update item's image URL with rotation transformation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE user_closet_items SET image_url = %s WHERE id = %s AND user_id = %s RETURNING id",
            (req.image_url, item_id, user_id)
        )
        updated = cursor.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail="Item not found")
        
        conn.commit()
        logger.info(f"Closet item rotated: id={item_id}")
        return {"id": item_id, "image_url": req.image_url}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Rotate error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()


@app.get("/v1/closet/items/{item_id}")
async def get_closet_item(item_id: int, user_id: str = "default"):
    """Get a single closet item with its embedding."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color, secondary_colors,
                      style_tags, season_tags, occasion_tags, material, fit,
                      embedding::text as embedding_text
               FROM user_closet_items 
               WHERE id = %s AND user_id = %s""",
            (item_id, user_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item = {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "image_url": row[3],
            "primary_color": row[4],
            "secondary_colors": row[5],
            "style_tags": row[6],
            "season_tags": row[7],
            "occasion_tags": row[8],
            "material": row[9],
            "fit": row[10],
        }
        
        # Parse embedding
        if row[11]:
            item["embedding"] = [float(x) for x in row[11].strip("[]").split(",")]
        
        return item
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/closet/outfits:generate")
async def generate_closet_outfits(
    request: Request, 
    item_id: int = Form(None),
    file: UploadFile = File(None),
    user_id: str = Form("default"),
    lat: float = Form(None),
    lon: float = Form(None)
):
    """
    Generate outfits using ONLY items from user's closet.
    Either pick an existing closet item (item_id) or upload a new image.
    Optionally pass lat/lon to factor in weather.
    """
    logger.info(f"Closet outfit generation: item_id={item_id}, has_file={file is not None}, weather={lat},{lon}")
    
    base_url = str(request.base_url).rstrip("/")
    
    # Get base item - either from closet or uploaded
    if item_id:
        # Use existing closet item
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """SELECT id, name, category, image_url, primary_color, secondary_colors,
                          style_tags, season_tags, occasion_tags, material, fit,
                          embedding::text
                   FROM user_closet_items 
                   WHERE id = %s AND user_id = %s""",
                (item_id, user_id)
            )
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Closet item not found")
            
            base_item = {
                "category": row[2],
                "primary_color": row[4],
                "secondary_colors": row[5],
                "style_tags": row[6],
                "season_tags": row[7],
                "occasion_tags": row[8],
                "material": row[9],
                "fit": row[10],
            }
            embedding = [float(x) for x in row[11].strip("[]").split(",")] if row[11] else None
            input_image_url = row[3]
            description = row[1]
            image_hash = f"closet_{item_id}"
        finally:
            cursor.close()
            conn.close()
    elif file:
        # Upload new image and add to closet
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="No image uploaded")
        
        # Process image locally and upload to Cloudinary
        try:
            logger.info("Processing image locally...")
            processed_bytes = process_clothing_image(contents)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            processed_bytes = contents
        
        try:
            upload_result = cloudinary.uploader.upload(
                processed_bytes,
                folder="closet",
                resource_type="image"
            )
            input_image_url = upload_result["secure_url"]
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
        
        # Process through vision/parser pipeline
        description = describe_image(contents)
        base_item = parse_description(description)
        embedding = embed_base_item(base_item)
        image_hash = hashlib.sha256(contents).hexdigest()
        
        # Add to closet
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            name = f"{base_item.get('primary_color', '')} {base_item.get('category', 'item')}".strip().title()
            cursor.execute(
                """INSERT INTO user_closet_items 
                   (user_id, name, category, image_url, primary_color, secondary_colors,
                    style_tags, season_tags, occasion_tags, material, fit, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    user_id, name, base_item.get("category", "top"), input_image_url,
                    base_item.get("primary_color"), base_item.get("secondary_colors"),
                    base_item.get("style_tags"), base_item.get("season_tags"),
                    base_item.get("occasion_tags"), base_item.get("material"),
                    base_item.get("fit"), embedding
                )
            )
            new_item_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added uploaded item to closet: id={new_item_id}")
        finally:
            cursor.close()
            conn.close()
    else:
        raise HTTPException(status_code=400, detail="Provide either item_id or file")
    
    if not embedding:
        raise HTTPException(status_code=400, detail="Item has no embedding")
    
    # Fetch weather if location provided
    weather_data = None
    weather_adjustments = None
    if lat is not None and lon is not None:
        weather_data = await fetch_weather(lat, lon)
        if weather_data:
            weather_adjustments = get_weather_outfit_adjustments(weather_data)
            logger.info(f"Weather: {weather_data.city} {weather_data.temperature_c}°C - {weather_adjustments['notes']}")
    
    # Build query embeddings for closet retrieval
    directions = ["Classic", "Trendy", "Bold"]
    base_category = base_item.get("category", "top")
    
    retrieval_tasks = []
    query_texts = []
    for outfit_idx, direction in enumerate(directions):
        slots = get_slots_for_outfit(base_category, outfit_idx)
        for slot in slots:
            query_text = build_query_text(base_item, direction, slot, {})
            retrieval_tasks.append((outfit_idx, direction, slot))
            query_texts.append(query_text)
    
    # Batch embed all queries
    logger.info(f"Batching {len(query_texts)} embeddings for closet retrieval...")
    query_embeddings = get_batch_embeddings(query_texts)
    task_embeddings = dict(zip([(t[0], t[1], t[2]) for t in retrieval_tasks], query_embeddings))
    
    # Retrieve from CLOSET only (reuses same function with use_closet=True)
    def retrieve_closet_task(task):
        outfit_idx, direction, slot = task
        precomputed_emb = task_embeddings.get((outfit_idx, direction, slot))
        try:
            candidates = retrieve_for_slot(
                base_item=base_item,
                direction=direction,
                slot=slot,
                exclude_ids=[],
                chosen_items={},
                k=10,
                precomputed_embedding=precomputed_emb,
                use_closet=True,
                user_id=user_id
            )
            logger.info(f"  [{direction}] {slot}: {len(candidates)} closet candidates")
            return (outfit_idx, direction, slot, candidates)
        except Exception as e:
            logger.error(f"  [{direction}] {slot}: Retrieval error - {e}")
            return (outfit_idx, direction, slot, [])
    
    # Parallel closet retrieval
    all_candidates = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(retrieve_closet_task, task) for task in retrieval_tasks]
        for future in as_completed(futures):
            outfit_idx, direction, slot, candidates = future.result()
            all_candidates[(outfit_idx, slot)] = candidates
    
    logger.info("Closet candidates retrieved, selecting items...")
    
    # Sequential selection with diversity
    outfits_by_idx = {}
    used_ids_global = set()
    
    selection_order = list(range(len(directions)))
    random.shuffle(selection_order)
    
    for outfit_idx in selection_order:
        direction = directions[outfit_idx]
        slots = get_slots_for_outfit(base_category, outfit_idx)
        
        # Adjust slots based on weather
        if weather_adjustments:
            if weather_adjustments["force_layer"] and "layer" not in slots:
                slots = slots + ["layer"]
            elif weather_adjustments["skip_layer"] and "layer" in slots:
                slots = [s for s in slots if s != "layer"]
        
        # Filter out used items
        candidates_by_slot = {}
        for slot in slots:
            raw = all_candidates.get((outfit_idx, slot), [])
            filtered = [c for c in raw if c["id"] not in used_ids_global]
            random.shuffle(filtered)
            candidates_by_slot[slot] = filtered
        
        # Generate and score candidate outfits
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8
        )
        
        if not candidate_outfits:
            logger.warning(f"  [{direction}] No valid outfit combinations from closet")
            outfits_by_idx[outfit_idx] = {
                "direction": direction,
                "items": [],
                "explanation": "Not enough items in closet for this outfit style.",
                "score": 0
            }
            continue
        
        best_items, score_details = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=embedding
        )
        
        # Track used IDs
        for slot, item in best_items.items():
            if item:
                used_ids_global.add(item["id"])
        
        outfit = assemble_outfit(direction, base_item, best_items, embedding)
        outfits_by_idx[outfit_idx] = outfit
    
    outfits = [outfits_by_idx[i] for i in range(len(directions))]
    
    # Store generation
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO outfit_generations 
               (input_image_url, input_image_hash, input_description, parsed_tags, 
                base_item_embedding, output_outfits, input_type) 
               VALUES (%s, %s, %s, %s, %s, %s, %s) 
               RETURNING id""",
            (input_image_url, image_hash, description, Json(base_item), embedding, Json(outfits), "closet")
        )
        generation_id = cursor.fetchone()[0]
        conn.commit()
        
        # Generate collages
        base_item_for_collage = {
            "image_url": input_image_url,
            "category": base_category
        }
        
        for outfit in outfits:
            direction = outfit["direction"]
            items = outfit.get("items", [])
            try:
                collage_path = generate_outfit_collage(
                    generation_id, direction, items,
                    base_item=base_item_for_collage
                )
                outfit["collage_url"] = make_absolute_url(base_url, collage_path)
            except Exception as e:
                logger.error(f"  {direction} collage failed: {e}")
                outfit["collage_url"] = None
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()
    
    response = {
        "generation_id": generation_id,
        "base_item": base_item,
        "description": description,
        "outfits": outfits,
        "source": "closet"
    }
    
    # Include weather info if available
    if weather_data:
        response["weather"] = weather_data.to_dict()
        response["weather_notes"] = weather_adjustments["notes"] if weather_adjustments else []
    
    return response


@app.get("/v1/closet/daily")
async def get_daily_outfits(
    request: Request,
    lat: float = None,
    lon: float = None,
    user_id: str = "default"
):
    """
    Generate 3 weather-appropriate outfits automatically.
    Picks different base items from closet for variety.
    """
    logger.info(f"Daily outfits: lat={lat}, lon={lon}")
    base_url = str(request.base_url).rstrip("/")
    
    # Fetch weather
    weather_data = None
    weather_adjustments = None
    if lat is not None and lon is not None:
        weather_data = await fetch_weather(lat, lon)
        if weather_data:
            weather_adjustments = get_weather_outfit_adjustments(weather_data)
            logger.info(f"Weather: {weather_data.city} {weather_data.temperature_c}°C")
    
    # Get all closet items
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color, secondary_colors,
                      style_tags, season_tags, occasion_tags, material, fit, embedding::text
               FROM user_closet_items 
               WHERE user_id = %s AND embedding IS NOT NULL
               ORDER BY created_at DESC""",
            (user_id,)
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    
    if len(rows) < 3:
        return {
            "outfits": [],
            "weather": weather_data.to_dict() if weather_data else None,
            "message": "Need at least 3 items in closet for daily outfits"
        }
    
    # Parse items
    all_items = []
    for row in rows:
        item = {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "image_url": row[3],
            "primary_color": row[4],
            "secondary_colors": row[5],
            "style_tags": row[6],
            "season_tags": row[7] or [],
            "occasion_tags": row[8],
            "material": row[9],
            "fit": row[10],
            "embedding": [float(x) for x in row[11].strip("[]").split(",")] if row[11] else None
        }
        all_items.append(item)
    
    # Filter items by weather season if available
    preferred_seasons = weather_adjustments.get("preferred_seasons", []) if weather_adjustments else []
    avoid_seasons = weather_adjustments.get("avoid_seasons", []) if weather_adjustments else []
    
    def season_score(item):
        tags = item.get("season_tags") or []
        score = 0
        for s in preferred_seasons:
            if s in tags or "all-season" in tags:
                score += 1
        for s in avoid_seasons:
            if s in tags:
                score -= 2
        return score
    
    # Pick 3 different base items (prefer tops/layers for weather)
    base_categories = ["top", "layer", "bottom"]
    if weather_adjustments and weather_adjustments.get("force_layer"):
        base_categories = ["layer", "top", "bottom"]  # Prioritize layers in cold
    elif weather_adjustments and weather_adjustments.get("skip_layer"):
        base_categories = ["top", "bottom", "dress"]  # Skip layers in hot
    
    selected_bases = []
    used_ids = set()
    
    for pref_cat in base_categories:
        candidates = [i for i in all_items if i["category"] == pref_cat and i["id"] not in used_ids]
        if candidates:
            # Sort by season appropriateness
            candidates.sort(key=season_score, reverse=True)
            selected = candidates[0]
            selected_bases.append(selected)
            used_ids.add(selected["id"])
        if len(selected_bases) >= 3:
            break
    
    # Fill remaining with any category
    if len(selected_bases) < 3:
        remaining = [i for i in all_items if i["id"] not in used_ids]
        remaining.sort(key=season_score, reverse=True)
        for item in remaining:
            selected_bases.append(item)
            used_ids.add(item["id"])
            if len(selected_bases) >= 3:
                break
    
    if len(selected_bases) < 1:
        return {
            "outfits": [],
            "weather": weather_data.to_dict() if weather_data else None,
            "message": "Not enough items in closet"
        }
    
    # Generate one outfit per base item
    outfits = []
    used_ids_global = set(i["id"] for i in selected_bases)
    
    directions = ["Classic", "Trendy", "Bold"]
    
    for idx, base_item in enumerate(selected_bases[:3]):
        direction = directions[idx] if idx < len(directions) else "Classic"
        base_category = base_item["category"]
        embedding = base_item.get("embedding")
        
        if not embedding:
            continue
        
        # Get slots for this outfit
        slots = get_slots_for_outfit(base_category, idx)
        
        # Adjust for weather
        if weather_adjustments:
            if weather_adjustments["force_layer"] and "layer" not in slots and base_category != "layer":
                slots = slots + ["layer"]
            elif weather_adjustments["skip_layer"] and "layer" in slots:
                slots = [s for s in slots if s != "layer"]
        
        # Build query texts
        query_texts = []
        for slot in slots:
            query_text = build_query_text(base_item, direction, slot, {})
            query_texts.append(query_text)
        
        # Get embeddings
        query_embeddings = get_batch_embeddings(query_texts)
        
        # Retrieve candidates from closet
        candidates_by_slot = {}
        for i, slot in enumerate(slots):
            try:
                candidates = retrieve_for_slot(
                    base_item=base_item,
                    direction=direction,
                    slot=slot,
                    exclude_ids=list(used_ids_global),
                    chosen_items={},
                    k=10,
                    precomputed_embedding=query_embeddings[i],
                    use_closet=True,
                    user_id=user_id
                )
                candidates_by_slot[slot] = candidates
            except Exception as e:
                logger.error(f"Retrieval error for {slot}: {e}")
                candidates_by_slot[slot] = []
        
        # Generate and score
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8
        )
        
        if not candidate_outfits:
            outfit = {
                "direction": f"Outfit {idx + 1}",
                "base_item": base_item,
                "items": [{"slot": base_category, **base_item}],
                "explanation": "Limited items in closet",
                "collage_url": None
            }
        else:
            best_items, score_details = select_best_outfit(
                candidate_outfits=candidate_outfits,
                base_item=base_item,
                direction=direction,
                base_embedding=embedding
            )
            
            # Track used items
            for slot, item in best_items.items():
                if item:
                    used_ids_global.add(item["id"])
            
            outfit = assemble_outfit(direction, base_item, best_items, embedding)
            outfit["direction"] = f"Outfit {idx + 1}"
            outfit["base_item"] = base_item
        
        # Generate collage
        try:
            items_for_collage = outfit.get("items", [])
            collage_path = generate_outfit_collage(
                generation_id=f"daily_{idx}",
                direction=f"outfit_{idx + 1}",
                items=items_for_collage,
                base_item={"image_url": base_item["image_url"], "category": base_category}
            )
            outfit["collage_url"] = make_absolute_url(base_url, collage_path)
        except Exception as e:
            logger.error(f"Collage error: {e}")
            outfit["collage_url"] = None
        
        outfits.append(outfit)
    
    response = {
        "outfits": outfits,
        "source": "closet_daily"
    }
    
    if weather_data:
        response["weather"] = weather_data.to_dict()
        response["weather_notes"] = weather_adjustments["notes"] if weather_adjustments else []
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
