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
async def serve_index():
    """Serve the main frontend page."""
    return FileResponse("static/index.html")


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
                k=15,  # Get extra candidates for diversity across outfits
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
    outfits = []
    used_ids_global = set()  # Track used item IDs across ALL outfits
    
    for outfit_idx, direction in enumerate(directions):
        logger.info(f"Selecting {direction} outfit...")
        slots = get_slots_for_outfit(base_category, outfit_idx)
        
        # Filter out already-used items from candidates
        candidates_by_slot = {}
        for slot in slots:
            raw_candidates = all_candidates.get((outfit_idx, slot), [])
            # Remove items already used in previous outfits
            filtered = [c for c in raw_candidates if c["id"] not in used_ids_global]
            
            # Add randomness: shuffle top candidates to get variety across generations
            # Keep top 10, shuffle them, then add the rest
            if len(filtered) > 5:
                top_candidates = filtered[:10]
                rest = filtered[10:]
                random.shuffle(top_candidates)
                filtered = top_candidates + rest
            
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
        
        # Score and select best outfit
        best_items, score_details = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=embedding
        )
        
        # Log selection and track used IDs
        logger.info(f"  [{direction}] Best score: {score_details.get('total', 0):.3f}")
        for slot, item in best_items.items():
            if item:
                logger.info(f"    [{direction}] {slot}: #{item['id']} - {item['name'][:35]}")
                used_ids_global.add(item["id"])
        
        # Assemble final outfit
        outfit = assemble_outfit(direction, base_item, best_items, embedding)
        outfits.append(outfit)
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
