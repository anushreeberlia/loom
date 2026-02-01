from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uuid
import logging
from pathlib import Path
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
from services.retrieval import retrieve_for_slot, extract_item_subtype


class FeedbackRequest(BaseModel):
    generation_id: int
    outfit_index: int  # 0, 1, or 2
    liked: bool

app = FastAPI(title="AI Outfit Styler")

DATABASE_URL = "postgresql://localhost:5432/outfit_styler"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/outfits:generate")
async def generate_outfits(file: UploadFile = File(...)):
    logger.info("Request received: POST /v1/outfits:generate")

    # Validate file
    contents = await file.read()
    if not contents:
        logger.warning("No image uploaded or file is empty")
        raise HTTPException(status_code=400, detail="No image uploaded or file is empty")

    # 1. Save uploaded file
    ext = file.filename.split(".")[-1] if file.filename else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    upload_path = Path("uploads") / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(contents)
    logger.info(f"Image stored at: {upload_path}")

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

    # 5. Retrieve candidates and assemble outfits using candidate scoring
    logger.info("Retrieving outfit candidates...")
    outfits = []
    directions = ["Classic", "Trendy", "Bold"]
    used_ids_by_slot = {}  # Track used item IDs for diversity
    used_subtypes_by_slot = {}  # Track used item subtypes for variety
    
    base_category = base_item.get("category", "top")
    
    for outfit_idx, direction in enumerate(directions):
        logger.info(f"Building {direction} outfit...")
        
        slots = get_slots_for_outfit(base_category, outfit_idx)
        all_candidates_by_slot = {}
        
        # Step 1: Retrieve candidates for all slots
        for slot in slots:
            exclude_ids = list(used_ids_by_slot.get(slot, set()))
            used_subtypes = used_subtypes_by_slot.get(slot, set())
            
            try:
                candidates = retrieve_for_slot(
                    base_item=base_item,
                    direction=direction,
                    slot=slot,
                    exclude_ids=exclude_ids,
                    chosen_items={},  # No sequential conditioning in candidate phase
                    used_subtypes=used_subtypes,
                    k=10  # Get more candidates for scoring
                )
                all_candidates_by_slot[slot] = candidates
                logger.info(f"  {slot}: {len(candidates)} candidates retrieved")
                
            except Exception as e:
                logger.error(f"  {slot}: Retrieval error - {e}")
                all_candidates_by_slot[slot] = []
        
        # Step 2: Generate candidate outfits (combinations)
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=all_candidates_by_slot,
            max_candidates=8
        )
        logger.info(f"  Generated {len(candidate_outfits)} candidate outfits")
        
        # Step 3: Score and select best outfit
        best_items, score_details = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=embedding
        )
        
        # Log selection
        logger.info(f"  Best outfit score: {score_details.get('total', 0):.3f}")
        if score_details.get("violations"):
            logger.warning(f"  Violations: {score_details['violations']}")
        
        for slot, item in best_items.items():
            if item:
                logger.info(f"    {slot}: #{item['id']} - {item['name'][:35]} ({item.get('primary_color', '?')})")
                
                # Track for diversity across outfits
                if slot not in used_ids_by_slot:
                    used_ids_by_slot[slot] = set()
                used_ids_by_slot[slot].add(item["id"])
                
                subtype = extract_item_subtype(item.get("name", ""), slot)
                if subtype:
                    if slot not in used_subtypes_by_slot:
                        used_subtypes_by_slot[slot] = set()
                    used_subtypes_by_slot[slot].add(subtype)
        
        # Step 4: Assemble final outfit
        outfit = assemble_outfit(direction, base_item, best_items, embedding)
        outfits.append(outfit)

    # 6. Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO outfit_generations 
               (input_image_url, input_description, parsed_tags, base_item_embedding, output_outfits, input_type) 
               VALUES (%s, %s, %s, %s, %s, %s) 
               RETURNING id""",
            (str(upload_path), description, Json(base_item), embedding, Json(outfits), "image")
        )
        generation_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Generation created with id: {generation_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

    # 7. Return response
    return {
        "generation_id": generation_id,
        "base_item": base_item,
        "description": description,
        "outfits": outfits
    }


@app.post("/v1/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit like/dislike feedback for a generated outfit."""
    logger.info(f"Feedback received: gen={req.generation_id}, outfit={req.outfit_index}, liked={req.liked}")
    
    if req.outfit_index not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="outfit_index must be 0, 1, or 2")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Verify generation exists
        cursor.execute("SELECT id FROM outfit_generations WHERE id = %s", (req.generation_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Generation not found")
        
        # Insert feedback
        cursor.execute(
            """INSERT INTO feedback_events (generation_id, outfit_index, liked) 
               VALUES (%s, %s, %s) 
               RETURNING id""",
            (req.generation_id, req.outfit_index, req.liked)
        )
        feedback_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Feedback stored: id={feedback_id}")
        
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
