from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import logging
import json
from pathlib import Path
import psycopg2
from psycopg2.extras import Json

from services.vision import describe_image
from services.parser import parse_description
from services.embedding import embed_base_item

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

    # 5. Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO outfit_generations 
               (input_image_url, input_description, parsed_tags, base_item_embedding, input_type) 
               VALUES (%s, %s, %s, %s, %s) 
               RETURNING id""",
            (str(upload_path), description, Json(base_item), embedding, "image")
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

    # 6. Return response (placeholder outfits for now)
    return {
        "generation_id": generation_id,
        "base_item": base_item,
        "description": description,
        "outfits": [
            {"direction": "Classic", "items": []},
            {"direction": "Trendy", "items": []},
            {"direction": "Bold", "items": []}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
