from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import logging
from pathlib import Path
import psycopg2

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
    with open(upload_path, "wb") as f:
        f.write(contents)
    logger.info(f"Image stored at: {upload_path}")

    # 2. Insert row into database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO outfit_generations (input_image_url) VALUES (%s) RETURNING id",
        (str(upload_path),)
    )
    generation_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Generation created with id: {generation_id}")

    # 3. Return dummy response
    return {
        "generation_id": generation_id,
        "outfits": [
            {"direction": "Classic", "items": []},
            {"direction": "Trendy", "items": []},
            {"direction": "Bold", "items": []}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

