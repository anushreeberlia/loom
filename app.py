from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Cookie, Response, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import uuid
import random
import logging
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import Json

from services.vision import analyze_image, describe_image
from services.parser import parse_description
from services.embedding import embed_base_item, embed_item_image
from services.outfit import get_slots_for_outfit, assemble_outfit
from services.collage import generate_outfit_collage
from services.weather import fetch_weather, get_weather_outfit_adjustments, get_occasion_from_time, get_material_weather_score, WeatherData
from services.image_processor import process_clothing_image
from services.ingestion_worker import process_batch_items, recover_stale_items, orphan_sweep_loop
from services.auth import (
    get_google_auth_url,
    exchange_code_for_tokens,
    get_google_user_info,
    create_jwt_token,
    verify_jwt_token,
    get_user_id_from_token
)

import os
import httpx
import bcrypt
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# Password hashing helpers
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))



class FeedbackRequest(BaseModel):
    generation_id: int
    outfit_index: int  # 0, 1, or 2
    liked: bool
    session_id: str = None  # For taste vector tracking

class ClosetFeedbackRequest(BaseModel):
    item_ids: list[int]  # IDs of items in the outfit
    liked: bool
    user_id: str = "default"

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = None

class LoginRequest(BaseModel):
    email: str
    password: str

app = FastAPI(title="AI Outfit Styler")

from shopify_app import app as shopify_app_instance


@app.on_event("startup")
async def _startup():
    from services.fashion_clip import warmup
    warmup()
    recover_stale_items()
    asyncio.create_task(orphan_sweep_loop())


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
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


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
    from services.fashion_clip import EMBEDDING_DIM
    valid_embeddings = [e for e in item_embeddings if e and len(e) == EMBEDDING_DIM]
    if not valid_embeddings:
        return
    
    # Average the item embeddings from this outfit
    outfit_embedding = [
        sum(e[i] for e in valid_embeddings) / len(valid_embeddings)
        for i in range(EMBEDDING_DIM)
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
                    for i in range(EMBEDDING_DIM)
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
                    for i in range(EMBEDDING_DIM)
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


# ============== AUTH ENDPOINTS ==============

def get_current_user(auth_token: Optional[str] = Cookie(None)) -> Optional[dict]:
    """Get current user from auth cookie"""
    if not auth_token:
        return None
    payload = verify_jwt_token(auth_token)
    if not payload:
        return None
    return payload


def get_or_create_user(email: str, name: str, google_id: str, profile_image: str = None) -> dict:
    """Get existing user or create new one. Migrates 'default' closet data to new user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if user exists
        cursor.execute("SELECT id, email, name, profile_image FROM users WHERE google_id = %s", (google_id,))
        row = cursor.fetchone()
        
        if row:
            # Update last login
            cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (row[0],))
            conn.commit()
            return {"id": row[0], "email": row[1], "name": row[2], "profile_image": row[3]}
        
        # Create new user
        cursor.execute(
            """INSERT INTO users (email, name, google_id, profile_image) 
               VALUES (%s, %s, %s, %s) 
               RETURNING id""",
            (email, name, google_id, profile_image)
        )
        user_id = cursor.fetchone()[0]
        
        # Migrate "default" closet data to this new user
        cursor.execute(
            """UPDATE user_closet_items SET user_id = %s WHERE user_id = 'default'""",
            (str(user_id),)
        )
        migrated_count = cursor.rowcount
        
        # Also migrate taste vectors
        cursor.execute(
            """UPDATE taste_vectors SET session_id = %s WHERE session_id = 'default'""",
            (str(user_id),)
        )
        
        conn.commit()
        
        logger.info(f"Created new user: {email} (id={user_id}), migrated {migrated_count} closet items")
        return {"id": user_id, "email": email, "name": name, "profile_image": profile_image}
    except Exception as e:
        logger.error(f"User creation error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def require_auth(auth_token: Optional[str]) -> int:
    """
    Require authentication and return user_id.
    Raises HTTPException 401 if not authenticated.
    """
    if not auth_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    
    payload = verify_jwt_token(auth_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please sign in again.")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token. Please sign in again.")
    
    return str(user_id)


def migrate_default_data_to_user(user_id: int):
    """Migrate closet items and taste vectors from 'default' to a new user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if this user already has items (to prevent re-migration)
        cursor.execute("SELECT COUNT(*) FROM user_closet_items WHERE user_id = %s", (str(user_id),))
        if cursor.fetchone()[0] > 0:
            logger.info(f"User {user_id} already has closet items, skipping migration")
            return
        
        # Check if there are default items to migrate
        cursor.execute("SELECT COUNT(*) FROM user_closet_items WHERE user_id = 'default'")
        default_count = cursor.fetchone()[0]
        if default_count == 0:
            logger.info("No default items to migrate")
            return
        
        # Migrate closet items from 'default' to user
        cursor.execute(
            "UPDATE user_closet_items SET user_id = %s WHERE user_id = 'default'",
            (str(user_id),)
        )
        migrated_items = cursor.rowcount
        logger.info(f"Migrated {migrated_items} closet items from 'default' to user {user_id}")
        
        # Migrate taste vectors
        cursor.execute(
            "UPDATE taste_vectors SET session_id = %s WHERE session_id = 'default'",
            (str(user_id),)
        )
        migrated_taste = cursor.rowcount
        logger.info(f"Migrated {migrated_taste} taste vectors from 'default' to user {user_id}")
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error migrating default data to user {user_id}: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/auth/google")
async def google_login(request: Request):
    """Redirect to Google OAuth"""
    # Force HTTPS for production (Railway proxy reports http)
    base = str(request.base_url).rstrip("/")
    if base.startswith("http://") and ("railway.app" in base or "loom-style.com" in base):
        base = base.replace("http://", "https://", 1)
    redirect_uri = base + "/auth/google/callback"
    auth_url = get_google_auth_url(redirect_uri)
    return RedirectResponse(url=auth_url)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = None, error: str = None):
    """Handle Google OAuth callback"""
    if error:
        logger.error(f"Google OAuth error: {error}")
        return RedirectResponse(url="/?error=auth_failed")
    
    if not code:
        return RedirectResponse(url="/?error=no_code")
    
    try:
        # Force HTTPS for production (Railway proxy reports http)
        base = str(request.base_url).rstrip("/")
        if base.startswith("http://") and ("railway.app" in base or "loom-style.com" in base):
            base = base.replace("http://", "https://", 1)
        redirect_uri = base + "/auth/google/callback"
        
        # Exchange code for tokens
        tokens = await exchange_code_for_tokens(code, redirect_uri)
        access_token = tokens.get("access_token")
        
        if not access_token:
            raise Exception("No access token received")
        
        # Get user info from Google
        google_user = await get_google_user_info(access_token)
        
        # Create or get user in our DB
        user = get_or_create_user(
            email=google_user.get("email"),
            name=google_user.get("name"),
            google_id=google_user.get("id"),
            profile_image=google_user.get("picture")
        )
        
        # Migrate default closet data to this user (first login)
        migrate_default_data_to_user(user["id"])
        
        # Create JWT token
        jwt_token = create_jwt_token(user["id"], user["email"])
        
        # Set cookie and redirect to closet
        response = RedirectResponse(url="/closet")
        response.set_cookie(
            key="auth_token",
            value=jwt_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 7  # 1 week
        )
        return response
        
    except Exception as e:
        logger.error(f"Google callback error: {e}")
        return RedirectResponse(url="/?error=auth_failed")


@app.get("/auth/me")
async def get_current_user_info(auth_token: Optional[str] = Cookie(None)):
    """Get current authenticated user info"""
    if not auth_token:
        return {"authenticated": False}
    
    payload = verify_jwt_token(auth_token)
    if not payload:
        return {"authenticated": False}
    
    user_id = int(payload.get("sub"))
    
    # Get user from DB
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, email, name, profile_image FROM users WHERE id = %s",
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {"authenticated": False}
        
        return {
            "authenticated": True,
            "user": {
                "id": row[0],
                "email": row[1],
                "name": row[2],
                "profile_image": row[3]
            }
        }
    finally:
        cursor.close()
        conn.close()


@app.get("/auth/logout")
async def logout():
    """Log out user"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("auth_token")
    return response


@app.post("/auth/register")
async def register(request: RegisterRequest):
    """Register a new user with email and password."""
    # Validate email format
    if "@" not in request.email or "." not in request.email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Validate password strength
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (request.email.lower(),))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password and create user
        password_hash = hash_password(request.password)
        name = request.name or request.email.split("@")[0]
        
        cursor.execute(
            """INSERT INTO users (email, name, password_hash) 
               VALUES (%s, %s, %s) 
               RETURNING id""",
            (request.email.lower(), name, password_hash)
        )
        user_id = cursor.fetchone()[0]
        
        # Migrate "default" closet data to this new user
        cursor.execute(
            """UPDATE user_closet_items SET user_id = %s WHERE user_id = 'default'""",
            (str(user_id),)
        )
        cursor.execute(
            """UPDATE taste_vectors SET session_id = %s WHERE session_id = 'default'""",
            (str(user_id),)
        )
        
        conn.commit()
        logger.info(f"Registered new user: {request.email} (id={user_id})")
        
        # Create JWT token and set cookie
        token = create_jwt_token(str(user_id), request.email.lower())
        
        return {"success": True, "user_id": user_id, "token": token}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        cursor.close()
        conn.close()


@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login with email and password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, email, name, password_hash, profile_image FROM users WHERE email = %s",
            (request.email.lower(),)
        )
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        user_id, email, name, password_hash, profile_image = row
        
        # Check if user has a password (could be OAuth-only user)
        if not password_hash:
            raise HTTPException(status_code=401, detail="This account uses Google Sign-In. Please sign in with Google.")
        
        # Verify password
        if not verify_password(request.password, password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user_id,))
        conn.commit()
        
        # Create JWT token
        token = create_jwt_token(str(user_id), email)
        
        logger.info(f"User logged in: {email} (id={user_id})")
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "name": name,
                "profile_image": profile_image
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        cursor.close()
        conn.close()


@app.get("/login")
async def serve_login():
    """Serve login page"""
    return FileResponse("static/login.html")


# ============== PAGE ROUTES ==============

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
# Mount static files - order matters: more specific paths first
app.mount("/static/generated", StaticFiles(directory="collages"), name="generated")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/v1/outfits:generate")
async def generate_outfits(request: Request, file: UploadFile = File(...), session_id: str = Form(None)):
    logger.info(f"Request received: POST /v1/outfits:generate (session={session_id[:8] if session_id else 'none'}...)")
    
    # Get base URL for absolute URLs
    base_url = str(request.base_url).rstrip("/")
    
    # Get taste vectors for personalization (soft scoring, no hard exclusions)
    taste_vector, dislike_vector = get_taste_vector(session_id) if session_id else (None, None)
    if taste_vector or dislike_vector:
        logger.info(f"Taste vectors found (likes={taste_vector is not None}, dislikes={dislike_vector is not None})")

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
        logger.info(f"Cache HIT for image {image_hash[:12]}... - skipping Vision/Embedding")
        base_item = cached["base_item"]
        embedding = cached["embedding"]
    else:
        logger.info(f"Cache MISS for image {image_hash[:12]}... - calling APIs")

        # Vision: Single-call image analysis (GPT-4o-mini vision → structured JSON)
        logger.info("Analyzing image...")
        try:
            base_item = analyze_image(contents)
            logger.info(f"BaseItem: {base_item}")
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Vision analysis error: {str(e)}")

        # Embedding: Generate image embedding via FashionCLIP
        logger.info("Generating FashionCLIP image embedding...")
        try:
            embedding = embed_item_image(contents)
            logger.info(f"Embedding generated (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    # 5. Generate outfits via cascade retrieval engine
    base_category = base_item.get("category", "top")
    from services.outfit_generator import run_outfit_generation

    full_item = {
        **base_item,
        "id": image_hash,
        "name": f"{base_item.get('primary_color', '')} {base_category}".strip().title(),
        "image_url": str(upload_path),
        "embedding": embedding,
    }

    raw_outfits = run_outfit_generation(
        full_item,
        source=CATALOG_SOURCE,
        taste_vector=taste_vector,
        dislike_vector=dislike_vector,
    )

    outfits = []
    for ro in raw_outfits:
        items = []
        for oi in ro.get("outfit_items", []):
            items.append({
                "slot": oi["slot"],
                "id": oi["id"],
                "name": oi["name"],
                "image_url": oi["image_url"],
                "product_url": oi.get("product_url"),
                "price": oi.get("price"),
                "primary_color": oi.get("primary_color"),
                "is_anchor": oi.get("is_anchor", False),
            })
        outfits.append({
            "direction": ro["direction"],
            "explanation": ro.get("explanation", ""),
            "items": items,
        })

    logger.info("All outfits built via cascade engine")

    # 6. Store in database (get generation_id first for collages)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO outfit_generations 
               (input_image_url, input_image_hash, input_description, parsed_tags, base_item_embedding, output_outfits, input_type) 
               VALUES (%s, %s, %s, %s, %s, %s, %s) 
               RETURNING id""",
            (str(upload_path), image_hash, f"{base_item.get('primary_color', '')} {base_item.get('category', 'item')}".strip() or None, Json(base_item), embedding, Json(outfits), "image")
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


@app.post("/v1/closet/feedback")
async def submit_closet_feedback(req: ClosetFeedbackRequest, auth_token: Optional[str] = Cookie(None)):
    """Submit like/dislike feedback for closet outfits - updates taste vector. Requires authentication."""
    user_id = require_auth(auth_token)
    logger.info(f"Closet feedback: items={req.item_ids}, liked={req.liked}, user={user_id}")
    
    if not req.item_ids:
        raise HTTPException(status_code=400, detail="item_ids required")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get embeddings for the items
        placeholders = ','.join(['%s'] * len(req.item_ids))
        cursor.execute(
            f"""SELECT id, embedding::text FROM user_closet_items 
               WHERE id IN ({placeholders}) AND user_id = %s AND embedding IS NOT NULL""",
            (*req.item_ids, user_id)
        )
        rows = cursor.fetchall()
        
        item_embeddings = []
        for row in rows:
            if row[1]:
                embedding = [float(x) for x in row[1].strip("[]").split(",")]
                item_embeddings.append(embedding)
        
        if item_embeddings:
            # Use user_id as session_id for closet users
            update_taste_vector(user_id, item_embeddings, req.liked)
            logger.info(f"Taste vector updated: {len(item_embeddings)} embeddings, liked={req.liked}")
        
        return {"status": "recorded", "items_processed": len(item_embeddings)}
    except Exception as e:
        logger.error(f"Closet feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


class SaveOutfitRequest(BaseModel):
    outfit_data: dict  # Full outfit items with IDs, names, image_urls
    collage_url: Optional[str] = None
    occasion: Optional[str] = None
    base_item_id: Optional[int] = None


@app.post("/v1/closet/outfits/save")
async def save_outfit(req: SaveOutfitRequest, auth_token: Optional[str] = Cookie(None)):
    """Save an outfit for later. Uploads collage to Cloudinary for permanent storage."""
    user_id = require_auth(auth_token)
    
    permanent_collage_url = req.collage_url
    
    # If collage_url is a local/generated path, upload to Cloudinary
    if req.collage_url and ('/static/generated/' in req.collage_url or req.collage_url.startswith('collages/')):
        try:
            # Extract local path from URL (e.g., daily_0/outfit_1.jpg)
            local_path = req.collage_url
            if '/static/generated/' in local_path:
                local_path = local_path.split('/static/generated/')[-1]
            # Remove any query params (like ?t=timestamp)
            if '?' in local_path:
                local_path = local_path.split('?')[0]
            
            # Try multiple possible paths (static mount vs actual location)
            possible_paths = [
                os.path.join("static", "generated", local_path),
                os.path.join("collages", local_path),
            ]
            
            full_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    full_path = path
                    break
            
            if full_path:
                logger.info(f"Uploading collage to Cloudinary: {full_path}")
                upload_result = cloudinary.uploader.upload(
                    full_path,
                    folder="outfits",
                    resource_type="image"
                )
                permanent_collage_url = upload_result["secure_url"]
                logger.info(f"Collage uploaded to Cloudinary: {permanent_collage_url}")
            else:
                logger.warning(f"Local collage not found at any path: {possible_paths}")
        except Exception as e:
            logger.error(f"Failed to upload collage to Cloudinary: {e}")
            # Keep original URL as fallback
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        import json
        cursor.execute(
            """INSERT INTO saved_outfits (user_id, outfit_data, collage_url, occasion, base_item_id, status)
               VALUES (%s, %s, %s, %s, %s, 'saved')
               RETURNING id""",
            (user_id, json.dumps(req.outfit_data), permanent_collage_url, req.occasion, req.base_item_id)
        )
        outfit_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Outfit saved: id={outfit_id}, user={user_id}, collage={permanent_collage_url}")
        return {"status": "saved", "outfit_id": outfit_id, "collage_url": permanent_collage_url}
    except Exception as e:
        conn.rollback()
        logger.error(f"Save outfit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/closet/outfits/{outfit_id}/worn")
async def mark_outfit_worn(outfit_id: int, auth_token: Optional[str] = Cookie(None)):
    """Mark a saved outfit as worn. Moves the base top to back of FIFO queue."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get the outfit and verify ownership
        cursor.execute(
            """SELECT outfit_data, occasion, base_item_id FROM saved_outfits 
               WHERE id = %s AND user_id = %s""",
            (outfit_id, user_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Outfit not found")
        
        outfit_data, occasion, base_item_id = row
        
        # Mark as worn
        cursor.execute(
            """UPDATE saved_outfits SET status = 'worn', worn_at = NOW()
               WHERE id = %s""",
            (outfit_id,)
        )
        
        # Update FIFO queue - move this top to back of queue for this occasion
        if base_item_id and occasion:
            cursor.execute(
                """INSERT INTO top_suggestions (user_id, item_id, occasion, last_suggested_at, suggestion_count)
                   VALUES (%s, %s, %s, NOW(), 1)
                   ON CONFLICT (user_id, item_id, occasion) 
                   DO UPDATE SET last_suggested_at = NOW(), suggestion_count = top_suggestions.suggestion_count + 1""",
                (user_id, base_item_id, occasion)
            )
        
        conn.commit()
        logger.info(f"Outfit marked as worn: id={outfit_id}, user={user_id}")
        return {"status": "worn", "outfit_id": outfit_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Mark worn error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


class MarkItemWornRequest(BaseModel):
    occasion: str = "casual"  # Default occasion if not specified


@app.post("/v1/closet/items/{item_id}/worn")
async def mark_item_worn(item_id: int, req: MarkItemWornRequest = None, auth_token: Optional[str] = Cookie(None)):
    """
    Mark a single item as worn - adds it to the FIFO queue so it's deprioritized.
    Also creates a record in saved_outfits with status='worn' for history.
    Useful for tracking tops worn outside of app-generated outfits.
    """
    user_id = require_auth(auth_token)
    occasion = req.occasion if req else "casual"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get item details
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color 
               FROM user_closet_items WHERE id = %s AND user_id = %s""",
            (item_id, user_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_id, name, category, image_url, primary_color = row
        if category != "top":
            raise HTTPException(status_code=400, detail="Only tops can be marked as worn for rotation")
        
        # Add to FIFO queue
        cursor.execute(
            """INSERT INTO top_suggestions (user_id, item_id, occasion, last_suggested_at, suggestion_count)
               VALUES (%s, %s, %s, NOW(), 1)
               ON CONFLICT (user_id, item_id, occasion) 
               DO UPDATE SET last_suggested_at = NOW(), suggestion_count = top_suggestions.suggestion_count + 1""",
            (user_id, item_id, occasion)
        )
        
        # Also create a worn outfit record for history
        import json
        outfit_data = {
            "items": [{
                "id": item_id,
                "name": name,
                "category": category,
                "slot": "top",
                "image_url": image_url,
                "primary_color": primary_color
            }],
            "base_item": {
                "id": item_id,
                "name": name,
                "category": category,
                "image_url": image_url
            },
            "explanation": f"Single item worn ({occasion})",
            "direction": "Quick wear"
        }
        
        cursor.execute(
            """INSERT INTO saved_outfits (user_id, outfit_data, occasion, base_item_id, status, worn_at)
               VALUES (%s, %s, %s, %s, 'worn', NOW())
               RETURNING id""",
            (user_id, json.dumps(outfit_data), occasion, item_id)
        )
        outfit_id = cursor.fetchone()[0]
        
        conn.commit()
        logger.info(f"Item {item_id} marked as worn for occasion {occasion}, user={user_id}, outfit_id={outfit_id}")
        return {"status": "worn", "item_id": item_id, "occasion": occasion, "outfit_id": outfit_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Mark item worn error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get("/v1/closet/outfits/saved")
async def list_saved_outfits(auth_token: Optional[str] = Cookie(None)):
    """List all saved outfits (not yet worn)."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, outfit_data, collage_url, occasion, saved_at
               FROM saved_outfits 
               WHERE user_id = %s AND status = 'saved'
               ORDER BY saved_at DESC""",
            (user_id,)
        )
        rows = cursor.fetchall()
        
        outfits = []
        for row in rows:
            outfit_id, outfit_data, collage_url, occasion, saved_at = row
            
            # Fallback for old saved outfits with local URLs (won't work after restart)
            if collage_url and '/static/generated/' in collage_url and 'cloudinary' not in collage_url:
                items = outfit_data.get('items', []) if outfit_data else []
                if items and items[0].get('image_url'):
                    collage_url = items[0]['image_url']
                else:
                    collage_url = None
            
            outfits.append({
                "id": outfit_id,
                "outfit_data": outfit_data,
                "collage_url": collage_url,
                "occasion": occasion,
                "saved_at": saved_at.isoformat() if saved_at else None
            })
        
        return {"outfits": outfits, "count": len(outfits)}
    finally:
        cursor.close()
        conn.close()


@app.get("/v1/closet/outfits/worn")
async def list_worn_outfits(auth_token: Optional[str] = Cookie(None)):
    """List outfit history (previously worn)."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, outfit_data, collage_url, occasion, worn_at
               FROM saved_outfits 
               WHERE user_id = %s AND status = 'worn'
               ORDER BY worn_at DESC
               LIMIT 50""",
            (user_id,)
        )
        rows = cursor.fetchall()
        
        outfits = []
        for row in rows:
            outfit_id, outfit_data, collage_url, occasion, worn_at = row
            
            # Fallback for old worn outfits with local URLs
            if collage_url and '/static/generated/' in collage_url and 'cloudinary' not in collage_url:
                items = outfit_data.get('items', []) if outfit_data else []
                if items and items[0].get('image_url'):
                    collage_url = items[0]['image_url']
                else:
                    collage_url = None
            
            outfits.append({
                "id": outfit_id,
                "outfit_data": outfit_data,
                "collage_url": collage_url,
                "occasion": occasion,
                "worn_at": worn_at.isoformat() if worn_at else None
            })
        
        return {"outfits": outfits, "count": len(outfits)}
    finally:
        cursor.close()
        conn.close()


@app.delete("/v1/closet/outfits/{outfit_id}")
async def delete_saved_outfit(outfit_id: int, auth_token: Optional[str] = Cookie(None)):
    """Delete a saved outfit."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """DELETE FROM saved_outfits WHERE id = %s AND user_id = %s RETURNING id""",
            (outfit_id, user_id)
        )
        deleted = cursor.fetchone()
        conn.commit()
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Outfit not found")
        
        return {"status": "deleted", "outfit_id": outfit_id}
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Delete outfit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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
async def list_closet_items(auth_token: Optional[str] = Cookie(None)):
    """List all items in user's closet. Requires authentication."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color, secondary_colors,
                      style_tags, season_tags, occasion_tags, material, fit, created_at,
                      status, batch_id
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
                "created_at": row[11].isoformat() if row[11] else None,
                "status": row[12] if row[12] else "ready",
                "batch_id": row[13],
            })
        
        return {"items": items, "count": len(items)}
    finally:
        cursor.close()
        conn.close()


MAX_BATCH_SIZE = 50
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _cloudinary_upload_safe(image_bytes: bytes) -> dict | None:
    """Upload one image to Cloudinary after processing. Returns None on failure."""
    try:
        processed = process_clothing_image(image_bytes)
        result = cloudinary.uploader.upload(
            processed, folder="closet", resource_type="image"
        )
        return {"image_url": result["secure_url"]}
    except Exception as e:
        logger.error(f"Cloudinary upload failed (skipping): {e}")
        return None


@app.post("/v1/closet/items:batch")
async def batch_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload N images in one request. Returns immediately with batch_id and
    Cloudinary URLs for instant thumbnails. Processing happens in background.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Maximum {MAX_BATCH_SIZE} files per batch")

    raw_images = []
    for f in files:
        data = await f.read()
        if not data or len(data) > MAX_FILE_SIZE:
            continue
        raw_images.append(data)

    if not raw_images:
        raise HTTPException(400, "No valid images provided")

    batch_id = str(uuid.uuid4())

    # Parallel Cloudinary upload (I/O bound, ~2-3s for 20 images)
    uploaded = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_cloudinary_upload_safe, img) for img in raw_images]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                uploaded.append(result)

    if not uploaded:
        raise HTTPException(500, "All image uploads failed")

    # Bulk INSERT as pending
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        item_rows = []
        for u in uploaded:
            cur.execute(
                """INSERT INTO user_closet_items
                   (user_id, image_url, status, batch_id)
                   VALUES (%s, %s, 'pending', %s)
                   RETURNING id""",
                (user_id, u["image_url"], batch_id),
            )
            item_id = cur.fetchone()[0]
            item_rows.append({"id": item_id, "image_url": u["image_url"]})
        conn.commit()
    finally:
        cur.close()
        conn.close()

    # Kick off background processing
    background_tasks.add_task(
        process_batch_items, batch_id,
        [(r["id"], r["image_url"]) for r in item_rows]
    )

    return {
        "batch_id": batch_id,
        "count": len(item_rows),
        "items": [
            {"id": r["id"], "image_url": r["image_url"], "status": "pending"}
            for r in item_rows
        ],
    }


@app.get("/v1/closet/items:batch-status/{batch_id}")
async def batch_status(batch_id: str, request: Request):
    """Poll processing progress for a batch."""
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT id, image_url, name, category, status, processing_error
               FROM user_closet_items
               WHERE batch_id = %s AND user_id = %s
               ORDER BY id""",
            (batch_id, user_id),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    items = []
    counts = {"pending": 0, "processing": 0, "ready": 0, "error": 0}
    for row in rows:
        status = row[4]
        counts[status] = counts.get(status, 0) + 1
        items.append({
            "id": row[0],
            "image_url": row[1],
            "name": row[2],
            "category": row[3],
            "status": status,
            "error": row[5] if status == "error" else None,
        })

    total = len(items)
    return {
        "batch_id": batch_id,
        "total": total,
        "counts": counts,
        "done": (counts["ready"] + counts["error"]) == total and total > 0,
        "items": items,
    }


@app.post("/v1/closet/items/{item_id}/retry")
async def retry_item(
    item_id: int,
    request: Request,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Retry a failed item."""
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """UPDATE user_closet_items
               SET status = 'pending', processing_error = NULL, retry_count = 0
               WHERE id = %s AND user_id = %s AND status = 'error'
               RETURNING image_url""",
            (item_id, user_id),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Item not found or not in error state")
        conn.commit()
        image_url = row[0]
    finally:
        cur.close()
        conn.close()

    background_tasks.add_task(process_batch_items, "retry", [(item_id, image_url)])
    return {"status": "retrying", "id": item_id}


@app.post("/v1/closet/items")
async def add_closet_item(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload a single item to user's closet. Now routes through the batch pipeline
    for consistent processing (blended embedding, proper error handling).
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Adding closet item for user: {user_id}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image uploaded")

    batch_id = str(uuid.uuid4())

    # Process and upload to Cloudinary (same image used for storage + vision)
    try:
        processed_bytes = process_clothing_image(contents)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        processed_bytes = contents

    try:
        upload_result = cloudinary.uploader.upload(
            processed_bytes, folder="closet", resource_type="image"
        )
        image_url = upload_result["secure_url"]
    except Exception as e:
        logger.error(f"Cloudinary upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

    # Insert as pending
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO user_closet_items
               (user_id, image_url, status, batch_id)
               VALUES (%s, %s, 'pending', %s)
               RETURNING id""",
            (user_id, image_url, batch_id),
        )
        item_id = cur.fetchone()[0]
        conn.commit()
    finally:
        cur.close()
        conn.close()

    # Process in background (vision + blended embedding)
    background_tasks.add_task(process_batch_items, batch_id, [(item_id, image_url)])

    return {
        "id": item_id,
        "name": None,
        "category": None,
        "image_url": image_url,
        "status": "pending",
        "batch_id": batch_id,
    }


MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB


@app.post("/v1/closet/items:detect-video")
async def detect_from_video(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload a video. Server extracts frames, detects clothing items per frame,
    deduplicates by embedding similarity, and processes unique items.
    Returns batch_id immediately; extraction + processing happen in background.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)

    video_bytes = await file.read()
    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(400, "Video too large (max 100MB)")

    batch_id = str(uuid.uuid4())

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.close()

    background_tasks.add_task(
        _extract_and_process_video, tmp.name, user_id, batch_id
    )

    return {"batch_id": batch_id, "status": "extracting"}


def _extract_and_process_video(video_path: str, user_id: str, batch_id: str):
    """
    Full pipeline: extract frames -> YOLO detect -> ByteTrack -> best crop per object -> upload -> process.
    Handles multiple items in a single video via persistent object tracking.
    """
    import subprocess
    import tempfile
    import os
    from pathlib import Path
    from PIL import Image
    import io

    from services.object_tracker import process_video_frames

    VIDEO_FPS = 3  # Extract 3 frames/sec for tracking continuity
    MIN_TRACK_FRAMES = 5

    try:
        # 1. Extract frames with ffmpeg at higher FPS for tracking
        frames_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(frames_dir, "frame_%06d.jpg")

        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={VIDEO_FPS},scale=1280:-1",
                "-q:v", "2", "-y",
                output_pattern,
            ],
            capture_output=True, check=True, timeout=120,
        )

        frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
        logger.info(f"Video: extracted {len(frame_paths)} frames at {VIDEO_FPS} fps")

        if not frame_paths:
            logger.warning("No frames extracted from video")
            return

        # 2. Load frames as (bytes, width, height) tuples
        frames = []
        for fp in frame_paths:
            try:
                img_bytes = fp.read_bytes()
                img = Image.open(io.BytesIO(img_bytes))
                frames.append((img_bytes, img.width, img.height))
            except Exception as e:
                logger.warning(f"Frame {fp.name} failed to load: {e}")

        if not frames:
            return

        # 3. Run YOLO + ByteTrack pipeline
        crop_results = process_video_frames(
            frames, fps=VIDEO_FPS, min_frames=MIN_TRACK_FRAMES
        )

        logger.info(f"Video: tracked {len(crop_results)} unique items")

        if not crop_results:
            logger.info("Video: no items detected with sufficient confidence")
            return

        # 4. Upload best crops to Cloudinary (maintain order for embedding mapping)
        upload_futures = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, crop in enumerate(crop_results):
                future = pool.submit(_cloudinary_upload_safe, crop.crop_bytes)
                upload_futures[future] = i

        uploaded = []  # (index, cloudinary_result)
        for future in as_completed(upload_futures):
            result = future.result()
            if result:
                uploaded.append((upload_futures[future], result))

        uploaded.sort(key=lambda x: x[0])

        # 5. Insert into DB as pending, with pre-computed temporal embeddings
        conn = get_db_connection()
        cur = conn.cursor()
        items_to_process = []
        items_with_embeddings = {}  # item_id -> temporal_embedding
        try:
            for idx, u in uploaded:
                crop = crop_results[idx]
                cur.execute(
                    """INSERT INTO user_closet_items
                       (user_id, image_url, status, batch_id)
                       VALUES (%s, %s, 'pending', %s)
                       RETURNING id""",
                    (user_id, u["image_url"], batch_id),
                )
                item_id = cur.fetchone()[0]
                items_to_process.append((item_id, u["image_url"]))

                if crop.temporal_embedding:
                    items_with_embeddings[item_id] = crop.temporal_embedding
            conn.commit()
        finally:
            cur.close()
            conn.close()

        # 6. Process through ingestion worker with pre-computed embeddings
        if items_to_process:
            process_batch_items(batch_id, items_to_process, precomputed_embeddings=items_with_embeddings)

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out processing video")
    except Exception as e:
        logger.error(f"Video extraction failed: {e}", exc_info=True)
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


@app.delete("/v1/closet/items/{item_id}")
async def delete_closet_item(item_id: int, auth_token: Optional[str] = Cookie(None)):
    """Delete an item from user's closet. Requires authentication."""
    user_id = require_auth(auth_token)
    
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
async def rotate_closet_item(item_id: int, req: RotateRequest, auth_token: Optional[str] = Cookie(None)):
    user_id = require_auth(auth_token)
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


@app.api_route("/v1/closet/retag-all", methods=["GET", "POST"])
async def retag_all_closet_items(user_id: str = "default"):
    """Re-tag ALL items in closet with updated parser tags."""
    import httpx
    
    conn = get_db_connection()
    cursor = conn.cursor()
    results = {"success": 0, "failed": 0, "items": []}
    
    try:
        # Get all items
        cursor.execute(
            "SELECT id, image_url, name FROM user_closet_items WHERE user_id = %s",
            (user_id,)
        )
        rows = cursor.fetchall()
        logger.info(f"Re-tagging {len(rows)} items...")
        
        for row in rows:
            item_id, image_url, old_name = row
            try:
                # Download image
                response = httpx.get(image_url, timeout=15)
                if response.status_code != 200:
                    results["failed"] += 1
                    continue
                
                # Single-call image analysis
                parsed = analyze_image(response.content)
                
                # Generate new name
                name = f"{parsed.get('primary_color', '')} {parsed.get('category', 'item')}".strip().title()
                
                # Update database
                cursor.execute(
                    """UPDATE user_closet_items 
                       SET name = %s, category = %s, primary_color = %s, secondary_colors = %s,
                           style_tags = %s, season_tags = %s, occasion_tags = %s, material = %s, fit = %s
                       WHERE id = %s AND user_id = %s""",
                    (
                        name,
                        parsed.get("category", "top"),
                        parsed.get("primary_color"),
                        parsed.get("secondary_colors"),
                        parsed.get("style_tags"),
                        parsed.get("season_tags"),
                        parsed.get("occasion_tags"),
                        parsed.get("material"),
                        parsed.get("fit"),
                        item_id, user_id
                    )
                )
                results["success"] += 1
                results["items"].append({
                    "id": item_id,
                    "name": name,
                    "style_tags": parsed.get("style_tags"),
                    "occasion_tags": parsed.get("occasion_tags")
                })
                logger.info(f"  [{item_id}] {old_name} -> {name} | {parsed.get('style_tags')} | {parsed.get('occasion_tags')}")
                
            except Exception as e:
                logger.error(f"  [{item_id}] Error: {e}")
                results["failed"] += 1
        
        conn.commit()
        logger.info(f"Re-tag complete: {results['success']} success, {results['failed']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Re-tag all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/closet/items/{item_id}/retag")
async def retag_single_item(item_id: int, auth_token: Optional[str] = Cookie(None)):
    """Re-tag a single item with updated vision + parser. Requires authentication."""
    import httpx
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get item
        cursor.execute(
            "SELECT id, image_url, name FROM user_closet_items WHERE id = %s AND user_id = %s",
            (item_id, user_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_id, image_url, old_name = row
        
        # Download image
        response = httpx.get(image_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Could not download image")
        
        # Single-call image analysis
        parsed = analyze_image(response.content)
        
        # Generate new name
        name = f"{parsed.get('primary_color', '')} {parsed.get('category', 'item')}".strip().title()
        
        # Blended embedding: image + text metadata
        from services.embedding import embed_item_blended
        embedding = embed_item_blended(response.content, parsed)
        
        # Update database with tags AND embedding
        cursor.execute(
            """UPDATE user_closet_items 
               SET name = %s, category = %s, primary_color = %s, secondary_colors = %s,
                   style_tags = %s, season_tags = %s, occasion_tags = %s, material = %s, fit = %s,
                   embedding = %s
               WHERE id = %s AND user_id = %s""",
            (
                name,
                parsed.get("category", "top"),
                parsed.get("primary_color"),
                parsed.get("secondary_colors"),
                parsed.get("style_tags"),
                parsed.get("season_tags"),
                parsed.get("occasion_tags"),
                parsed.get("material"),
                parsed.get("fit"),
                embedding,
                item_id, user_id
            )
        )
        conn.commit()
        
        logger.info(f"Retagged [{item_id}] {old_name} -> {name} | {parsed.get('style_tags')} | {parsed.get('occasion_tags')}")
        
        return {
            "id": item_id,
            "old_name": old_name,
            "new_name": name,
            "category": parsed.get("category"),
            "primary_color": parsed.get("primary_color"),
            "style_tags": parsed.get("style_tags"),
            "occasion_tags": parsed.get("occasion_tags"),
            "material": parsed.get("material")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retag error for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/admin/recompute-embeddings")
async def recompute_all_embeddings(auth_token: Optional[str] = Cookie(None)):
    """
    Nuclear reset: re-download every item image from Cloudinary,
    compute a fresh FashionCLIP image embedding, blend 70/30 with
    text metadata, and store the result. Fixes any corruption from
    previous double-blends.
    """
    from concurrent.futures import ThreadPoolExecutor
    import httpx as _httpx
    from services.embedding import embed_item_blended

    user_id = require_auth(auth_token)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, primary_color, style_tags,
                      occasion_tags, season_tags, material, fit, image_url
               FROM user_closet_items
               WHERE user_id = %s AND image_url IS NOT NULL""",
            (user_id,)
        )
        rows = cursor.fetchall()

        def download(url: str) -> bytes | None:
            try:
                r = _httpx.get(url, timeout=15.0)
                return r.content if r.status_code == 200 else None
            except Exception:
                return None

        image_urls = [row[9] for row in rows]
        with ThreadPoolExecutor(max_workers=8) as pool:
            image_data = list(pool.map(download, image_urls))

        updated, skipped = 0, 0
        for row, img_bytes in zip(rows, image_data):
            if not img_bytes:
                skipped += 1
                continue

            item_id = row[0]
            base_item = {
                "name": row[1],
                "category": row[2],
                "primary_color": row[3],
                "style_tags": row[4],
                "occasion_tags": row[5],
                "season_tags": row[6],
                "material": row[7],
                "fit": row[8],
            }

            blended = embed_item_blended(img_bytes, base_item)
            cursor.execute(
                "UPDATE user_closet_items SET embedding = %s WHERE id = %s AND user_id = %s",
                (blended, item_id, user_id)
            )
            updated += 1

        conn.commit()
        logger.info(f"Recomputed embeddings for {updated} items, skipped {skipped} (user {user_id})")
        return {"updated": updated, "skipped": skipped, "total": len(rows)}
    except Exception as e:
        conn.rollback()
        logger.error(f"Recompute embeddings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.post("/v1/closet/items/{item_id}/reupload")
async def reupload_closet_item(request: Request, item_id: int, file: UploadFile = File(...)):
    """Re-upload an item's image (after client-side background removal). Requires authentication."""
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    
    try:
        # Read image bytes directly - no processing to preserve transparency
        image_bytes = await file.read()
        logger.info(f"Re-uploading item {item_id}: {len(image_bytes)} bytes")
        
        # Upload to Cloudinary directly (keep PNG transparency)
        result = cloudinary.uploader.upload(
            image_bytes,
            folder="closet",
            resource_type="image",
            format="png"  # Keep as PNG with transparency
        )
        new_url = result["secure_url"]
        logger.info(f"Re-uploaded item {item_id}: {new_url}")
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE user_closet_items SET image_url = %s WHERE id = %s AND user_id = %s RETURNING id",
                (new_url, item_id, user_id)
            )
            updated = cursor.fetchone()
            if not updated:
                raise HTTPException(status_code=404, detail="Item not found")
            
            conn.commit()
            return {"id": item_id, "image_url": new_url}
        finally:
            cursor.close()
            conn.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Re-upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/closet/items/{item_id}")
async def get_closet_item(item_id: int, auth_token: Optional[str] = Cookie(None)):
    """Get a single closet item with its embedding. Requires authentication."""
    user_id = require_auth(auth_token)
    
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
    lat: float = Form(None),
    lon: float = Form(None),
    mood_text: str = Form(None)
):
    """
    Generate outfits using ONLY items from user's closet. Requires authentication.
    Either pick an existing closet item (item_id) or upload a new image.
    Optionally pass lat/lon to factor in weather.
    Optionally pass mood_text to specify occasion (e.g., "work", "date night").
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Closet outfit generation: item_id={item_id}, has_file={file is not None}, weather={lat},{lon}, mood={mood_text}")
    
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
        
        # Single-call image analysis
        base_item = analyze_image(contents)
        embedding = embed_item_image(contents)
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
    
    # Log mood if provided - we'll use it directly for semantic matching
    # No need to map to predefined occasions anymore!
    if mood_text:
        logger.info(f"Single-item generation with mood: {mood_text} (using direct embedding)")
    
    base_category = base_item.get("category", "top")

    anchor_id = item_id if item_id else new_item_id
    anchor_name = description if item_id else f"{base_item.get('primary_color', '')} {base_category}".strip().title()
    full_item = {
        **base_item,
        "id": anchor_id,
        "name": anchor_name,
        "image_url": input_image_url,
        "embedding": embedding,
    }

    # Use cascade retrieval (same pipeline as Shopify/eval)
    from services.outfit_generator import run_outfit_generation
    raw_outfits = await asyncio.to_thread(
        run_outfit_generation,
        full_item,
        use_closet=True,
        user_id=user_id,
        mood_text=mood_text,
    )

    # Convert run_outfit_generation output to the format this endpoint expects
    outfits = []
    for ro in raw_outfits:
        items = []
        for oi in ro.get("outfit_items", []):
            if oi.get("is_anchor"):
                continue
            items.append({
                "slot": oi["slot"],
                "id": oi["id"],
                "name": oi["name"],
                "image_url": oi["image_url"],
            })
        outfits.append({
            "direction": ro["direction"],
            "explanation": ro.get("explanation", ""),
            "items": items,
        })
    
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
    refresh: bool = False,  # If true, regenerate even if cached
    from_style: bool = False,  # If true and default mood: skip cache, generate new (once per day when user goes to Style)
    tz_offset: float = None,  # Timezone offset from UTC in hours (e.g., -8 for PST)
    occasion: str = None,  # Manual occasion override (work, casual, going-out, etc.)
    mood: str = None  # Free-form mood description (e.g., "cozy day", "fancy dinner")
):
    """
    Generate 3 weather-appropriate outfits automatically. Requires authentication.
    Picks different base items from closet for variety.
    When from_style=true and no mood/occasion (default mood), always generate fresh for the day.
    Otherwise results are cached per user/day/occasion - use cache when available.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Daily outfits: lat={lat}, lon={lon}, tz={tz_offset}, mood={mood}, refresh={refresh}, from_style={from_style}")
    base_url = str(request.base_url).rstrip("/")
    
    # Compute user's local date (not server UTC) for cache keying
    from datetime import datetime, timedelta, timezone
    user_tz = timezone(timedelta(hours=tz_offset)) if tz_offset is not None else timezone.utc
    user_today = datetime.now(user_tz).date()
    
    # Determine occasion first (needed for cache lookup)
    # Mood requests: no GPT call — FashionCLIP embedding handles item selection directly
    # Default/dropdown: use predefined occasion configs or auto-detect from time
    if mood:
        mood_slug = mood.strip().lower().replace(" ", "-")
        occasion_info = {
            "occasion": mood_slug,
            "prefer_occasions": [],
            "avoid_occasions": [],
            "note": mood.strip(),
            "needs_layer": None,
        }
        logger.info(f"Mood '{mood}' -> direct embedding (occasion slug: {mood_slug})")
    elif occasion:
        # Manual occasion selected (fallback for dropdown if used)
        OCCASION_CONFIGS = {
            "work": {
                "occasion": "work",
                "prefer_occasions": ["work", "office", "business-casual", "professional", "elegant", "classic"],
                "avoid_occasions": [],
                "note": "Work / Office"
            },
            "casual": {
                "occasion": "casual",
                "prefer_occasions": ["casual", "everyday", "relaxed", "brunch"],
                "avoid_occasions": ["formal", "black-tie"],
                "note": "Casual / Everyday 😎"
            },
            "going-out": {
                "occasion": "going-out",
                "prefer_occasions": ["going-out", "dinner", "date", "night-out", "party", "elegant", "chic"],
                "avoid_occasions": ["work", "office", "gym", "workout", "sporty", "athletic", "activewear"],
                "note": "Going Out"
            },
        }
        occasion_info = OCCASION_CONFIGS.get(occasion, OCCASION_CONFIGS["casual"])
        logger.info(f"Manual occasion: {occasion_info['occasion']} - {occasion_info['note']}")
    else:
        # Auto-detect from time
        occasion_info = get_occasion_from_time(tz_offset)
        logger.info(f"Auto occasion: {occasion_info['occasion']} - {occasion_info['note']}")
    
    occasion_name = occasion_info["occasion"]
    is_default_mood = not mood and not occasion
    has_manual_mood = bool(mood and mood.strip())

    # Skip cache when: explicit refresh, manual mood (never use default cache for mood-specific requests), or Style + default
    skip_cache = refresh or has_manual_mood or (from_style and is_default_mood)
    if not skip_cache:
        conn_cache = get_db_connection()
        cursor_cache = conn_cache.cursor()
        try:
            cursor_cache.execute(
                """SELECT outfits_json, weather_json, created_at FROM daily_outfit_cache 
                   WHERE user_id = %s AND cache_date = %s AND occasion = %s""",
                (user_id, user_today, occasion_name)
            )
            cached = cursor_cache.fetchone()
            if cached:
                logger.info(f"Cache hit for user {user_id}, occasion {occasion_name}")
                cached_outfits = cached[0]
                cached_weather = cached[1]
                cache_created = cached[2]

                _REQUIRED_SLOTS = {"bottom", "shoes"}
                for outfit in (cached_outfits or []):
                    filled = {it.get("slot") for it in outfit.get("items", []) if it}
                    if not _REQUIRED_SLOTS.issubset(filled):
                        logger.info("Cache stale: outfit missing slots %s, regenerating",
                                    _REQUIRED_SLOTS - filled)
                        cached = None
                        break

                _SCORING_VERSION_TS = "2026-05-15 07:00:00"
                if cached and cache_created:
                    from datetime import datetime as _dt
                    cutoff = _dt.fromisoformat(_SCORING_VERSION_TS)
                    if cache_created < cutoff:
                        logger.info("Cache stale: created %s before scoring version %s",
                                    cache_created, _SCORING_VERSION_TS)
                        cached = None

            if cached:
                cached_outfits = cached[0]
                cached_weather = cached[1]

                # Verify collage files exist - if not, regenerate them
                import os
                collages_valid = True
                for idx, outfit in enumerate(cached_outfits):
                    collage_url = outfit.get("collage_url", "")
                    # Extract file path from URL (handle both relative and absolute)
                    if collage_url and not collage_url.startswith("http"):
                        # Convert URL path to actual file path
                        # /static/generated/X -> collages/X
                        file_path = collage_url.lstrip("/")
                        if file_path.startswith("static/generated/"):
                            file_path = file_path.replace("static/generated/", "collages/")
                        if not os.path.exists(file_path):
                            logger.info(f"Collage missing: {file_path}, regenerating...")
                            # Regenerate this collage - use user_id and occasion in path
                            try:
                                items_for_collage = outfit.get("items", [])
                                base_item = outfit.get("base_item", {})
                                collage_path = generate_outfit_collage(
                                    generation_id=f"u{user_id}_{occasion_name}_{idx}",
                                    direction=f"outfit_{idx + 1}",
                                    items=items_for_collage,
                                    base_item={"image_url": base_item.get("image_url"), "category": base_item.get("category", "top")},
                                    force=True
                                )
                                outfit["collage_url"] = collage_path
                            except Exception as e:
                                logger.error(f"Failed to regenerate collage: {e}")
                                collages_valid = False
                    
                    # Update collage URLs to be absolute
                    if outfit.get("collage_url") and not outfit["collage_url"].startswith("http"):
                        outfit["collage_url"] = make_absolute_url(base_url, outfit["collage_url"])
                
                if collages_valid:
                    response = {
                        "outfits": cached_outfits,
                        "source": "closet_daily",
                        "occasion": occasion_name,
                        "occasion_note": occasion_info["note"],
                        "cached": True
                    }
                    if cached_weather:
                        response["weather"] = cached_weather
                    return response
                else:
                    logger.info("Collages invalid, regenerating all outfits")
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        finally:
            cursor_cache.close()
            conn_cache.close()
    
    logger.info(f"Generating fresh outfits for user {user_id}, occasion {occasion_name}")
    
    # Get taste vectors for personalization (user_id acts as session_id for closet)
    taste_vector, dislike_vector = get_taste_vector(user_id)
    if taste_vector or dislike_vector:
        logger.info(f"Taste vectors found for closet user {user_id}")
    
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
    
    # Filter items by weather season and occasion
    preferred_seasons = weather_adjustments.get("preferred_seasons", []) if weather_adjustments else []
    avoid_seasons = weather_adjustments.get("avoid_seasons", []) if weather_adjustments else []
    prefer_occasions = occasion_info.get("prefer_occasions", [])
    avoid_occasions = occasion_info.get("avoid_occasions", [])
    
    def item_score(item):
        """Score item by season, occasion, style, and material appropriateness."""
        from services.retrieval import compute_occasion_score
        from services.outfit import infer_outfit_occasion, infer_formality_continuous
        score = 0
        
        occasion_tags = item.get("occasion_tags") or []
        style_tags = item.get("style_tags") or []
        all_item_tags = set(occasion_tags + style_tags)
        
        if has_manual_mood:
            if item.get("embedding"):
                semantic_score = compute_occasion_score(
                    item["embedding"], mood_text=mood, item_tags=all_item_tags
                )
                score += semantic_score * 80

            from services.retrieval import compute_tag_mood_score
            tag_score = compute_tag_mood_score(style_tags, mood)
            score += tag_score * 20

            item_occ = infer_outfit_occasion(item)
            from services.retrieval import classify_mood_to_group
            target_group = classify_mood_to_group(mood) or "casual"
            if item_occ == target_group:
                score += 25
            elif (target_group == "going-out" and item_occ == "casual") or \
                 (target_group == "work" and item_occ == "active"):
                score -= 50

            formality_pt, _, _ = infer_formality_continuous(item)
            if target_group == "going-out" and formality_pt < 2.8:
                score -= 40
            elif target_group == "work" and formality_pt < 3.0:
                score -= 30
        else:
            # Auto occasion: use season, material, tag matching, and embeddings
            season_tags = item.get("season_tags") or []
            for s in preferred_seasons:
                if s in season_tags or "all-season" in season_tags:
                    score += 1
            for s in avoid_seasons:
                if s in season_tags:
                    score -= 2
            
            direct_matches = sum(1 for o in prefer_occasions if o in all_item_tags)
            score += direct_matches * 5
            for o in avoid_occasions:
                if o in all_item_tags:
                    score -= 5
            
            if item.get("embedding") and occasion_name:
                semantic_score = compute_occasion_score(
                    item["embedding"], occasion=occasion_name, item_tags=all_item_tags
                )
                score += semantic_score * 15

            item_occ = infer_outfit_occasion(item)
            auto_occ = occasion_name or "casual"
            _occ_to_group = {
                "going-out": "going-out", "night-out": "going-out", "date-night": "going-out",
                "work": "work", "office": "work",
                "casual": "casual", "everyday": "casual",
                "workout": "active",
            }
            target_group = _occ_to_group.get(auto_occ, "casual")
            if item_occ == target_group:
                score += 15
            elif (target_group == "going-out" and item_occ == "casual") or \
                 (target_group == "work" and item_occ == "active"):
                score -= 30

            formality_pt, _, _ = infer_formality_continuous(item)
            if target_group == "going-out" and formality_pt < 2.8:
                score -= 25
            elif target_group == "work" and formality_pt < 3.0:
                score -= 15

            if weather_adjustments:
                material = item.get("material") or ""
                material_score = get_material_weather_score(material, weather_adjustments)
                if item.get("category") == "layer":
                    material_score *= 2
                score += material_score
        
        return score
    
    # Pick top 3 TOPS as base items with FIFO rotation
    # Tops suggested recently are penalized to ensure variety across days
    # NOTE: FIFO rotation only applies to auto-detected occasions (not manual mood)
    import random
    from datetime import datetime, timedelta
    
    # When refreshing, read previously shown top IDs from cache so we can avoid them
    refresh_exclude_ids = set()
    if refresh:
        try:
            conn_prev = get_db_connection()
            cur_prev = conn_prev.cursor()
            cur_prev.execute(
                """SELECT outfits_json FROM daily_outfit_cache
                   WHERE user_id = %s AND cache_date = %s AND occasion = %s""",
                (user_id, user_today, occasion_name)
            )
            row_prev = cur_prev.fetchone()
            if row_prev and row_prev[0]:
                for outfit in row_prev[0]:
                    base_id = (outfit.get("base_item") or {}).get("id")
                    if base_id:
                        refresh_exclude_ids.add(base_id)
            cur_prev.close()
            conn_prev.close()
            if refresh_exclude_ids:
                logger.info(f"Refresh: excluding previously shown tops {refresh_exclude_ids}")
        except Exception:
            pass

    # Get recent top suggestions for this user+occasion (FIFO queue)
    # Only use rotation for default occasion (no manual mood or occasion)
    recent_suggestions = {}
    if is_default_mood:
        try:
            conn_suggest = get_db_connection()
            cursor_suggest = conn_suggest.cursor()
            cursor_suggest.execute(
                """SELECT item_id, last_suggested_at 
                   FROM top_suggestions 
                   WHERE user_id = %s AND occasion = %s""",
                (user_id, occasion_name)
            )
            for row in cursor_suggest.fetchall():
                recent_suggestions[row[0]] = row[1]
            cursor_suggest.close()
            conn_suggest.close()
        except Exception as e:
            logger.warning(f"Could not fetch recent suggestions: {e}")
    
    # Filter to just tops, score them with recency penalty
    tops_only = [item for item in all_items if item.get("category") == "top" and item.get("embedding")]
    
    def score_with_recency(item):
        """Score item, penalizing recently suggested tops."""
        base_score = item_score(item)
        item_id = item["id"]

        if item_id in refresh_exclude_ids:
            base_score -= 100

        if item_id in recent_suggestions:
            last_suggested = recent_suggestions[item_id]
            if last_suggested:
                days_ago = (datetime.now() - last_suggested).days
                if days_ago < 1:
                    base_score -= 20
                elif days_ago < 3:
                    base_score -= 10
                elif days_ago < 7:
                    base_score -= 5

        return base_score
    
    scored_tops = [(item, score_with_recency(item)) for item in tops_only]
    scored_tops.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Scored {len(scored_tops)} tops for occasion={occasion_name}, manual_mood={has_manual_mood}")
    for item, score in scored_tops[:10]:
        recency_note = ""
        if item["id"] in recent_suggestions:
            days = (datetime.now() - recent_suggestions[item["id"]]).days if recent_suggestions[item["id"]] else 0
            recency_note = f" (last:{days}d)"
        base = item_score(item)
        tags_str = ",".join((item.get("style_tags") or [])[:3])
        mat = (item.get("material") or "?")[:10]
        logger.info(f"  #{item['id']} {item['name']} base={base:.1f} rec={score:.1f} mat={mat} tags=[{tags_str}]{recency_note}")
    
    selected_bases = []
    used_ids = set()
    
    if scored_tops:
        best_score = scored_tops[0][1] if scored_tops else 0
        
        # Tier window: include items reasonably close to the best score
        # Jitter: random shuffle within the tier to create variety across requests
        tier_window = 10 if has_manual_mood else 8
        jitter_max = 5.0 if has_manual_mood else 4.0
        
        top_tier = [item for item, score in scored_tops if score >= best_score - tier_window]
        if len(top_tier) < 5:
            top_tier = [item for item, _ in scored_tops[:min(8, len(scored_tops))]]
        logger.info(f"Top tier ({best_score:.1f} to {best_score-tier_window:.1f}): {len(top_tier)} tops")
        
        randomized = [(item, score_with_recency(item) + random.uniform(0, jitter_max)) for item in top_tier]
        randomized.sort(key=lambda x: x[1], reverse=True)
        
        import numpy as np

        def _too_similar(candidate, selected, threshold=0.92):
            """Check if candidate's image embedding is too close to any selected item."""
            c_emb = candidate.get("embedding")
            if not c_emb:
                return False
            c_arr = np.array(c_emb)
            c_norm = np.linalg.norm(c_arr)
            if c_norm == 0:
                return False
            for s in selected:
                s_emb = s.get("embedding")
                if not s_emb:
                    continue
                s_arr = np.array(s_emb)
                s_norm = np.linalg.norm(s_arr)
                if s_norm == 0:
                    continue
                sim = float(np.dot(c_arr, s_arr) / (c_norm * s_norm))
                if sim >= threshold:
                    return True
            return False

        for item, rand_score in randomized:
            if item["id"] not in used_ids and not _too_similar(item, selected_bases):
                item["score"] = item_score(item)
                selected_bases.append(item)
                used_ids.add(item["id"])
                logger.info(f"  Selected: {item['name']} (score={item['score']:.1f}, rand={rand_score:.1f})")
            if len(selected_bases) >= 3:
                break

        if len(selected_bases) < 3:
            remaining = [(item, score) for item, score in scored_tops
                         if item["id"] not in used_ids and score > -20]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for item, score in remaining:
                if _too_similar(item, selected_bases):
                    continue
                item["score"] = score
                selected_bases.append(item)
                used_ids.add(item["id"])
                logger.info(f"  Filled: {item['name']} (score={score:.1f})")
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
    used_ids_global = set()  # Don't pre-exclude bases - only exclude as used
    
    directions = ["Classic", "Trendy", "Bold"]
    
    for idx, base_item in enumerate(selected_bases[:3]):
        # Exclude this outfit's base item from its own retrieval
        used_ids_global.add(base_item["id"])
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
        
        # Get occasion name for query text (better retrieval) and filtering
        occasion_name = occasion_info.get("occasion", "casual")
        
        # Use cascade retrieval (same pipeline as single-item generation)
        from services.outfit_generator import _cascade_one_outfit
        full_item = {**base_item, "id": base_item["id"], "name": base_item["name"],
                     "image_url": base_item["image_url"], "embedding": embedding}

        cascade_result = _cascade_one_outfit(
            full_item, base_item, embedding, direction, idx,
            used_ids_global,
            use_closet=True, user_id=user_id,
            occasion=occasion_name if not mood else None,
            mood_text=mood if mood else None,
        )

        if cascade_result is None:
            outfit = {
                "direction": f"Outfit {idx + 1}",
                "base_item": base_item,
                "items": [{"slot": base_category, **base_item}],
                "explanation": "Limited items in closet",
                "collage_url": None
            }
        else:
            best_items, score_details = cascade_result

            for slot, item in best_items.items():
                if item:
                    used_ids_global.add(item["id"])

            outfit = assemble_outfit(
                direction, base_item, best_items, embedding,
                taste_vector=taste_vector,
                dislike_vector=dislike_vector
            )
            outfit["direction"] = f"Outfit {idx + 1}"
            outfit["base_item"] = base_item
        
        # Generate collage — occasion_name is already the mood slug for mood requests
        try:
            items_for_collage = outfit.get("items", [])
            collage_path = generate_outfit_collage(
                generation_id=f"u{user_id}_{occasion_name}_{idx}",
                direction=f"outfit_{idx + 1}",
                items=items_for_collage,
                base_item={"image_url": base_item["image_url"], "category": base_category},
                force=refresh
            )
            outfit["collage_url"] = make_absolute_url(base_url, collage_path)
        except Exception as e:
            logger.error(f"Collage error: {e}")
            outfit["collage_url"] = None
        
        outfits.append(outfit)
    
    # Record suggested tops to FIFO queue (so they're deprioritized next time)
    # Only record for default occasion (no manual mood or occasion)
    if is_default_mood:
        try:
            conn_record = get_db_connection()
            cursor_record = conn_record.cursor()
            for base_item in selected_bases:
                cursor_record.execute(
                    """INSERT INTO top_suggestions (user_id, item_id, occasion, last_suggested_at, suggestion_count)
                       VALUES (%s, %s, %s, NOW(), 1)
                       ON CONFLICT (user_id, item_id, occasion) 
                       DO UPDATE SET last_suggested_at = NOW(), suggestion_count = top_suggestions.suggestion_count + 1""",
                    (user_id, base_item["id"], occasion_name)
                )
            conn_record.commit()
            cursor_record.close()
            conn_record.close()
            logger.info(f"Recorded {len(selected_bases)} top suggestions for user {user_id}, occasion {occasion_name}")
        except Exception as e:
            logger.warning(f"Could not record top suggestions: {e}")
    else:
        logger.info(f"Skipping FIFO recording for manual mood: {mood}")
    
    # Don't cache manual mood outfits - only cache default (no user mood input)
    if not has_manual_mood:
        try:
            import json
            # Store outfits with relative URLs for caching
            outfits_for_cache = []
            for o in outfits:
                cached_outfit = dict(o)
                # Convert absolute collage URL back to relative for storage
                if cached_outfit.get("collage_url") and base_url in str(cached_outfit["collage_url"]):
                    cached_outfit["collage_url"] = cached_outfit["collage_url"].replace(base_url, "")
                outfits_for_cache.append(cached_outfit)
            
            weather_for_cache = weather_data.to_dict() if weather_data else None
            
            conn_cache = get_db_connection()
            cursor_cache = conn_cache.cursor()
            cursor_cache.execute(
                """INSERT INTO daily_outfit_cache (user_id, cache_date, occasion, mood_text, outfits_json, weather_json)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (user_id, cache_date, occasion) 
                   DO UPDATE SET outfits_json = EXCLUDED.outfits_json, 
                                 weather_json = EXCLUDED.weather_json,
                                 mood_text = EXCLUDED.mood_text,
                                 created_at = NOW()""",
                (user_id, user_today, occasion_name, None, json.dumps(outfits_for_cache), json.dumps(weather_for_cache) if weather_for_cache else None)
            )
            conn_cache.commit()
            cursor_cache.close()
            conn_cache.close()
            logger.info(f"Cached daily outfits for user {user_id}, occasion {occasion_name}")
        except Exception as e:
            logger.warning(f"Could not cache daily outfits: {e}")
    else:
        logger.info("Skipping cache for manual mood outfits")
    
    response = {
        "outfits": outfits,
        "source": "closet_daily",
        "occasion": occasion_info["occasion"],
        "occasion_note": occasion_info["note"]
    }
    
    if weather_data:
        response["weather"] = weather_data.to_dict()
        response["weather_notes"] = weather_adjustments["notes"] if weather_adjustments else []
    
    return response


@app.get("/v1/closet/daily/regenerate/{idx}")
async def regenerate_single_outfit(
    request: Request,
    idx: int,
    lat: float = None,
    lon: float = None,
    exclude_ids: str = None,
    tz_offset: float = None,
    mood_text: str = None,
    base_item_id: int = None  # If provided, force use this base item (for single-item outfit regen)
):
    """
    Regenerate a single outfit (after dislike). Requires authentication.
    Excludes items from the disliked outfit to get something different.
    If base_item_id is provided, keeps the same base item (for single-item outfit generation).
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Regenerating outfit {idx} (exclude: {exclude_ids}, tz_offset: {tz_offset}, mood: {mood_text}, base_item_id: {base_item_id})")
    base_url = str(request.base_url).rstrip("/")
    
    from datetime import datetime, timedelta, timezone
    user_tz = timezone(timedelta(hours=tz_offset)) if tz_offset is not None else timezone.utc
    user_today = datetime.now(user_tz).date()
    
    # Parse exclude IDs
    excluded = set()
    if exclude_ids:
        excluded = set(int(x) for x in exclude_ids.split(",") if x.strip())
    
    # Only exclude base items from OTHER outfits if we're not forcing a specific base item
    # (For single-item outfit generation, we want to keep the same base item)
    if not base_item_id:
        try:
            conn_check = get_db_connection()
            cursor_check = conn_check.cursor()
            # Determine occasion for cache lookup
            if mood_text:
                occasion_for_cache = mood_text.strip().lower().replace(" ", "-")
            else:
                temp_occasion = get_occasion_from_time(tz_offset)
                occasion_for_cache = temp_occasion.get("occasion") if temp_occasion else "casual"
            
            cursor_check.execute(
                """SELECT outfits_json FROM daily_outfit_cache 
                   WHERE user_id = %s AND cache_date = %s AND occasion = %s""",
                (user_id, user_today, occasion_for_cache)
            )
            cached = cursor_check.fetchone()
            if cached and cached[0]:
                cached_outfits = cached[0]
                if isinstance(cached_outfits, list):
                    for i, outfit in enumerate(cached_outfits):
                        if i != idx and outfit:  # Skip the outfit being regenerated
                            base = outfit.get("base_item")
                            if base and base.get("id"):
                                excluded.add(base["id"])
                                logger.info(f"Excluding base item {base['id']} ({base.get('name')}) from other outfit {i}")
            cursor_check.close()
            conn_check.close()
        except Exception as e:
            logger.warning(f"Could not check cached outfits for exclusion: {e}")
    
    # Get taste vectors
    taste_vector, dislike_vector = get_taste_vector(user_id)
    
    # Fetch weather
    weather_data = None
    weather_adjustments = None
    if lat is not None and lon is not None:
        weather_data = await fetch_weather(lat, lon)
        if weather_data:
            weather_adjustments = get_weather_outfit_adjustments(weather_data)
    
    # Determine occasion (from mood or auto-detect)
    # Mood requests: no GPT call — embedding handles item selection
    occasion_info = None
    if mood_text:
        occasion_info = {
            "occasion": mood_text.strip().lower().replace(" ", "-"),
            "prefer_occasions": [],
            "avoid_occasions": [],
        }
    else:
        occasion_info = get_occasion_from_time(tz_offset)
    
    prefer_occasions = occasion_info.get("prefer_occasions", []) if occasion_info else []
    avoid_occasions = occasion_info.get("avoid_occasions", []) if occasion_info else []
    
    # Get all closet items
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """SELECT id, name, category, image_url, primary_color, secondary_colors,
                      style_tags, season_tags, occasion_tags, material, fit, embedding::text
               FROM user_closet_items 
               WHERE user_id = %s AND embedding IS NOT NULL
               ORDER BY RANDOM()""",  # Random order for variety
            (user_id,)
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    
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
    
    # Pick a base item NOT in excluded set, filtered by occasion
    def is_occasion_appropriate(item):
        item_occasions = item.get("occasion_tags") or []
        item_styles = item.get("style_tags") or []
        all_tags = set(item_occasions + item_styles)
        if not all_tags:
            return True
        for avoid_tag in avoid_occasions:
            if avoid_tag in all_tags:
                return False
        return True
    
    base_item = None
    
    # If base_item_id is provided (single-item outfit regen), force use that item
    if base_item_id:
        for item in all_items:
            if item["id"] == base_item_id:
                base_item = item
                logger.info(f"Using specified base item: {base_item['name']} (id={base_item_id})")
                break
    
    # Otherwise pick a new base item (daily outfit regen)
    if not base_item:
        base_categories = ["top", "layer", "bottom", "dress"]
        if weather_adjustments and weather_adjustments.get("force_layer"):
            base_categories = ["layer", "top", "bottom"]
        
        for pref_cat in base_categories:
            candidates = [i for i in all_items 
                         if i["category"] == pref_cat 
                         and i["id"] not in excluded 
                         and is_occasion_appropriate(i)]
            if candidates:
                base_item = candidates[0]  # Already randomized
                break
        
        # Fallback to any non-excluded item (skip occasion filter)
        if not base_item:
            for item in all_items:
                if item["id"] not in excluded:
                    base_item = item
                    break
    
    if not base_item:
        raise HTTPException(status_code=400, detail="No available items to create outfit")
    
    direction = ["Classic", "Trendy", "Bold"][idx % 3]
    base_category = base_item["category"]
    embedding = base_item.get("embedding")
    
    # Get slots
    slots = get_slots_for_outfit(base_category, idx)
    if weather_adjustments:
        if weather_adjustments["force_layer"] and "layer" not in slots and base_category != "layer":
            slots = slots + ["layer"]
        elif weather_adjustments["skip_layer"] and "layer" in slots:
            slots = [s for s in slots if s != "layer"]
    
    # Use cascade retrieval engine
    occasion_name = occasion_info.get("occasion") if occasion_info else None
    from services.outfit_generator import _cascade_one_outfit

    cascade_result = _cascade_one_outfit(
        base_item, base_item, embedding, direction, idx,
        excluded,
        use_closet=True, user_id=user_id,
        occasion=occasion_name if not mood_text else None,
        mood_text=mood_text if mood_text else None,
        taste_vector=taste_vector,
        dislike_vector=dislike_vector,
    )

    if cascade_result is None:
        outfit = {
            "direction": f"Outfit {idx + 1}",
            "base_item": base_item,
            "items": [{"slot": base_category, **base_item}],
            "explanation": "Limited items in closet"
        }
    else:
        best_items, _ = cascade_result
        outfit = assemble_outfit(
            direction, base_item, best_items, embedding,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
        )
        outfit["direction"] = f"Outfit {idx + 1}"
        outfit["base_item"] = base_item
    
    # Get occasion name for collage path and caching
    occasion_name = occasion_info.get("occasion") if occasion_info else "casual"
    
    # Generate new collage (forced) - use user_id and occasion in path
    try:
        items_for_collage = outfit.get("items", [])
        collage_path = generate_outfit_collage(
            generation_id=f"u{user_id}_{occasion_name}_{idx}",
            direction=f"outfit_{idx + 1}",
            items=items_for_collage,
            base_item={"image_url": base_item["image_url"], "category": base_category},
            force=True
        )
        outfit["collage_url"] = make_absolute_url(base_url, collage_path)
    except Exception as e:
        logger.error(f"Collage error: {e}")
        outfit["collage_url"] = None
    
    # Don't write manual mood outfits to daily cache
    if not (mood_text and mood_text.strip()):
        try:
            conn_cache = get_db_connection()
            cursor_cache = conn_cache.cursor()
            
            # Get current cached outfits
            cursor_cache.execute(
                """SELECT outfits_json FROM daily_outfit_cache 
                   WHERE user_id = %s AND cache_date = %s AND occasion = %s""",
                (user_id, user_today, occasion_name)
            )
            cached = cursor_cache.fetchone()
            
            if cached and cached[0]:
                cached_outfits = cached[0]
                if isinstance(cached_outfits, list) and 0 <= idx < len(cached_outfits):
                    # Prepare outfit for cache (remove embeddings, etc.)
                    outfit_for_cache = {
                        "direction": outfit.get("direction"),
                        "explanation": outfit.get("explanation"),
                        "collage_url": collage_path,  # Store relative path
                        "base_item": {k: v for k, v in base_item.items() if k != "embedding"},
                        "items": [{k: v for k, v in item.items() if k != "embedding"} for item in outfit.get("items", [])]
                    }
                    cached_outfits[idx] = outfit_for_cache
                    
                    # Save back to cache
                    cursor_cache.execute(
                        """UPDATE daily_outfit_cache 
                           SET outfits_json = %s, created_at = NOW()
                           WHERE user_id = %s AND cache_date = %s AND occasion = %s""",
                        (json.dumps(cached_outfits), user_id, user_today, occasion_name)
                    )
                    conn_cache.commit()
                    logger.info(f"Updated cache with regenerated outfit {idx} for user {user_id}")
            
            cursor_cache.close()
            conn_cache.close()
        except Exception as e:
            logger.error(f"Failed to update cache: {e}")
    else:
        logger.info("Skipping cache update for manual mood outfit")
    
    return outfit


# Mount Shopify API last so it does not shadow /, /health, /static, or /v1/* routes.
app.mount("/", shopify_app_instance)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
