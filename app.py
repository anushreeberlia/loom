from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Cookie, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
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
from services.weather import fetch_weather, get_weather_outfit_adjustments, get_occasion_from_time, get_material_weather_score, WeatherData
from services.image_processor import process_clothing_image
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


def interpret_mood_for_occasion(mood: str) -> dict:
    """
    Use AI to interpret a free-form mood description into occasion preferences.
    
    Args:
        mood: Free-form text like "cozy day at home", "fancy dinner date", etc.
        
    Returns:
        Dict with occasion, prefer_occasions, avoid_occasions, note, needs_layer
    """
    if not OPENAI_API_KEY:
        # Fallback to casual if no API key
        return {
            "occasion": "casual",
            "prefer_occasions": ["casual", "everyday"],
            "avoid_occasions": [],
            "note": mood,
            "needs_layer": None  # Let weather decide
        }
    
    prompt = f"""Based on this mood/feeling description, determine the appropriate outfit occasion.

Mood: "{mood}"

Return JSON with:
- occasion: MUST be exactly one of: work, casual, going-out, smart-casual, workout
- prefer_occasions: array of 3-5 tags to look for
- avoid_occasions: array of 3-5 tags to exclude
- note: short summary (max 30 chars, no emoji)
- needs_layer: boolean - true if going outside/commuting/needs jacket, false if staying indoors/at home/gym, null if unclear

Examples:
- "cozy day at home" → needs_layer: false (indoor, no jacket needed)
- "office today" → needs_layer: true (going outside, commuting)
- "dinner date" → needs_layer: true (going out)
- "working from home" → needs_layer: false (indoor)
- "gym session" → needs_layer: false (indoor workout)

JSON only."""

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a fashion stylist. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            import json
            text = response.json()["choices"][0]["message"]["content"]
            # Clean up markdown if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            
            result = json.loads(text)
            
            # Normalize occasion to known values
            KNOWN_OCCASIONS = {"work", "casual", "going-out", "smart-casual", "workout"}
            OCCASION_ALIASES = {
                "active": "workout", "gym": "workout", "fitness": "workout", "exercise": "workout",
                "party": "going-out", "date": "going-out", "dinner": "going-out", "night-out": "going-out",
                "formal": "smart-casual", "business": "work", "office": "work",
                "brunch": "casual", "cozy": "casual", "relaxed": "casual"
            }
            raw_occasion = result.get("occasion", "casual").lower()
            if raw_occasion not in KNOWN_OCCASIONS:
                raw_occasion = OCCASION_ALIASES.get(raw_occasion, "casual")
            
            # Parse needs_layer - can be true, false, or null
            needs_layer = result.get("needs_layer")
            if needs_layer is not None:
                needs_layer = bool(needs_layer)
            
            return {
                "occasion": raw_occasion,
                "prefer_occasions": result.get("prefer_occasions", ["casual", "everyday"]),
                "avoid_occasions": result.get("avoid_occasions", []),
                "note": result.get("note", mood),
                "needs_layer": needs_layer
            }
    except Exception as e:
        logger.error(f"Mood interpretation error: {e}")
    
    # Fallback
    return {
        "occasion": "casual",
        "prefer_occasions": ["casual", "everyday"],
        "avoid_occasions": [],
        "note": mood,
        "needs_layer": None  # Let weather decide
    }


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
    if base.startswith("http://") and "railway.app" in base:
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
        if base.startswith("http://") and "railway.app" in base:
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
                exclude_ids=[],  # No hard exclusions - taste vectors handle preferences
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
    """Save an outfit for later. Requires authentication."""
    user_id = require_auth(auth_token)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        import json
        cursor.execute(
            """INSERT INTO saved_outfits (user_id, outfit_data, collage_url, occasion, base_item_id, status)
               VALUES (%s, %s, %s, %s, %s, 'saved')
               RETURNING id""",
            (user_id, json.dumps(req.outfit_data), req.collage_url, req.occasion, req.base_item_id)
        )
        outfit_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Outfit saved: id={outfit_id}, user={user_id}")
        return {"status": "saved", "outfit_id": outfit_id}
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
            outfits.append({
                "id": row[0],
                "outfit_data": row[1],
                "collage_url": row[2],
                "occasion": row[3],
                "saved_at": row[4].isoformat() if row[4] else None
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
            outfits.append({
                "id": row[0],
                "outfit_data": row[1],
                "collage_url": row[2],
                "occasion": row[3],
                "worn_at": row[4].isoformat() if row[4] else None
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
async def add_closet_item(request: Request, file: UploadFile = File(...)):
    """
    Upload a new item to user's closet. Requires authentication.
    Processes image through vision/parser pipeline and stores with embedding.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
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
                
                # Re-run vision + parser
                description = describe_image(response.content)
                parsed = parse_description(description)
                
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
        
        # Re-run vision + parser
        description = describe_image(response.content)
        parsed = parse_description(description)
        
        # Generate new name
        name = f"{parsed.get('primary_color', '')} {parsed.get('category', 'item')}".strip().title()
        
        # Generate new embedding with updated tags
        embedding = embed_base_item(parsed)
        
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
            "description": description,
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
    lon: float = Form(None)
):
    """
    Generate outfits using ONLY items from user's closet. Requires authentication.
    Either pick an existing closet item (item_id) or upload a new image.
    Optionally pass lat/lon to factor in weather.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
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
        # If weather forces layer, make it required
        require_layer = weather_adjustments.get("force_layer", False) if weather_adjustments else False
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8,
            require_layer=require_layer
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
    refresh: bool = False,  # If true, regenerate even if cached
    tz_offset: float = None,  # Timezone offset from UTC in hours (e.g., -8 for PST)
    occasion: str = None,  # Manual occasion override (work, casual, going-out, etc.)
    mood: str = None  # Free-form mood description (e.g., "cozy day", "fancy dinner")
):
    """
    Generate 3 weather-appropriate outfits automatically. Requires authentication.
    Picks different base items from closet for variety.
    Results are cached per user/day/occasion - only regenerated on explicit refresh.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Daily outfits: lat={lat}, lon={lon}, tz={tz_offset}, mood={mood}, refresh={refresh}")
    base_url = str(request.base_url).rstrip("/")
    
    # Determine occasion first (needed for cache lookup)
    # Get occasion - from mood description, manual selection, or auto-detect
    if mood:
        # Interpret mood using AI
        occasion_info = interpret_mood_for_occasion(mood)
        logger.info(f"Mood '{mood}' -> {occasion_info['occasion']} - {occasion_info['note']}")
    elif occasion:
        # Manual occasion selected (fallback for dropdown if used)
        OCCASION_CONFIGS = {
            "work": {
                "occasion": "work",
                "prefer_occasions": ["work", "office", "business-casual", "professional", "elegant", "classic"],
                "avoid_occasions": [],  # Semantic filtering handles this now
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
    
    # Check cache first (unless refresh requested)
    if not refresh:
        conn_cache = get_db_connection()
        cursor_cache = conn_cache.cursor()
        try:
            cursor_cache.execute(
                """SELECT outfits_json, weather_json FROM daily_outfit_cache 
                   WHERE user_id = %s AND cache_date = CURRENT_DATE AND occasion = %s""",
                (user_id, occasion_name)
            )
            cached = cursor_cache.fetchone()
            if cached:
                logger.info(f"Cache hit for user {user_id}, occasion {occasion_name}")
                cached_outfits = cached[0]
                cached_weather = cached[1]
                
                # Verify collage files exist - if not, regenerate them
                import os
                collages_valid = True
                for idx, outfit in enumerate(cached_outfits):
                    collage_url = outfit.get("collage_url", "")
                    # Extract file path from URL (handle both relative and absolute)
                    if collage_url and not collage_url.startswith("http"):
                        file_path = collage_url.lstrip("/")
                        if not os.path.exists(file_path):
                            logger.info(f"Collage missing: {file_path}, regenerating...")
                            # Regenerate this collage
                            try:
                                items_for_collage = outfit.get("items", [])
                                base_item = outfit.get("base_item", {})
                                collage_path = generate_outfit_collage(
                                    generation_id=f"daily_{idx}",
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
        score = 0
        
        # Season scoring
        season_tags = item.get("season_tags") or []
        for s in preferred_seasons:
            if s in season_tags or "all-season" in season_tags:
                score += 1
        for s in avoid_seasons:
            if s in season_tags:
                score -= 2
        
        # Occasion AND style scoring - check both tag types
        occasion_tags = item.get("occasion_tags") or []
        style_tags = item.get("style_tags") or []
        all_item_tags = set(occasion_tags + style_tags)
        
        # Count direct tag matches (not "everyday" fallback)
        direct_matches = sum(1 for o in prefer_occasions if o in all_item_tags)
        score += direct_matches * 5  # Strong boost for direct matches
        
        # Penalize items with avoided tags
        for o in avoid_occasions:
            if o in all_item_tags:
                score -= 5  # Strong penalty for avoided tags
        
        # SEMANTIC scoring using embeddings - this is the key for workout/active occasions
        if item.get("embedding") and occasion_name:
            semantic_score = compute_occasion_score(item["embedding"], occasion_name, all_item_tags)
            score += semantic_score * 15  # Strong weight for semantic fit
        
        # Material scoring for weather (especially important for layers)
        if weather_adjustments:
            material = item.get("material") or ""
            material_score = get_material_weather_score(material, weather_adjustments)
            # Extra weight for layers - material matters more
            if item.get("category") == "layer":
                material_score *= 2
            score += material_score
        
        return score
    
    # Pick top 3 TOPS as base items with FIFO rotation
    # Tops suggested recently are penalized to ensure variety across days
    import random
    from datetime import datetime, timedelta
    
    # Get recent top suggestions for this user+occasion (FIFO queue)
    recent_suggestions = {}
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
        
        if item_id in recent_suggestions:
            last_suggested = recent_suggestions[item_id]
            if last_suggested:
                days_ago = (datetime.now() - last_suggested).days
                # Heavy penalty for recently suggested (within 3 days)
                # Gradually decreases over time
                if days_ago < 1:
                    base_score -= 20  # Very recent - strong penalty
                elif days_ago < 3:
                    base_score -= 10  # Recent - moderate penalty
                elif days_ago < 7:
                    base_score -= 5   # Within a week - small penalty
                # After 7 days, no penalty - back in rotation
        
        return base_score
    
    scored_tops = [(item, score_with_recency(item)) for item in tops_only]
    scored_tops.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Scored {len(scored_tops)} tops for occasion: {occasion_name}")
    # Log top 5 for debugging
    for item, score in scored_tops[:5]:
        recency_note = ""
        if item["id"] in recent_suggestions:
            days = (datetime.now() - recent_suggestions[item["id"]]).days if recent_suggestions[item["id"]] else 0
            recency_note = f" (last: {days}d ago)"
        logger.info(f"  Top candidate: {item['name']} score={score:.1f}{recency_note}")
    
    selected_bases = []
    used_ids = set()
    
    if scored_tops:
        best_score = scored_tops[0][1] if scored_tops else 0
        
        # Get tops within 5 points of best score (wider window for more variety)
        top_tier = [item for item, score in scored_tops if score >= best_score - 5]
        logger.info(f"Top tier ({best_score:.1f} to {best_score-5:.1f}): {len(top_tier)} tops")
        
        # Add randomness to scoring to get different selections each time
        # Each item gets a random boost of 0-3 points
        randomized = [(item, score_with_recency(item) + random.uniform(0, 3)) for item in top_tier]
        randomized.sort(key=lambda x: x[1], reverse=True)
        
        for item, rand_score in randomized:
            if item["id"] not in used_ids:
                item["score"] = item_score(item)  # Store actual score (without recency penalty)
                selected_bases.append(item)
                used_ids.add(item["id"])
                logger.info(f"  Selected: {item['name']} (score={item['score']:.1f}, rand={rand_score:.1f})")
            if len(selected_bases) >= 3:
                break
        
        # Fill from remaining tops if needed
        if len(selected_bases) < 3:
            remaining = [(item, score) for item, score in scored_tops if item["id"] not in used_ids]
            random.shuffle(remaining)
            for item, score in remaining:
                item["score"] = score
                selected_bases.append(item)
                used_ids.add(item["id"])
                logger.info(f"  Filled: {item['name']} (score={score})")
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
        
        # Build query texts WITH occasion for better retrieval
        query_texts = []
        for slot in slots:
            query_text = build_query_text(base_item, direction, slot, {}, occasion=occasion_name)
            query_texts.append(query_text)
        
        # Get embeddings
        query_embeddings = get_batch_embeddings(query_texts)
        candidates_by_slot = {}
        for i, slot in enumerate(slots):
            try:
                # First try without re-using items from other outfits
                candidates = retrieve_for_slot(
                    base_item=base_item,
                    direction=direction,
                    slot=slot,
                    exclude_ids=list(used_ids_global),  # Prefer diversity
                    chosen_items={},
                    k=10,
                    precomputed_embedding=query_embeddings[i],
                    use_closet=True,
                    user_id=user_id,
                    occasion=occasion_name
                )
                # If no good candidates, allow re-use
                if not candidates:
                    candidates = retrieve_for_slot(
                        base_item=base_item,
                        direction=direction,
                        slot=slot,
                        exclude_ids=[],  # Allow re-use as fallback
                        chosen_items={},
                        k=10,
                        precomputed_embedding=query_embeddings[i],
                        use_closet=True,
                        user_id=user_id,
                        occasion=occasion_name
                    )
                candidates_by_slot[slot] = candidates
            except Exception as e:
                logger.error(f"Retrieval error for {slot}: {e}")
                candidates_by_slot[slot] = []
        
        # Generate and score
        # If weather forces layer, make it required
        require_layer = weather_adjustments.get("force_layer", False) if weather_adjustments else False
        candidate_outfits = generate_candidate_outfits(
            slots=slots,
            candidates_by_slot=candidates_by_slot,
            max_candidates=8,
            require_layer=require_layer
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
                base_embedding=embedding,
                taste_vector=taste_vector,
                dislike_vector=dislike_vector
            )
            
            # Track used items
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
        
        # Generate collage
        try:
            items_for_collage = outfit.get("items", [])
            collage_path = generate_outfit_collage(
                generation_id=f"daily_{idx}",
                direction=f"outfit_{idx + 1}",
                items=items_for_collage,
                base_item={"image_url": base_item["image_url"], "category": base_category},
                force=refresh  # Force regeneration when refreshing
            )
            outfit["collage_url"] = make_absolute_url(base_url, collage_path)
        except Exception as e:
            logger.error(f"Collage error: {e}")
            outfit["collage_url"] = None
        
        outfits.append(outfit)
    
    # Record suggested tops to FIFO queue (so they're deprioritized next time)
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
    
    # Cache the generated outfits for today
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
               VALUES (%s, CURRENT_DATE, %s, %s, %s, %s)
               ON CONFLICT (user_id, cache_date, occasion) 
               DO UPDATE SET outfits_json = EXCLUDED.outfits_json, 
                             weather_json = EXCLUDED.weather_json,
                             mood_text = EXCLUDED.mood_text,
                             created_at = NOW()""",
            (user_id, occasion_name, mood, json.dumps(outfits_for_cache), json.dumps(weather_for_cache) if weather_for_cache else None)
        )
        conn_cache.commit()
        cursor_cache.close()
        conn_cache.close()
        logger.info(f"Cached daily outfits for user {user_id}, occasion {occasion_name}")
    except Exception as e:
        logger.warning(f"Could not cache daily outfits: {e}")
    
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
    mood_text: str = None
):
    """
    Regenerate a single outfit (after dislike). Requires authentication.
    Excludes items from the disliked outfit to get something different.
    """
    auth_token = request.cookies.get("auth_token")
    user_id = require_auth(auth_token)
    logger.info(f"Regenerating outfit {idx} (exclude: {exclude_ids}, tz_offset: {tz_offset}, mood: {mood_text})")
    base_url = str(request.base_url).rstrip("/")
    
    # Parse exclude IDs
    excluded = set()
    if exclude_ids:
        excluded = set(int(x) for x in exclude_ids.split(",") if x.strip())
    
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
    occasion_info = None
    if mood_text:
        occasion_info = interpret_mood_for_occasion(mood_text)
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
    
    base_categories = ["top", "layer", "bottom", "dress"]
    if weather_adjustments and weather_adjustments.get("force_layer"):
        base_categories = ["layer", "top", "bottom"]
    
    base_item = None
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
    
    # Build queries and get embeddings (with occasion if present)
    occasion_name = occasion_info.get("occasion") if occasion_info else None
    query_texts = [build_query_text(base_item, direction, slot, {}, occasion=occasion_name) for slot in slots]
    query_embeddings = get_batch_embeddings(query_texts)
    
    # Retrieve candidates (excluding disliked items)
    candidates_by_slot = {}
    for i, slot in enumerate(slots):
        try:
            candidates = retrieve_for_slot(
                base_item=base_item,
                direction=direction,
                slot=slot,
                exclude_ids=list(excluded),
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
    
    # Generate outfit
    require_layer = weather_adjustments.get("force_layer", False) if weather_adjustments else False
    candidate_outfits = generate_candidate_outfits(
        slots=slots,
        candidates_by_slot=candidates_by_slot,
        max_candidates=8,
        require_layer=require_layer
    )
    
    if not candidate_outfits:
        outfit = {
            "direction": f"Outfit {idx + 1}",
            "base_item": base_item,
            "items": [{"slot": base_category, **base_item}],
            "explanation": "Limited items in closet"
        }
    else:
        best_items, _ = select_best_outfit(
            candidate_outfits=candidate_outfits,
            base_item=base_item,
            direction=direction,
            base_embedding=embedding,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
        )
        
        outfit = assemble_outfit(
            direction, base_item, best_items, embedding,
            taste_vector=taste_vector,
            dislike_vector=dislike_vector
        )
        outfit["direction"] = f"Outfit {idx + 1}"
        outfit["base_item"] = base_item
    
    # Generate new collage (forced)
    try:
        items_for_collage = outfit.get("items", [])
        collage_path = generate_outfit_collage(
            generation_id=f"daily_{idx}_regen_{uuid.uuid4().hex[:6]}",
            direction=f"outfit_{idx + 1}",
            items=items_for_collage,
            base_item={"image_url": base_item["image_url"], "category": base_category},
            force=True
        )
        outfit["collage_url"] = make_absolute_url(base_url, collage_path)
    except Exception as e:
        logger.error(f"Collage error: {e}")
        outfit["collage_url"] = None
    
    return outfit


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
