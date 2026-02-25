-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    password_hash TEXT,                    -- bcrypt hash for email/password auth (null for OAuth-only users)
    profile_image TEXT,
    google_id VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP DEFAULT NOW()
);

-- Index for fast email lookup
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);

-- catalog_items: Your clothing inventory (300-800 items)
CREATE TABLE catalog_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,        -- top, bottom, shoes, layer, accessory, bag, dress
    image_url TEXT NOT NULL,
    product_url TEXT,                      -- shop link (optional)
    
    -- Source tracking (for multi-catalog support)
    source TEXT DEFAULT 'kaggle_fashion', -- kaggle_fashion, h_and_m
    source_item_id TEXT,                   -- original ID from source dataset
    brand TEXT,                            -- H&M, etc.
    
    -- Colors
    primary_color TEXT,                    -- main color (e.g., 'navy')
    secondary_colors TEXT[],               -- other colors (e.g., ['white', 'gold'])
    
    -- Tags (populated by LLM tagger)
    style_tags TEXT[],                     -- ['classic', 'minimalist']
    season_tags TEXT[],                    -- ['spring', 'fall']
    occasion_tags TEXT[],                  -- ['work', 'casual']
    material VARCHAR(100),
    fit VARCHAR(50),
    
    -- Tagging metadata
    tagged_at TIMESTAMP,                   -- when LLM tagged this item
    tagging_error TEXT,                    -- error message if tagging failed
    
    -- Vector embedding (512 dims from FashionCLIP 2.0)
    embedding vector(512),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast vector similarity search (HNSW for better recall)
CREATE INDEX ON catalog_items USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- outfit_generations: Tracks each generation request
CREATE TABLE outfit_generations (
    id SERIAL PRIMARY KEY,
    input_image_url TEXT,
    input_image_hash TEXT,                 -- SHA-256 hash for caching
    input_description TEXT,                -- plain text description from vision
    parsed_tags JSONB,                     -- BaseItem JSON extracted from description
    base_item_embedding vector(512),       -- embedding for retrieval
    output_outfits JSONB,                  -- the 3 outfits returned
    input_type TEXT,                       -- image or text
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast cache lookups by image hash
CREATE INDEX idx_generations_image_hash ON outfit_generations(input_image_hash);

-- feedback_events: Like/dislike tracking (one per generation + outfit)
CREATE TABLE feedback_events (
    id SERIAL PRIMARY KEY,
    generation_id INTEGER REFERENCES outfit_generations(id),
    outfit_index INTEGER NOT NULL,         -- 0, 1, or 2 (Classic, Trendy, Bold)
    liked BOOLEAN NOT NULL,
    session_id TEXT,                        -- Session ID for taste vector tracking
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(generation_id, outfit_index)    -- One feedback per generation + outfit
);

-- taste_vectors: Per-session style preferences
CREATE TABLE taste_vectors (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    taste_embedding vector(512),            -- Aggregated preference embedding (likes)
    dislike_embedding vector(512),          -- Aggregated dislike embedding (to penalize)
    like_count INTEGER DEFAULT 0,           -- Number of likes contributing to taste
    dislike_count INTEGER DEFAULT 0,        -- Number of dislikes
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- user_closet_items: Personal wardrobe inventory
CREATE TABLE user_closet_items (
    id SERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',         -- For future multi-user support
    name VARCHAR(255),                       -- Auto-generated or user-provided
    category VARCHAR(50) NOT NULL,           -- top, bottom, shoes, layer, accessory, dress
    image_url TEXT NOT NULL,                 -- Cloudinary URL
    
    -- Colors
    primary_color TEXT,
    secondary_colors TEXT[],
    
    -- Tags (from vision/parser pipeline)
    style_tags TEXT[],
    season_tags TEXT[],
    occasion_tags TEXT[],
    material VARCHAR(100),
    fit VARCHAR(50),
    
    -- Vector embedding for retrieval (512 dims from FashionCLIP 2.0)
    embedding vector(512),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for vector similarity search on closet items (HNSW for better recall)
CREATE INDEX idx_closet_embedding ON user_closet_items 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
    
-- Index for filtering by user
CREATE INDEX idx_closet_user ON user_closet_items(user_id);

-- top_suggestions: FIFO queue for top rotation (don't show same tops every day)
CREATE TABLE IF NOT EXISTS top_suggestions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    item_id INTEGER REFERENCES user_closet_items(id) ON DELETE CASCADE,
    occasion VARCHAR(50) NOT NULL,          -- work, casual, going-out, etc.
    last_suggested_at TIMESTAMP DEFAULT NOW(),
    suggestion_count INTEGER DEFAULT 1,      -- how many times this top has been suggested
    UNIQUE(user_id, item_id, occasion)
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_top_suggestions_user_occasion 
    ON top_suggestions(user_id, occasion, last_suggested_at);

-- saved_outfits: Bookmarked and worn outfits
CREATE TABLE IF NOT EXISTS saved_outfits (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    outfit_data JSONB NOT NULL,              -- full outfit items with IDs, names, image_urls
    collage_url TEXT,                        -- generated collage image
    occasion VARCHAR(50),                    -- work, casual, going-out
    base_item_id INTEGER REFERENCES user_closet_items(id) ON DELETE SET NULL,
    saved_at TIMESTAMP DEFAULT NOW(),
    worn_at TIMESTAMP,                       -- NULL if not worn yet
    status VARCHAR(20) DEFAULT 'saved'       -- 'saved', 'worn'
);

-- Index for listing user's saved outfits
CREATE INDEX IF NOT EXISTS idx_saved_outfits_user ON saved_outfits(user_id, saved_at DESC);

-- daily_outfit_cache: Cache daily outfits so they persist for the day
CREATE TABLE IF NOT EXISTS daily_outfit_cache (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    cache_date DATE NOT NULL DEFAULT CURRENT_DATE,
    occasion VARCHAR(50),                        -- work, casual, going-out
    mood_text TEXT,                              -- user's mood input if any
    outfits_json JSONB NOT NULL,                 -- the 3 daily outfits
    weather_json JSONB,                          -- weather data at generation time
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, cache_date, occasion)
);

CREATE INDEX IF NOT EXISTS idx_daily_cache_user_date ON daily_outfit_cache(user_id, cache_date);