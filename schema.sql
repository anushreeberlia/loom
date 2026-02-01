-- catalog_items: Your clothing inventory (300-800 items)
CREATE TABLE catalog_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,        -- top, bottom, shoes, layer, accessory, bag, dress
    image_url TEXT NOT NULL,
    product_url TEXT,                      -- shop link (optional)
    
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
    
    -- Vector embedding (1536 dims from OpenAI text-embedding-3-small)
    embedding vector(1536),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast vector similarity search
CREATE INDEX ON catalog_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- outfit_generations: Tracks each generation request
CREATE TABLE outfit_generations (
    id SERIAL PRIMARY KEY,
    input_image_url TEXT,
    input_description TEXT,                -- plain text description from vision
    parsed_tags JSONB,                     -- BaseItem JSON extracted from description
    base_item_embedding vector(1536),      -- embedding for retrieval
    output_outfits JSONB,                  -- the 3 outfits returned
    input_type TEXT,                       -- image or text
    created_at TIMESTAMP DEFAULT NOW()
);

-- feedback_events: Like/dislike tracking
CREATE TABLE feedback_events (
    id SERIAL PRIMARY KEY,
    generation_id INTEGER REFERENCES outfit_generations(id),
    outfit_index INTEGER,                  -- 1, 2, or 3
    liked BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);