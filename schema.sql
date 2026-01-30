-- catalog_items: Your clothing inventory (300-800 items)
CREATE TABLE catalog_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,        -- top, bottom, shoes, layer, accessory, bag, dress
    image_url TEXT NOT NULL,
    product_url TEXT,                      -- shop link (optional)
    
    -- Tags (populated by LLM tagger)
    colors TEXT[],                         -- ['navy', 'white']
    style_tags TEXT[],                     -- ['classic', 'minimalist']
    season_tags TEXT[],                    -- ['spring', 'fall']
    occasion_tags TEXT[],                  -- ['work', 'casual']
    material VARCHAR(100),
    fit VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- outfit_generations: Tracks each generation request
CREATE TABLE outfit_generations (
    id SERIAL PRIMARY KEY,
    input_image_url TEXT,
    parsed_tags JSONB,                     -- tags extracted from input image
    output_outfits JSONB,                  -- the 3 outfits returned
    input_type TEXT,                     -- image or text
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