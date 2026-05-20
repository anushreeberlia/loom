-- Migration: Add DINOv2 multi-head embedding columns.
-- Each head produces a 128-dim L2-normalized vector for task-specific retrieval.
-- The backbone_embedding stores the raw 768-dim DINOv2 CLS token (for re-projection if heads are retrained).

-- DINOv2 backbone embedding (768-dim) -- kept for re-projection when heads are retrained
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS backbone_embedding vector(768);
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS backbone_embedding vector(768);

-- Multi-head projection embeddings (128-dim each)
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS style_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS fit_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS material_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS compat_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS occasion_embedding vector(128);

ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS style_embedding vector(128);
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS fit_embedding vector(128);
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS material_embedding vector(128);
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS compat_embedding vector(128);
ALTER TABLE catalog_items ADD COLUMN IF NOT EXISTS occasion_embedding vector(128);

-- HNSW indexes on heads used for primary retrieval (closet items)
-- compat_head: main retrieval axis for outfit building ("what goes with this?")
CREATE INDEX IF NOT EXISTS idx_closet_compat_emb
    ON user_closet_items USING hnsw (compat_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- style_head: direction-based retrieval (Classic/Trendy/Bold)
CREATE INDEX IF NOT EXISTS idx_closet_style_emb
    ON user_closet_items USING hnsw (style_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- occasion_head: occasion filtering
CREATE INDEX IF NOT EXISTS idx_closet_occasion_emb
    ON user_closet_items USING hnsw (occasion_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Catalog items indexes (same heads)
CREATE INDEX IF NOT EXISTS idx_catalog_compat_emb
    ON catalog_items USING hnsw (compat_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_catalog_style_emb
    ON catalog_items USING hnsw (style_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_catalog_occasion_emb
    ON catalog_items USING hnsw (occasion_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
