-- Migration: Add batch processing columns to user_closet_items
-- Run this on existing databases to support the new ingestion pipeline.

-- Allow category to be NULL during pending state (vision sets it later)
ALTER TABLE user_closet_items ALTER COLUMN category DROP NOT NULL;

-- Add processing state columns
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'ready';
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS batch_id UUID;
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS processing_error TEXT;
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP;
ALTER TABLE user_closet_items ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;

-- Partial indexes for worker queries
CREATE INDEX IF NOT EXISTS idx_closet_status_pending ON user_closet_items (status, created_at)
    WHERE status IN ('pending', 'processing');
CREATE INDEX IF NOT EXISTS idx_closet_batch ON user_closet_items (batch_id)
    WHERE batch_id IS NOT NULL;
