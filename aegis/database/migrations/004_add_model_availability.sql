-- Migration 004: Track model availability/health
-- Safe to run once; subsequent runs may no-op if columns already exist.

-- Add availability columns to models table
ALTER TABLE models ADD COLUMN availability TEXT DEFAULT 'unknown';
ALTER TABLE models ADD COLUMN availability_checked_at TIMESTAMP NULL;

-- Helpful index for filtering by availability
CREATE INDEX IF NOT EXISTS idx_models_availability ON models(availability);
