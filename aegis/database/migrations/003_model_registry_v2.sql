-- Migration 003: Extend models table for ModelRegistryV2
-- Adds support for multiple roles, parsers, and enhanced metadata

-- Add new columns to models table
ALTER TABLE models ADD COLUMN roles_json TEXT DEFAULT '[]';  -- JSON array of roles
ALTER TABLE models ADD COLUMN parser_id TEXT DEFAULT NULL;   -- Parser identifier
ALTER TABLE models ADD COLUMN model_type TEXT DEFAULT 'ollama_local';  -- Model type enum
ALTER TABLE models ADD COLUMN status TEXT DEFAULT 'registered';  -- registered/disabled/unavailable

-- Create index on model_type for faster filtering
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);

-- Migrate existing role column to roles_json (single role → array)
UPDATE models
SET roles_json = CASE
    WHEN role IS NOT NULL THEN json_array(role)
    ELSE '[]'
END
WHERE roles_json = '[]';

-- Note: We keep the old 'role' column for backward compatibility
-- New code should use roles_json; old code continues to work
