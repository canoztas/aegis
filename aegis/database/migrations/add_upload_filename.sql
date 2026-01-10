-- Add upload_filename column to scans table
-- Migration: add_upload_filename
-- Date: 2026-01-08

ALTER TABLE scans ADD COLUMN upload_filename TEXT;

-- Create index for faster searches
CREATE INDEX IF NOT EXISTS idx_scans_upload_filename ON scans(upload_filename);
