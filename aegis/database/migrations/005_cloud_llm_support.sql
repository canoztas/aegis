-- Migration 005: Cloud LLM Support (Phase E)
-- Adds tables for API credentials and usage tracking

-- Table for storing encrypted API credentials
CREATE TABLE IF NOT EXISTS api_credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    key_name TEXT NOT NULL,
    key_value TEXT NOT NULL,
    encrypted BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, key_name)
);

-- Index for fast credential lookups
CREATE INDEX IF NOT EXISTS idx_credentials_provider ON api_credentials(provider, key_name);

-- Table for tracking API usage and costs
CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    scan_id TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    request_id TEXT,
    metadata TEXT
);

-- Indexes for usage queries
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_provider ON api_usage(provider, model_name);
CREATE INDEX IF NOT EXISTS idx_usage_scan_id ON api_usage(scan_id);
CREATE INDEX IF NOT EXISTS idx_usage_cost ON api_usage(cost_usd);

-- Trigger to update updated_at on credentials
CREATE TRIGGER IF NOT EXISTS update_credentials_timestamp
AFTER UPDATE ON api_credentials
FOR EACH ROW
BEGIN
    UPDATE api_credentials SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
