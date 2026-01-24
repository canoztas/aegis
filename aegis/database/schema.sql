-- Aegis Database Schema
-- SQLite schema for persistent storage of providers, models, scans, findings, and telemetry

-- Provider configurations (Ollama, OpenAI, Anthropic, custom providers)
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,           -- 'ollama', 'openai', 'anthropic', 'custom'
    type TEXT NOT NULL,                  -- 'llm', 'classic', 'rest'
    base_url TEXT,
    config_json TEXT,                    -- JSON blob for provider-specific config
    rate_limit_per_second REAL DEFAULT 10.0,
    timeout_seconds INTEGER DEFAULT 300,
    retry_max_attempts INTEGER DEFAULT 3,
    retry_backoff_factor REAL DEFAULT 2.0,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models (instances of providers with role assignment)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL,
    model_id TEXT UNIQUE NOT NULL,       -- e.g., 'ollama:qwen2.5-coder:7b'
    display_name TEXT NOT NULL,
    model_name TEXT NOT NULL,            -- actual model identifier for API
    role TEXT DEFAULT 'scan',            -- DEPRECATED: Use roles_json instead. Kept for backward compatibility.
    roles_json TEXT DEFAULT '[]',        -- JSON array of roles: ["deep_scan", "judge"]
    config_json TEXT,                    -- model-specific overrides
    weight REAL DEFAULT 1.0,             -- for weighted consensus
    enabled BOOLEAN DEFAULT 1,
    supports_streaming BOOLEAN DEFAULT 0,
    supports_json BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
);

-- Scans (scan history with status tracking)
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT UNIQUE NOT NULL,        -- UUID
    status TEXT NOT NULL,                -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    upload_filename TEXT,                -- Original uploaded file name
    pipeline_config_json TEXT,           -- LEGACY: Stores scan config (models, judge_model_id). Name kept for DB compatibility.
    consensus_strategy TEXT DEFAULT 'union',
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    processed_chunks INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scan files (for code snippet display)
CREATE TABLE IF NOT EXISTS scan_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT,
    lines_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
);

-- Findings (vulnerability findings - both per-model and consensus)
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    model_id TEXT,                       -- NULL if consensus finding
    is_consensus BOOLEAN DEFAULT 0,
    fingerprint TEXT NOT NULL,
    name TEXT NOT NULL,
    severity TEXT NOT NULL,
    cwe TEXT NOT NULL,
    file TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    message TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE
);

-- Model execution telemetry (per-model performance tracking)
CREATE TABLE IF NOT EXISTS model_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    file_path TEXT,
    chunk_index INTEGER,
    status TEXT NOT NULL,                -- 'success', 'error', 'retried'
    latency_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    token_usage_json TEXT,               -- JSON blob: {prompt_tokens, completion_tokens, total_tokens}
    cost_usd REAL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE SET NULL
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_upload_filename ON scans(upload_filename);
CREATE INDEX IF NOT EXISTS idx_findings_scan_id ON findings(scan_id);
CREATE INDEX IF NOT EXISTS idx_findings_consensus ON findings(is_consensus);
CREATE INDEX IF NOT EXISTS idx_findings_fingerprint ON findings(fingerprint);
CREATE INDEX IF NOT EXISTS idx_model_executions_scan_id ON model_executions(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_files_scan_id ON scan_files(scan_id);
-- Note: idx_models_role removed - role column is deprecated, use roles_json instead
CREATE INDEX IF NOT EXISTS idx_models_enabled ON models(enabled);
