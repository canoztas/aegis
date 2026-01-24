# Aegis API Reference

Complete API documentation for Aegis SAST Framework.

---

## Base URL

```
http://localhost:5000/api
```

---

## Authentication

Currently, Aegis does not require authentication. Cloud provider API keys are configured via environment variables.

---

## Scans

### Upload Files for Scan

```http
POST /api/scans/upload
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| files | file[] | Yes | Files to scan (multiple allowed) |
| model_ids | string[] | Yes | Model IDs to use |
| consensus_strategy | string | No | `union`, `majority`, `judge` (default: `union`) |
| judge_model_id | string | No | Required if strategy is `judge` |

**Example:**
```bash
curl -X POST http://localhost:5000/api/scans/upload \
  -F "files=@app.py" \
  -F "files=@utils.py" \
  -F "model_ids=ollama:llama3.2" \
  -F "model_ids=hf:codebert-insecure" \
  -F "consensus_strategy=union"
```

**Response:**
```json
{
  "scan_id": "abc123",
  "status": "queued",
  "message": "Scan queued successfully"
}
```

### Submit Inline Code

```http
POST /api/scans/submit
Content-Type: application/json
```

**Body:**
```json
{
  "code": "eval(user_input)",
  "language": "python",
  "model_ids": ["openai:gpt-4o-mini"],
  "consensus_strategy": "union"
}
```

### Get Scan Status

```http
GET /api/scans/{scan_id}/status
```

**Response:**
```json
{
  "scan_id": "abc123",
  "status": "completed",
  "progress": 100,
  "findings_count": 5,
  "models_completed": 2,
  "models_total": 2
}
```

### Get Scan Results

```http
GET /api/scans/{scan_id}
```

### Export Results

```http
GET /api/scans/{scan_id}/export/{format}
```

**Formats:** `sarif`, `csv`, `json`

**Example:**
```bash
curl http://localhost:5000/api/scans/abc123/export/sarif -o results.sarif
```

### Stream Scan Progress (SSE)

```http
GET /api/scans/{scan_id}/stream
```

Returns Server-Sent Events for real-time progress updates.

---

## Models

### List Registered Models

```http
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "ollama:llama3.2",
      "display_name": "Llama 3.2",
      "model_type": "ollama",
      "roles": ["deep_scan"],
      "status": "ready"
    }
  ]
}
```

### Get Model Catalog

```http
GET /api/models/catalog
```

Returns all available models from the catalog (pre-configured models).

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| category | string | Filter by category: `huggingface`, `cloud`, `ollama`, `classic_ml` |

### Install Model from Catalog

```http
POST /api/models/catalog/{catalog_id}/install
```

Downloads and registers a model from the catalog.

### Download Model Files

```http
POST /api/models/catalog/{catalog_id}/download
```

Downloads model files without registering.

### Register Custom Model

```http
POST /api/models/registry
Content-Type: application/json
```

**Body:**
```json
{
  "model_type": "hf_local",
  "provider_id": "huggingface",
  "model_name": "microsoft/codebert-base",
  "display_name": "CodeBERT Base",
  "roles": ["triage"],
  "parser_id": "hf_classification",
  "settings": {
    "task_type": "text-classification"
  }
}
```

### Update Model

```http
PUT /api/models/{model_id}
Content-Type: application/json
```

### Delete Model

```http
DELETE /api/models/{model_id}
```

### Test Model

```http
POST /api/models/{model_id}/test
Content-Type: application/json
```

**Body:**
```json
{
  "code": "eval(input())",
  "language": "python"
}
```

---

## Ollama

### Discover Ollama Models

```http
GET /api/models/ollama/discover
```

Returns models available in local Ollama installation.

### Pull Ollama Model

```http
POST /api/models/ollama/pull
Content-Type: application/json
```

**Body:**
```json
{
  "model_name": "llama3.2"
}
```

### Check Ollama Status

```http
GET /api/models/ollama/status
```

---

## HuggingFace

### List HF Presets

```http
GET /api/models/hf/presets
```

### Register HF Preset

```http
POST /api/models/hf/register_preset
Content-Type: application/json
```

**Body:**
```json
{
  "preset_id": "codebert_insecure",
  "display_name": "Custom Name",
  "download": true
}
```

---

## ML Models

### List ML Presets

```http
GET /api/models/ml/presets
```

### Register ML Preset

```http
POST /api/models/ml/register_preset
Content-Type: application/json
```

**Body:**
```json
{
  "preset_id": "kaggle_rf_cfunctions",
  "download": true
}
```

---

## Findings

### List Findings

```http
GET /api/findings
```

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| scan_id | string | Filter by scan |
| severity | string | Filter by severity: `critical`, `high`, `medium`, `low` |
| status | string | Filter by status: `open`, `resolved`, `false_positive` |

### Get Finding

```http
GET /api/findings/{finding_id}
```

### Update Finding Status

```http
PATCH /api/findings/{finding_id}
Content-Type: application/json
```

**Body:**
```json
{
  "status": "false_positive",
  "notes": "This is expected behavior"
}
```

---

## History

### List Scan History

```http
GET /api/history
```

**Query Parameters:**
| Name | Type | Description |
|------|------|-------------|
| limit | int | Number of results (default: 50) |
| offset | int | Pagination offset |

### Clear History

```http
DELETE /api/history
```

---

## Health

### System Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "ollama": "available",
  "version": "2.0.0"
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message here",
  "details": "Additional details if available"
}
```

**HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limiting

Cloud provider APIs have built-in rate limiting:
- **OpenAI**: 5 requests/second
- **Anthropic**: 5 requests/second
- **Google**: 10 requests/second

Local models (Ollama, HuggingFace, sklearn) have no rate limits.
