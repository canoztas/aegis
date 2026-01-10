# Model Management Refactor Guide

## Overview

This refactor implements a robust, extensible model management system for Aegis with:
- **Discovery-Registration Flow**: Discover available models → Register for use → Execute in scans
- **HuggingFace Support**: Two starter models with flexible output parsing
- **Ollama Auto-Discovery**: Automatic detection of local Ollama models
- **Parser Architecture**: Pluggable parsers for heterogeneous model outputs
- **Role-Based Runners**: Triage, DeepScan, Judge, Explain

## Architecture

```
┌──────────────────┐
│   DISCOVERY      │  GET /api/models/discovered/ollama
│   (Available)    │  Lists all local Ollama models
└────────┬─────────┘
         │
         ▼ POST /api/models/register
┌──────────────────┐
│  REGISTRATION    │  DB: model_id, type, roles, parser
│   (Configured)   │  ModelRegistryV2
└────────┬─────────┘
         │
         ▼ GET /api/models/registered
┌──────────────────┐
│   EXECUTION      │  Runner → Provider → Parser
│   (Scan Time)    │  Returns FindingCandidates
└──────────────────┘
```

## Quick Start

### 1. Apply Database Migration

```bash
python migrate_models_v2.py
```

This adds the required columns to the `models` table:
- `roles_json` - JSON array of roles
- `parser_id` - Parser identifier
- `model_type` - Model type enum
- `status` - Registration status

### 2. Install Optional Dependencies

For HuggingFace support:
```bash
pip install transformers torch
```

### 3. Start the Application

```bash
python -m aegis
```

### 4. Discover Ollama Models

```bash
curl http://localhost:5000/api/models/discovered/ollama
```

Response:
```json
{
  "models": [
    {
      "name": "qwen2.5-coder:7b",
      "model_type": "ollama_local",
      "provider": "ollama",
      "size_bytes": 4900000000,
      "metadata": {...}
    }
  ]
}
```

### 5. Register a Model

```bash
curl -X POST http://localhost:5000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "ollama:qwen2.5-coder",
    "model_type": "ollama_local",
    "provider_id": "ollama",
    "model_name": "qwen2.5-coder:7b",
    "display_name": "Qwen 2.5 Coder 7B",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "settings": {
      "temperature": 0.1,
      "max_tokens": 2048
    }
  }'
```

### 6. Register HuggingFace Presets

```bash
# Register CodeBERT for triage
curl -X POST http://localhost:5000/api/models/hf/register_preset \
  -H "Content-Type: application/json" \
  -d '{
    "preset_id": "codebert_insecure",
    "display_name": "CodeBERT Triage"
  }'

# Register CodeAstra for deep scan
curl -X POST http://localhost:5000/api/models/hf/register_preset \
  -H "Content-Type: application/json" \
  -d '{
    "preset_id": "codeastra_7b",
    "display_name": "CodeAstra Deep Scan"
  }'
```

### 7. List Registered Models

```bash
curl http://localhost:5000/api/models/registered
```

Response:
```json
{
  "models": [
    {
      "id": 1,
      "model_id": "ollama:qwen2.5-coder",
      "model_type": "ollama_local",
      "provider_id": "ollama",
      "model_name": "qwen2.5-coder:7b",
      "display_name": "Qwen 2.5 Coder 7B",
      "roles": ["deep_scan", "judge"],
      "parser_id": "json_schema",
      "status": "registered"
    },
    {
      "id": 2,
      "model_id": "hf:codebert_insecure",
      "model_type": "hf_local",
      "provider_id": "huggingface",
      "model_name": "mrm8488/codebert-base-finetuned-detect-insecure-code",
      "display_name": "CodeBERT Triage",
      "roles": ["triage"],
      "parser_id": "hf_classification",
      "status": "registered"
    }
  ]
}
```

## API Endpoints

### Discovery

#### `GET /api/models/discovered/ollama`
Discover available Ollama models.

**Query Parameters:**
- `refresh` (optional): `true` to force cache refresh

**Response:**
```json
{
  "models": [
    {
      "name": "qwen2.5-coder:7b",
      "model_type": "ollama_local",
      "provider": "ollama",
      "size_bytes": 4900000000,
      "metadata": {...}
    }
  ]
}
```

### Registration

#### `POST /api/models/register`
Register a model for use in scans.

**Request:**
```json
{
  "model_id": "ollama:qwen2.5-coder",
  "model_type": "ollama_local",
  "provider_id": "ollama",
  "model_name": "qwen2.5-coder:7b",
  "display_name": "Qwen 2.5 Coder 7B",
  "roles": ["deep_scan", "judge"],
  "parser_id": "json_schema",
  "settings": {
    "temperature": 0.1,
    "max_tokens": 2048
  }
}
```

**Response:**
```json
{
  "model": {
    "id": 1,
    "model_id": "ollama:qwen2.5-coder",
    ...
  }
}
```

#### `GET /api/models/registered`
List all registered models.

**Query Parameters:**
- `type`: Filter by model_type (ollama_local, hf_local, etc.)
- `role`: Filter by role (triage, deep_scan, judge, explain)
- `status`: Filter by status (registered, disabled, unavailable)

#### `DELETE /api/models/<model_id>`
Delete a registered model.

#### `PUT /api/models/<model_id>/status`
Update model status.

**Request:**
```json
{
  "status": "disabled"
}
```

### HuggingFace Presets

#### `GET /api/models/hf/presets`
List available HuggingFace model presets.

**Response:**
```json
{
  "presets": [
    {
      "model_id": "mrm8488/codebert-base-finetuned-detect-insecure-code",
      "task_type": "text-classification",
      "name": "CodeBERT Insecure Code Detector",
      "description": "Binary classifier for detecting potentially insecure code",
      "recommended_roles": ["triage"],
      "recommended_parser": "hf_classification"
    },
    {
      "model_id": "rootxhacker/CodeAstra-7B",
      "task_type": "text-generation",
      "name": "CodeAstra 7B",
      "description": "Generative model for code analysis",
      "recommended_roles": ["deep_scan"],
      "recommended_parser": "json_schema"
    }
  ]
}
```

#### `POST /api/models/hf/register_preset`
Register a HuggingFace preset model.

**Request:**
```json
{
  "preset_id": "codebert_insecure",
  "display_name": "My CodeBERT Triage"
}
```

## Model Types

### Ollama Local (`ollama_local`)
- **Discovery**: Automatic via Ollama API
- **Provider**: OllamaProvider
- **Parsers**: json_schema
- **Roles**: deep_scan, judge, explain
- **Example**: `qwen2.5-coder:7b`, `codellama:7b`

### HuggingFace Local (`hf_local`)
- **Discovery**: Config-based
- **Provider**: HFLocalProvider
- **Parsers**: json_schema, hf_classification
- **Roles**: triage (classification), deep_scan (generation)
- **Examples**: CodeBERT, CodeAstra

### OpenAI Compatible (`openai_compatible`) - Stub
- **Discovery**: Manual
- **Provider**: TBD
- **Parsers**: json_schema
- **Roles**: All
- **Example**: GPT-4, Claude

### Tool Plugin (`tool_plugin`) - Stub
- **Discovery**: Config-based
- **Provider**: Custom
- **Parsers**: Custom
- **Roles**: Custom
- **Example**: Bandit, Semgrep wrappers

## Parsers

### JSON Schema Parser (`json_schema`)
Parses JSON-formatted findings from generative models.

**Expected Format:**
```json
{
  "findings": [
    {
      "file_path": "test.py",
      "line_start": 10,
      "line_end": 12,
      "snippet": "code snippet",
      "category": "sql_injection",
      "severity": "high",
      "description": "Detailed explanation",
      "confidence": 0.95
    }
  ]
}
```

**Features:**
- Handles fenced code blocks (```json ... ```)
- Brace matching fallback
- Size validation (max 100KB)
- Context defaults (file_path, snippet)

### HF Text Classification Parser (`hf_classification`)
Parses HuggingFace text-classification outputs.

**Expected Format:**
```json
[
  {"label": "LABEL_1", "score": 0.95},
  {"label": "LABEL_0", "score": 0.05}
]
```

**Configuration:**
```yaml
parser_config:
  positive_labels: ["LABEL_1", "VULNERABLE", "INSECURE"]
  negative_labels: ["LABEL_0", "SAFE", "SECURE"]
  threshold: 0.5
  severity_high_threshold: 0.85
  severity_medium_threshold: 0.65
```

**Output:**
- `TriageSignal` with is_suspicious, confidence
- Optional `FindingCandidate` if suspicious + context provided
- Severity mapped based on confidence thresholds

## Runners

### Base Runner
Abstract interface for role-based execution:
1. Execute model via provider
2. Parse output via parser
3. Return structured results

### Triage Runner
- **Role**: `triage`
- **Models**: Classification models (CodeBERT)
- **Output**: TriageSignal + optional findings
- **Use**: Initial filtering, identifying suspicious code

### Deep Scan Runner
- **Role**: `deep_scan`
- **Models**: Generative models (CodeAstra, GPT-4, Qwen)
- **Output**: Detailed FindingCandidates
- **Use**: Thorough vulnerability analysis

## Example Pipeline Configurations

### Full Scan (Triage → Deep Scan → Judge)

```yaml
full_scan:
  name: "Full Security Scan"
  steps:
    - role: "triage"
      model_id: "hf:codebert_insecure"
      parser: "hf_classification"
      filter_threshold: 0.5

    - role: "deep_scan"
      model_id: "hf:codeastra_7b"
      parser: "json_schema"
      only_if_suspicious: true

    - role: "judge"
      model_id: "ollama:qwen2.5-coder"
      parser: "json_schema"
      purpose: "final_verdict"
```

### Triage Only (Fast Filtering)

```yaml
triage_only:
  name: "Quick Triage"
  steps:
    - role: "triage"
      parser: "hf_classification"
```

### Deep Scan Only (Comprehensive)

```yaml
deep_scan_only:
  name: "Deep Scan Only"
  steps:
    - role: "deep_scan"
      parser: "json_schema"
```

## Adding New Models

### Adding a New Ollama Model

1. Ensure model is pulled locally:
```bash
ollama pull <model-name>
```

2. Discover:
```bash
curl http://localhost:5000/api/models/discovered/ollama
```

3. Register:
```bash
curl -X POST http://localhost:5000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "ollama:<model-name>",
    "model_type": "ollama_local",
    "provider_id": "ollama",
    "model_name": "<model-name>",
    "display_name": "<Display Name>",
    "roles": ["deep_scan"],
    "parser_id": "json_schema"
  }'
```

### Adding a New HuggingFace Model

1. Add to `aegis/config/models.yaml`:
```yaml
- model_id: "hf:my_custom_model"
  hf_model_id: "organization/model-name"
  task_type: "text-classification"
  display_name: "My Custom Model"
  roles: ["triage"]
  parser: "hf_classification"
  parser_config:
    positive_labels: ["VULNERABLE"]
    threshold: 0.6
```

2. Register via API:
```bash
curl -X POST http://localhost:5000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "hf:my_custom_model",
    "model_type": "hf_local",
    "provider_id": "huggingface",
    "model_name": "organization/model-name",
    "display_name": "My Custom Model",
    "roles": ["triage"],
    "parser_id": "hf_classification",
    "settings": {
      "task_type": "text-classification"
    }
  }'
```

### Creating a Custom Parser

1. Create parser class:
```python
# aegis/models/parsers/my_parser.py
from aegis.models.parsers.base import BaseParser
from aegis.models.schema import ParserResult, FindingCandidate

class MyCustomParser(BaseParser):
    def parse(self, raw_output, context=None):
        # Parse logic
        findings = []
        # ... extract findings from raw_output
        return ParserResult(findings=findings)
```

2. Register parser in config:
```yaml
parsers:
  - id: "my_parser"
    class: "aegis.models.parsers.my_parser.MyCustomParser"
    description: "Custom parser for specific output format"
    config: {}
```

3. Use in model registration:
```json
{
  "parser_id": "my_parser"
}
```

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Discovery tests
pytest tests/test_model_discovery.py -v

# Parser tests
pytest tests/test_model_parsers.py -v
```

### Test Ollama Discovery (Mock)

```bash
pytest tests/test_model_discovery.py::test_ollama_discovery_sync -v
```

### Test Parsers

```bash
pytest tests/test_model_parsers.py::TestJSONSchemaParser -v
pytest tests/test_model_parsers.py::TestHFTextClassificationParser -v
```

## Troubleshooting

### HuggingFace Models Not Available

**Error**: `transformers library not available`

**Solution**:
```bash
pip install transformers torch
```

### Ollama Discovery Fails

**Error**: `Connection refused`

**Solution**:
1. Ensure Ollama is running:
```bash
ollama serve
```

2. Check Ollama is accessible:
```bash
curl http://localhost:11434/api/tags
```

### Model Not Loading

**Error**: Model registered but not appearing in scans

**Solutions**:
1. Check status:
```bash
curl http://localhost:5000/api/models/registered?status=registered
```

2. Verify roles:
```bash
curl "http://localhost:5000/api/models/registered?role=deep_scan"
```

3. Check database:
```bash
sqlite3 data/aegis.db "SELECT * FROM models WHERE model_id='<model-id>'"
```

### Parser Errors

**Error**: `No valid JSON found in output`

**Solutions**:
1. Enable debug logging to see raw output
2. Adjust prompt template to enforce JSON output
3. Create custom parser for specific format

## File Structure

```
aegis/
├── models/
│   ├── __init__.py
│   ├── schema.py                   # Core data models
│   ├── registry.py                 # ModelRegistryV2
│   ├── discovery/
│   │   ├── __init__.py
│   │   └── ollama.py              # Ollama discovery client
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseParser interface
│   │   ├── json_schema.py         # JSON parser
│   │   └── hf_classification.py   # HF classification parser
│   └── runners/
│       ├── __init__.py
│       ├── base.py                # BaseRunner interface
│       ├── triage.py              # Triage runner
│       └── deep_scan.py           # DeepScan runner
├── providers/
│   └── hf_local.py                # HuggingFace provider
├── api/
│   └── routes_models.py           # Model API routes
├── config/
│   └── models.yaml                # Model configurations
└── database/
    └── migrations/
        └── 003_model_registry_v2.sql  # DB migration
```

## Next Steps

1. **Implement Runner Integration**: Connect runners to scan pipeline
2. **Add More Parsers**: XML, SARIF, custom formats
3. **Cloud Providers**: OpenAI, Anthropic integration
4. **Tool Plugins**: Bandit, Semgrep wrappers
5. **Model Versioning**: Track model versions and updates
6. **Performance Metrics**: Track latency, accuracy per model
7. **Model Ensembles**: Combine multiple models for better results

## Migration Notes

- ✅ Backward compatible with existing scans
- ✅ Old `role` column preserved
- ✅ New code uses `roles_json`
- ✅ Existing models continue to work
- ✅ Database migration is additive only

## Support

For issues or questions:
1. Check [MODEL_REFACTOR_GUIDE.md](MODEL_REFACTOR_GUIDE.md)
2. Review API responses for error messages
3. Check logs for detailed error traces
4. Test with curl commands from this guide
