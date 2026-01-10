# HuggingFace Integration - Implementation Summary

## Overview

Implemented a fully configurable, flexible architecture for integrating HuggingFace models into Aegis vulnerability scanning with a **saved model mechanism** that allows models to be added, configured, and managed entirely through the UI.

## What Was Built

### 1. Core Architecture

#### **Connector System** (`aegis/huggingface/connector.py`)
- `HuggingFaceConnector`: Base connector for loading and running HF models
- `PeftModelConnector`: Specialized connector for PEFT/LoRA models (like CodeAstra-7B)
- Supports:
  - Transformers pipeline API and direct inference
  - Quantization (4-bit, 8-bit) for memory efficiency
  - Device management (CPU, CUDA, MPS, auto)
  - Batch prediction
  - Model unloading

#### **Output Mapper System** (`aegis/huggingface/mappers.py`)
Five flexible mapper types to handle different model outputs:

1. **ClassifierMapper**: Binary/multi-class classifiers (e.g., CodeBERT)
   - Maps logits/probabilities to findings
   - Configurable thresholds and class labels

2. **GenerativeMapper**: Text generation models (e.g., CodeAstra-7B)
   - LLM-structured parsing with keyword matching
   - Regex-based extraction
   - Severity and CWE detection from generated text

3. **JsonMapper**: Models that output structured JSON
   - Schema-based field mapping
   - Configurable findings key and field names

4. **RegexMapper**: Custom regex pattern matching
   - Multiple pattern support
   - Named group extraction

5. **CustomMapper**: User-provided Python code
   - Execute custom parsing logic
   - Restricted environment for security

#### **Adapter** (`aegis/huggingface/adapter.py`)
- `HuggingFaceAdapter`: Integrates HF models into Aegis pipeline
- Features:
  - Lazy loading (models load on first use)
  - Automatic code chunking with overlap
  - Language filtering
  - Pre-configured factory functions for CodeBERT and CodeAstra-7B

### 2. Database Layer

#### **Schema** (`aegis/database/schema.sql`)
New `huggingface_models` table with:
- Model identification (model_id, display_name, hf_model_id)
- HuggingFace configuration (task, device, quantization)
- Mapper configuration (type, config JSON)
- Processing settings (chunk_size, overlap, supported_languages)
- Metadata (enabled, is_builtin, timestamps)

#### **Repository** (`aegis/database/hf_repository.py`)
- `HuggingFaceModelRepository`: Full CRUD operations
- Methods:
  - `create()`: Add new model configuration
  - `get_by_id()`: Retrieve model config
  - `list_all()`: List all models (with enabled filter)
  - `update()`: Update model settings
  - `delete()`: Remove model config
  - `toggle_enabled()`: Enable/disable models
  - `seed_builtin_models()`: Pre-populate CodeBERT and CodeAstra-7B
  - `get_adapter_config()`: Get config for adapter instantiation

### 3. API Layer

#### **Endpoints** (`aegis/routes.py`)
Complete REST API for model management:

```
GET    /api/huggingface/models              # List all models
GET    /api/huggingface/models/<model_id>   # Get specific model
POST   /api/huggingface/models              # Create new model
PUT    /api/huggingface/models/<model_id>   # Update model
DELETE /api/huggingface/models/<model_id>   # Delete model
POST   /api/huggingface/models/<model_id>/toggle  # Toggle enabled
POST   /api/huggingface/seed                # Seed built-in models
```

Features:
- Protected built-in models (can't be deleted)
- JSON validation
- Error handling

### 4. User Interface

#### **Management Page** (`aegis/templates/huggingface.html`)
Full-featured UI for model management:

**Features**:
- Model cards with status indicators (enabled/disabled, built-in badge)
- View full model configuration (JSON viewer)
- Enable/disable toggle switches
- Delete custom models (built-in protected)
- "Seed Built-in Models" button
- "Add Model" form with comprehensive fields

**Add Model Form**:
- Basic Info: Model ID, Display Name, HF Model ID, Description
- Configuration: Task Type, Connector Type, Device, Quantization
- Mapper: Mapper Type selection with auto-populated templates
- Processing: Chunk Size, Overlap, Version
- Languages: Comma-separated supported languages
- Status: Enable/disable checkbox

**Mapper Templates**:
Auto-populated JSON templates for each mapper type:
- Classifier: threshold, positive_class, labels
- Generative: severity_keywords, cwe_keywords
- JSON: findings_key, schema mapping
- Regex: pattern list
- Custom: Python code template

### 5. Documentation

#### **README** (`aegis/huggingface/README.md`)
Comprehensive documentation including:
- Architecture overview
- Usage examples (UI and API)
- 5 complete example configurations
- Mapper type explanations
- Code chunking explanation
- Programmatic usage examples
- Database schema
- Performance considerations
- Troubleshooting guide
- Best practices

## Pre-Configured Models

### 1. CodeBERT Insecure Code Detector
- **Model**: `mrm8488/codebert-base-finetuned-detect-insecure-code`
- **Type**: Binary classifier (secure vs insecure)
- **Accuracy**: 65.30% on test set
- **Usage**: Fast, lightweight scanning
- **Chunk Size**: 50 lines (small context for quick scans)

### 2. CodeAstra-7B Vulnerability Scanner
- **Model**: `rootxhacker/CodeAstra-7B`
- **Type**: LLM with PEFT/LoRA adapters
- **Accuracy**: 83% (vs 88.78% for GPT-4o)
- **Languages**: Python, Go, C, C++, Fortran, Ruby, Java, Kotlin, C#, PHP, Swift, JS, TS
- **Usage**: Deep analysis with detailed findings
- **Chunk Size**: 100 lines (larger context for LLM)
- **Quantization**: 4-bit by default (for memory efficiency)

## How It Works

### Adding a New Model (UI Flow)

1. User navigates to `/huggingface`
2. Clicks "Add Model"
3. Fills in form:
   - Identifies model on HuggingFace Hub
   - Selects task type (classification, generation)
   - Chooses mapper type (how to parse outputs)
   - Configures mapper (JSON template auto-populated)
   - Sets processing parameters
4. Clicks "Save Model"
5. Configuration saved to database
6. Model appears in list, can be toggled on/off

### Runtime Flow

1. User starts a scan (existing scan flow)
2. Aegis loads enabled HuggingFace models from database
3. For each enabled model:
   - Creates `HuggingFaceAdapter` from saved config
   - Adapter lazy-loads model on first use
   - Code is chunked based on chunk_size/overlap settings
   - Each chunk is passed through model
   - Mapper converts outputs to Aegis findings
4. Findings aggregated with other scan results
5. Models can be unloaded to free memory

### Output Mapping Example

**CodeBERT Output**:
```python
logits: [[-1.2, 2.5]]  # [secure_logit, insecure_logit]
```

**ClassifierMapper Processes**:
```python
probabilities = softmax(logits)  # [0.02, 0.98]
predicted_class = argmax(probabilities)  # 1 (insecure)
confidence = probabilities[1]  # 0.98
```

**Result**:
```python
{
  "name": "Insecure Code Detected",
  "severity": "high",
  "cwe": "CWE-1035",
  "file": "vulnerable.py",
  "start_line": 10,
  "end_line": 15,
  "message": "This code segment contains potential security vulnerabilities",
  "confidence": 0.98
}
```

## Key Features

### ✅ Fully Configurable
- Every aspect configurable through UI or API
- No code changes needed to add models
- JSON-based configuration

### ✅ Flexible Output Parsing
- 5 mapper types handle diverse model outputs
- Custom mappers for unique formats
- Extensible architecture

### ✅ Saved Model Mechanism
- Database-persisted configurations
- Models survive server restarts
- Version control friendly (configs in DB, not code)

### ✅ Memory Efficient
- Lazy loading (load on first use)
- Quantization support (4-bit, 8-bit)
- Explicit unload functionality

### ✅ Production Ready
- Error handling throughout
- Protected built-in models
- Enable/disable without deletion
- Comprehensive logging

## Files Created

```
aegis/
├── huggingface/
│   ├── __init__.py              # Module exports
│   ├── connector.py             # HF model loading and inference
│   ├── mappers.py               # Output mapping system (5 types)
│   ├── adapter.py               # Aegis integration adapter
│   └── README.md                # Comprehensive documentation
│
├── database/
│   ├── schema.sql               # Added huggingface_models table
│   └── hf_repository.py         # CRUD operations
│
├── templates/
│   └── huggingface.html         # Management UI
│
└── routes.py                    # Added /huggingface route + 7 API endpoints
```

## Technical Highlights

### 1. Chunking Strategy
```python
# Problem: Models have token limits (512-2048 tokens)
# Solution: Split code into overlapping chunks

chunk_size = 100 lines
overlap = 20 lines

File (300 lines):
  Chunk 1: Lines 1-100
  Chunk 2: Lines 81-180  (20 line overlap)
  Chunk 3: Lines 161-260
  Chunk 4: Lines 241-300

# Overlap prevents missing vulnerabilities at boundaries
```

### 2. Mapper Registry Pattern
```python
MAPPER_TYPES = {
    "classifier": ClassifierMapper,
    "generative": GenerativeMapper,
    "json": JsonMapper,
    "regex": RegexMapper,
    "custom": CustomMapper,
}

def create_mapper(mapper_type: str, config: Dict) -> OutputMapper:
    mapper_class = MAPPER_TYPES.get(mapper_type)
    return mapper_class(config)
```

### 3. Lazy Loading
```python
class HuggingFaceAdapter:
    def _ensure_loaded(self):
        if self._loaded:
            return

        # Load connector
        self.connector = create_connector(...)
        self.connector.load()  # Downloads/loads model

        # Create mapper
        self.mapper = create_mapper(...)

        self._loaded = True
```

## Performance Characteristics

| Model | Size | Load Time | Memory (RAM) | Memory (4-bit) | Speed |
|-------|------|-----------|--------------|----------------|-------|
| CodeBERT | 125M | 5-10s | ~500MB | N/A | Fast (CPU OK) |
| CodeAstra-7B | 7B | 30-60s | ~14GB | ~4GB | Slow (GPU recommended) |

## Security Considerations

1. **Custom Mapper Execution**:
   - Runs in restricted environment
   - Limited imports allowed
   - No file system access
   - Should only be used by admins

2. **Model Trust**:
   - Models execute arbitrary computations
   - Only use trusted HuggingFace models
   - Review model cards before adding

3. **Built-in Protection**:
   - Built-in models can't be deleted
   - Can only be disabled

## Future Enhancements

Possible extensions:
- Model performance metrics (track accuracy over time)
- Automatic mapper generation from examples
- Model marketplace/sharing
- Fine-tuning interface
- Ensemble predictions across models
- Cost tracking (for API-based models)

## Usage Example

```python
# Via UI
1. Navigate to /huggingface
2. Click "Seed Built-in Models"
3. Toggle models on/off as needed
4. Run normal scan - HF models automatically included

# Via API
import requests

# Add custom model
requests.post('http://localhost:5000/api/huggingface/models', json={
    "model_id": "hf_my_scanner",
    "display_name": "My Custom Scanner",
    "hf_model_id": "username/model-name",
    "hf_task": "text-classification",
    "mapper_type": "classifier",
    "mapper_config": {...}
})

# Enable model
requests.post('http://localhost:5000/api/huggingface/models/hf_my_scanner/toggle')

# Run scan (existing flow)
```

## Summary

This implementation provides a **complete, production-ready HuggingFace integration** with:

✅ **Flexible Architecture**: 5 mapper types, multiple model types
✅ **Saved Model Mechanism**: Database-persisted configs
✅ **Full CRUD**: Add, edit, delete, toggle models
✅ **User-Friendly UI**: Visual model management
✅ **Pre-Configured Models**: CodeBERT and CodeAstra-7B ready to use
✅ **Well-Documented**: Comprehensive README with examples
✅ **Production-Ready**: Error handling, security, memory management

The system solves the core challenge: **"every model will have different output parsing"** by providing a pluggable mapper architecture that can be configured entirely through JSON, without code changes.
