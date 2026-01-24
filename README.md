<div align="center">
  <img src="aegis/static/img/aegis-logo.svg" alt="aegis logo" width="200"/>
  <h1>Aegis - AI-Powered SAST Framework</h1>
  <p>Multi-model security scanning with LLMs, ML, AI (Local & Cloud)</p>
</div>

---

## Demos

[![Demo Video (Ollama)](https://img.youtube.com/vi/YsyRN238z_A/0.jpg)](https://youtu.be/YsyRN238z_A)

- **Aegis v1**: https://youtu.be/ZmT-3UpVOz8
- **Legacy version**: https://youtu.be/StXTwdxQyQI

---

## About

Aegis is an open-source SAST framework for multi-model orchestration, bridging generative and discriminative AI for security analysis. It provides standardized infrastructure to integrate, benchmark, and compare diverse architectures—from Large Language Models (OpenAI, Anthropic, Google, Ollama) to specialized classifiers (CodeBERT, VulBERTa) via HuggingFace, and traditional ML models (sklearn).

Every model is a pluggable component within a unified registry, enabling complex workflows such as consensus-based decision making, multi-layered scanning, and cross-architecture validation.

**Key Features:**
- **Multi-Provider Support**: Ollama, HuggingFace, OpenAI, Anthropic, Google, sklearn
- **Consensus Strategies**: Union, Majority, Judge, Cascade (coming soon), Weighted (coming soon)
- **Model Catalog**: Pre-configured models with one-click installation
- **Real-time Progress**: SSE streaming for live scan updates
- **Export Formats**: SARIF, CSV, JSON
- **Extensible**: Add custom models, parsers, and providers

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         User Upload                          │
│                    (ZIP, Code, Git Repo)                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Scan Worker (Queue)                       │
│             Background processing with SSE updates           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Pipeline Executor (Chunks)                  │
│           Parallel execution with ThreadPoolExecutor         │
└────────┬────────────────────────────────────────────┬────────┘
         │                                            │
         ▼                                            ▼
┌────────────────────┐                    ┌────────────────────┐
│   Model Registry   │                    │   Prompt Builder   │
│  (SQLite + Cache)  │                    │  (Role Templates)  │
└────────┬───────────┘                    └──────────┬─────────┘
         │                                           │
         ▼                                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   Provider Layer (Adapters)                  │
│      Ollama  │  HuggingFace  │  Cloud  │  Classic ML         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Parsers (JSON/Binary)                     │
│         Schema validation + fallback regex extraction        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Consensus Engine (Merge)                    │
│      Union │ Majority │ Judge │ Cascade │ Weighted           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Database (Scans, Findings, History)             │
│                Export: SARIF, CSV, JSON                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements/requirements.txt
```

**Installation Options:**
- **Standard** (CPU): `pip install -r requirements/requirements.txt`
- **GPU** (CUDA): `pip install -r requirements/requirements-gpu.txt`

### 2. Configure API Keys (Optional)

Create `.env` file for cloud providers:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### 3. Start Aegis

```bash
python app.py
```

Open: **http://localhost:5000**

---

## Consensus Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Union** | Combines all findings from all models | Maximum coverage |
| **Majority** | Only findings detected by 2+ models | Reducing false positives |
| **Judge** | A judge model evaluates and merges findings | High-precision analysis |
| **Cascade** *(coming soon)* | Two-pass: fast triage → deep scan on flagged files | Large codebases |
| **Weighted** *(coming soon)* | Confidence-weighted voting | Balanced precision/recall |

---

## Using Models

### Ollama (Local LLMs)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. In Aegis UI → **Models** → **OLLAMA** → Click **Discover**
4. Click **Register** on detected models

**Recommended**: `llama3.2`, `qwen2.5-coder:7b`, `codellama:7b`

### HuggingFace (Local Transformers)

1. In Aegis UI → **Models** → **HUGGING FACE**
2. Browse the **Model Catalog** for pre-configured models
3. Click **Install** to download and register

**Available Models**:
- **CodeBERT Insecure Classifier** - Fast triage classifier
- **VulBERTa** - C/C++ vulnerability detector
- **UnixCoder** - Multi-language vulnerability classifier
- **Qwen 2.5 Coder** - Code-specialized LLM

### Cloud LLMs (OpenAI, Anthropic, Google)

1. In Aegis UI → **Models** → **CLOUD**
2. Select provider and model from the catalog
3. Click **Add** (requires API key in `.env`)

**Supported**: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash, and more

### Classic ML Models (sklearn)

1. In Aegis UI → **Models** → **ML MODELS**
2. Click **Install** on available presets
3. Model downloads automatically

**Available**: Kaggle RF C-Functions Predictor (Random Forest for C/C++ vulnerability detection)

---

## Extending Aegis

### Adding HuggingFace Models

Add new HuggingFace models to the catalog in `aegis/models/catalog.py`:

```python
HF_MY_MODEL = {
    "catalog_id": "my_model",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "My Custom Model",
    "description": "Description of what this model does",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "organization/model-name",  # HuggingFace model ID
    "task_type": "text-classification",  # or "text-generation"
    "roles": ["triage"],  # triage, deep_scan, judge
    "parser_id": "hf_classification",  # Parser to use
    "parser_config": {
        "positive_labels": ["LABEL_1", "vulnerable"],
        "negative_labels": ["LABEL_0", "safe"],
        "threshold": 0.5,
    },
    "hf_kwargs": {
        "trust_remote_code": False,
    },
    "generation_kwargs": {
        "max_new_tokens": 512,
        "temperature": 0.1,
    },
    "size_mb": 500,
    "requires_gpu": False,
    "tags": ["classifier", "triage"],
}

# Add to MODEL_CATALOG list
MODEL_CATALOG.append(HF_MY_MODEL)
```

### Adding Cloud Models

Add cloud model presets in `aegis/models/catalog.py`:

```python
CLOUD_MY_MODEL = {
    "catalog_id": "my_cloud_model",
    "category": CatalogCategory.CLOUD,
    "display_name": "My Cloud Model",
    "description": "Cloud LLM for security analysis",
    "provider_id": "openai",  # openai, anthropic, google
    "model_type": "openai_cloud",  # openai_cloud, anthropic_cloud, google_cloud
    "model_name": "gpt-4o-mini",  # Provider's model name
    "task_type": "text-generation",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "generation_kwargs": {
        "max_tokens": 2048,
        "temperature": 0.1,
    },
    "requires_api_key": True,
    "tags": ["cloud", "llm", "deep_scan"],
}
```

### Adding ML Models (sklearn)

1. **Train and save your model as a Pipeline**:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', RandomForestClassifier(n_estimators=100))
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'my_model.pkl')
```

2. **Host the model** (e.g., GitHub Releases)

3. **Add to catalog** in `aegis/models/catalog.py`:

```python
ML_MY_MODEL = {
    "catalog_id": "my_ml_model",
    "category": CatalogCategory.CLASSIC_ML,
    "display_name": "My ML Classifier",
    "description": "Custom sklearn model for vulnerability detection",
    "provider_id": "tool_ml",
    "model_type": "tool_ml",
    "model_name": "my_ml_model",
    "tool_id": "sklearn_classifier",
    "task_type": "classification",
    "roles": ["triage"],
    "parser_id": "tool_result",
    "settings": {
        "model_url": "https://github.com/user/repo/releases/download/v1/my_model.pkl",
        "threshold": 0.5,
    },
    "size_mb": 50,
    "requires_gpu": False,
    "tags": ["ml", "sklearn", "triage"],
}
```

### Adding Custom Parsers

Create a new parser in `aegis/parsers/`:

```python
# aegis/parsers/my_parser.py
from aegis.parsers.base import BaseParser
from aegis.models.schema import ParserResult, FindingCandidate

class MyParser(BaseParser):
    """Custom parser for specific model output format."""

    parser_id = "my_parser"

    def parse(self, raw_output: str, context: dict) -> ParserResult:
        """Parse model output into findings."""
        findings = []

        # Your parsing logic here
        # Extract vulnerabilities from raw_output

        for vuln in extracted_vulns:
            findings.append(FindingCandidate(
                file_path=context.get("file_path", "unknown"),
                line_start=vuln.get("line", 1),
                line_end=vuln.get("line", 1),
                snippet=vuln.get("code", ""),
                title=vuln.get("title", "Vulnerability"),
                category=vuln.get("category", "security"),
                cwe=vuln.get("cwe"),
                severity=vuln.get("severity", "medium"),
                description=vuln.get("description", ""),
                recommendation=vuln.get("fix", ""),
                confidence=vuln.get("confidence", 0.5),
            ))

        return ParserResult(
            findings=findings,
            raw_output=raw_output,
        )
```

Register the parser in `aegis/parsers/__init__.py`:

```python
from aegis.parsers.my_parser import MyParser

PARSER_REGISTRY = {
    # ... existing parsers
    "my_parser": MyParser,
}
```

---

## Project Structure

```
aegis/
├── api/                  # REST API routes
├── consensus/            # Multi-model merging strategies
├── database/             # SQLite repositories
├── models/               # Model registry and catalog
│   ├── catalog.py       # Pre-configured model definitions
│   ├── registry.py      # Model registration (database)
│   └── runtime.py       # Model loading + caching
├── parsers/              # Output parsers (JSON, classification, etc.)
├── providers/            # Provider adapters (Ollama, HF, Cloud)
├── services/             # Scan worker + background queue
├── tools/                # ML tool plugins (sklearn, regex)
├── static/               # Web UI assets
└── templates/            # Jinja2 HTML templates

config/
├── models.yaml           # Legacy model presets
└── pipelines/            # Pipeline definitions

data/aegis.db             # SQLite database (auto-created)
```

---

## Troubleshooting

### Model Loading Issues

**HuggingFace models fail to load**:
- Ensure 8GB+ RAM (16GB recommended)
- Use 4-bit quantization in model settings
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Ollama models not detected**:
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Set custom URL: `OLLAMA_BASE_URL=http://host:11434`

**sklearn model errors**:
- Ensure model is saved as a Pipeline (includes vectorizer)
- Check sklearn version compatibility

### Scan Issues

**Scans stuck at "Running"**:
- Check browser console for SSE errors
- First scan may take 1-2 minutes (model loading)
- Check `logs/aegis.log`

**Cloud API rate limits**:
- Use Ollama/HF for high-throughput scanning
- Increase delay between requests

---

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for detailed endpoint documentation.

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Contributing

Pull requests welcome! Please:
- Follow existing code style (PEP 8)
- Add type hints to new functions
- Test with multiple providers
