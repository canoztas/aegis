<div align="center">
  <img src="aegis/static/img/aegis-logo.svg" alt="aegis logo" width="200"/>
  <h1>Aegis - AI-Powered SAST Framework</h1>
  <p>Multi-model security scanning with LLMs, ML, AI (Local & Cloud) </p>
</div>

---

## Demos

[![Demo Video (Ollama)](https://img.youtube.com/vi/YsyRN238z_A/0.jpg)](https://youtu.be/YsyRN238z_A)

- **Aegis v1 (of the vulnerable)**: https://youtu.be/ZmT-3UpVOz8
- Legacy version: https://youtu.be/StXTwdxQyQI
---

## About

This open-source framework is the practical culmination of theoretical research into multi-model orchestration, designed to bridge the gap between generative and discriminative AI. Rather than being limited to a single model family, it provides a standardized infrastructure to integrate, benchmark, and compare a diverse range of architectures—from Large Language Models (OpenAI, Anthropic, Gemini, Ollama) to specialized NLP classifiers like BERT and RoBERTa via HuggingFace. By treating every model as a pluggable component within a unified registry, the framework enables complex workflows such as consensus-based decision making, multi-layered scanning, and cross-architecture validation. It features a non-blocking background processing engine, real-time status updates via SSE, and native support for runtime optimizations like 4-bit/8-bit quantization.

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
│          Ollama  │  HuggingFace  │  Cloud (OpenAI, etc.)     │
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
│          Union │ Majority │ Weighted │ Judge Model           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Database (Scans, Findings, History)             │
│                Export: SARIF, CSV, JSON                      │
└──────────────────────────────────────────────────────────────┘
```



## Consensus Strategies

Aegis can combine results from multiple models using different strategies:

### Union (Default)
Combines all findings from all models. Best for maximum coverage.

### Majority
Only includes findings detected by 2 or more models. Reduces false positives.

### Judge
Uses a dedicated "judge" model to evaluate and merge findings from other models.

### Weighted (TODO)
Filters findings based on confidence scores. Configurable threshold.

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
- **Minimal** (cloud only, ~100MB): `pip install -r requirements/requirements-minimal.txt`
- **Standard** (CPU, ~3.5GB): `pip install -r requirements/requirements.txt`
- **GPU** (CUDA 11.8, ~5GB): `pip install -r requirements/requirements-gpu.txt`

See [requirements/REQUIREMENTS.md](requirements/REQUIREMENTS.md) for details.

### 2. Start Aegis

```bash
python app.py
```

Database auto-initializes on first run. Open: **http://localhost:5000**

That's it! No manual setup required.

---

## Using Models

### Ollama (Local)

**Setup:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Aegis UI → **Models** → **DISCOVER OLLAMA**
4. Click **Register** on detected models
5. Run your first scan!

**Recommended Models:**
- `llama3.2:latest` - Fast, general-purpose
- `qwen2.5-coder:7b` - Code-specialized
- `codellama:7b` - Security-focused

### HuggingFace (Local)

**Quick Start (UI):**
1. Aegis UI → **Models** → **HUGGING FACE**
2. Click **Register** on CodeBERT or CodeAstra
3. Models download automatically on first scan

**Add Custom Models:**

Via API:
```bash
curl -X POST http://localhost:5000/api/models/registry \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "hf_local",
    "provider_id": "huggingface",
    "model_name": "deepseek-ai/deepseek-coder-6.7b-base",
    "display_name": "DeepSeek Coder 6.7B",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "settings": {
      "task_type": "text-generation",
      "runtime": {
        "device_preference": ["cuda", "cpu"],
        "dtype": "bf16",
        "quantization": "4bit",
        "max_concurrency": 1
      },
      "generation_kwargs": {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9
      }
    }
  }'
```

Via YAML (`config/models.yaml`):
```yaml
models:
  - model_id: "hf:deepseek_coder"
    model_type: "hf_local"
    provider_id: "huggingface"
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    display_name: "DeepSeek Coder 6.7B"
    roles: [deep_scan]
    parser_id: "json_schema"
    settings:
      task_type: "text-generation"
      runtime:
        device_preference: ["cuda", "cpu"]
        dtype: "bf16"
        quantization: "4bit"
      generation_kwargs:
        max_new_tokens: 512
        temperature: 0.2
```

**Recommended Models:**

| Model | HuggingFace ID | Best For | Memory |
|-------|----------------|----------|--------|
| DeepSeek Coder | `deepseek-ai/deepseek-coder-6.7b-base` | Vulnerability detection | ~2GB (4-bit) |
| CodeLlama | `codellama/CodeLlama-7b-hf` | General security | ~2GB (4-bit) |
| StarCoder | `bigcode/starcoder` | Deep analysis | ~4GB (4-bit) |
| CodeBERT | `microsoft/codebert-base` | Fast triage | ~500MB |
| CodeT5 | `Salesforce/codet5-base` | Code understanding | ~900MB |

**Key Settings:**
- **device_preference**: `["cuda", "cpu"]` - GPU first, fallback to CPU
- **dtype**: `"bf16"` (modern GPUs), `"fp16"` (older GPUs), `"fp32"` (CPU)
- **quantization**: `"4bit"` (~2GB for 7B), `"8bit"` (~4GB), `null` (full)
- **temperature**: `0.1-0.2` for deterministic security analysis

### Cloud LLMs (OpenAI, Anthropic, Google)

**Register via API:**
```bash
# OpenAI GPT-4o mini
curl -X POST http://localhost:5000/api/models/registry \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "openai_cloud",
    "provider_id": "openai",
    "model_name": "gpt-4o-mini",
    "display_name": "GPT-4o mini",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "settings": {
      "max_tokens": 2048,
      "temperature": 0.1
    }
  }'

# Anthropic Claude 3.5 Sonnet
curl -X POST http://localhost:5000/api/models/registry \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "anthropic_cloud",
    "provider_id": "anthropic",
    "model_name": "claude-3-5-sonnet-20241022",
    "display_name": "Claude 3.5 Sonnet",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "settings": {
      "max_tokens": 4096,
      "temperature": 0.1
    }
  }'

# Google Gemini 1.5 Flash
curl -X POST http://localhost:5000/api/models/registry \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "google_cloud",
    "provider_id": "google",
    "model_name": "gemini-1.5-flash",
    "display_name": "Gemini 1.5 Flash",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "settings": {
      "max_tokens": 2048,
      "temperature": 0.1
    }
  }'
```

**Supported Providers:**
- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5-Turbo
- **Anthropic**: Claude 3.5 Opus, Sonnet, Haiku
- **Google**: Gemini 1.5 Pro, Flash, Gemini Pro

**Cloud Features:**
- Optimized system prompts for JSON compliance
- Auto-retry with exponential backoff
- Rate limiting protection
- Async execution with event loop isolation

---

## Running Scans

### Web UI

1. Go to **Home** → Upload ZIP or paste code
2. Select models to use (multi-select supported)
3. Choose **Consensus Strategy**:
   - **Union**: All findings from all models
   - **Majority**: Only findings found by 2+ models
   - **Weighted**: Score-based filtering by model confidence
   - **Judge**: Use a judge model to evaluate and merge
4. Click **Start Scan**
5. Watch real-time progress (SSE streaming)
6. View results, export SARIF/CSV

### API

**Upload files:**
```bash
curl -X POST http://localhost:5000/api/scans/upload \
  -F "files=@app.py" \
  -F "files=@utils.py" \
  -F "model_ids=ollama:llama3.2" \
  -F "model_ids=hf:deepseek_coder" \
  -F "consensus_strategy=union"
```

**Submit inline code:**
```bash
curl -X POST http://localhost:5000/api/scans/submit \
  -H "Content-Type: application/json" \
  -d '{
    "code": "eval(user_input)",
    "language": "python",
    "model_ids": ["openai:gpt-4o-mini"],
    "consensus_strategy": "union"
  }'
```

**Get scan status:**
```bash
curl http://localhost:5000/api/scans/{scan_id}/status
```

**Download results:**
```bash
# SARIF
curl http://localhost:5000/api/scans/{scan_id}/export/sarif -o results.sarif

# CSV
curl http://localhost:5000/api/scans/{scan_id}/export/csv -o results.csv
```

---




---

## Model Settings

Edit any registered model in the UI to configure:

### Runtime Settings
- **Device**: `cpu`, `cuda`, `mps` (Metal for Mac)
- **Quantization**: `4bit`, `8bit`, `null` (full precision)
- **Max Concurrency**: How many parallel scans per model
- **Keep-Alive**: Model unload timeout (Ollama)

### Generation Settings
- **Temperature**: `0.0-1.0` (lower = deterministic)
- **Max Tokens**: Maximum output length
- **Top-p**: Nucleus sampling threshold
- **Top-k**: Token sampling limit

### Provider-Specific
- **HuggingFace**: `device_map`, `dtype`, `adapter_model`, `trust_remote_code`
- **Ollama**: Custom `options` dict (num_ctx, num_gpu, etc.)
- **Cloud**: API endpoint overrides, custom headers


## Development

### Project Structure

```
aegis/
├── api/                  # REST API routes
├── config/               # YAML configuration loader
├── consensus/            # Multi-model merging strategies
├── database/             # SQLite repositories + migrations
│   ├── init.py          # Auto-initialization logic
│   └── repositories.py  # Model, scan, finding repos
├── models/               # Model registry and runtime
│   ├── registry.py      # ModelRegistryV2 (persistent)
│   ├── runtime.py       # Model loading + caching
│   ├── runners/         # Role-based execution
│   └── providers/       # Ollama, HF, Cloud adapters
├── parsers/              # JSON/binary output parsers
├── pipeline/             # Scan execution engine
├── services/             # Scan worker + background queue
├── static/               # Web UI assets (CSS/JS)
└── templates/            # Jinja2 HTML templates

config/models.yaml        # Model presets (seeded on init)
data/aegis.db             # SQLite database (auto-created)
requirements/             # Dependency files by use case
```

### Adding a New Provider

1. Create provider adapter in `aegis/models/providers/`:
```python
from aegis.models.providers.base import BaseProvider

class MyProvider(BaseProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass
```

2. Register in `aegis/models/provider_factory.py`:
```python
def create_provider(model: ModelRecord) -> BaseProvider:
    if model.model_type == ModelType.MY_PROVIDER:
        return MyProvider(model.model_name, model.settings)
    # ...
```

3. Add to database schema in `aegis/database/schema.sql` (if needed)

4. Create YAML preset in `config/models.yaml`


## Troubleshooting

### Database Issues

**Problem**: "Database locked" error
**Solution**: Only one Aegis instance can run at a time. Stop other instances or use separate databases with `AEGIS_DB_PATH` env var.

**Problem**: "No providers found"
**Solution**: Database auto-initializes on first run. If it fails, manually run `python scripts/migrate_to_v2.py`

### Model Loading

**Problem**: HuggingFace models fail to load
**Solution**:
- Ensure 8GB+ RAM available (16GB recommended)
- Use 4-bit quantization: `"quantization": "4bit"`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

**Problem**: Ollama models not detected
**Solution**:
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Set custom URL: `OLLAMA_BASE_URL=http://192.168.1.100:11434 python app.py`

**Problem**: Cloud LLMs return prose instead of JSON
**Solution**:
- Update to latest version (system prompts added)
- Lower temperature to 0.1 or 0.0
- Use `json_schema` parser (auto-validates)

### Scan Errors

**Problem**: Scans stuck at "Running"
**Solution**:
- Check browser console for SSE connection errors
- Verify model is loaded (first scan may take 1-2 minutes)
- Check `logs/aegis.log` for detailed errors

**Problem**: "asyncio event loop" errors
**Solution**: Fixed in latest version. Cloud providers now use ThreadPoolExecutor isolation.

**Problem**: High memory usage
**Solution**:
- Use quantization for HF models
- Reduce `max_concurrency` to 1
- Use smaller models (7B instead of 13B+)

### API Rate Limits

**Problem**: Cloud API rate limit errors
**Solution**:
- Default rate limits are conservative (5 req/sec)
- Edit provider config in database to increase
- Use Ollama/HF for high-throughput scanning

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Contributing

Pull requests welcome! Please:
- Follow existing code style (PEP 8)
- Add type hints to new functions
- Update documentation for new features
- Test with multiple providers (Ollama + Cloud)
