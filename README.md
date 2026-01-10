<div align="center">
  <img src="aegis/static/img/aegis-logo.svg" alt="aegis logo" width="200"/>
  <h1>Aegis v1.0.0 - AI-based SAST Framework</h1>
</div>

Aegis is an open-source SAST framework that uses LLMs to analyze source code for security vulnerabilities.
It produces structured findings and detailed reports. The architecture is intentionally flexible so you can
add new LLM providers, new Hugging Face models, or even classic ML tools.

## Demos

[![Demo Video (Ollama)](https://img.youtube.com/vi/StXTwdxQyQI/0.jpg)](https://youtu.be/StXTwdxQyQI)

- Demo video (Ollama): https://youtu.be/StXTwdxQyQI
- Demo video (Cloud AI): FOR PUBLIC NOT YET RELEASED
- Demo video (Hugging Face Models): FOR PUBLIC NOT YET RELEASED
- Demo video (Old-school ML Models): FOR PUBLIC NOT YET RELEASED

## What Aegis is

- A registry-first SAST framework for AI and ML models.
- A flexible pipeline that supports multiple roles (triage, deep_scan, judge, explain).
- A pluggable parsing system that converts raw model output into normalized findings.
- A web UI and API for model management, scans, and exports.

## Architecture at a Glance

- Model Registry: single source of truth stored in SQLite.
- Discovery vs Registration: Ollama discovery is separate from registry activation.
- Providers: Ollama (local), HF local, OpenAI-compatible, Anthropic-compatible.
- Runners: role-based prompt building and execution.
- Parsers: JSON and classification parsers with pluggable extension points.
- Runtime Manager: per-model CPU/GPU selection, device_map, dtype, quantization, concurrency.

## Features

- Registry-driven scans (no hidden model selection).
- Local Ollama discovery and registration.
- Hugging Face presets (CodeBERT triage, CodeAstra deep_scan).
- JSON and HF classification parsers with fallback handling.
- Runtime controls per model (CPU/GPU).
- Consensus engine for multi-model results.
- Pipeline execution for role-based scans.
- Web UI for model control and scan history.
- SARIF and CSV exports.

## Quickstart

### 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional local HF dependencies:
```bash
pip install transformers torch peft accelerate
# Optional for 4-bit/8-bit: pip install bitsandbytes
```

### 2) Initialize the registry

```bash
python scripts/migrate_to_v2.py
```

This seeds providers and models from `config/models.yaml` into `data/aegis.db`.

### 3) Run the server

```bash
python app.py
```

Defaults to `http://127.0.0.1:5000` (override with `HOST` and `PORT`).

### 4) Open the UI

- Home: `http://localhost:5000/`
- Models: `http://localhost:5000/models`
- History: `http://localhost:5000/history`

## Model Management (UI)

In Models:
- REGISTERED shows all registered models and lets you edit settings.
- DISCOVER_OLLAMA pulls local Ollama models.
- HUGGING_FACE shows presets and registers them with one click.

Editable settings:
- Runtime: device, device_map, dtype, quantization, max concurrency, keep-alive.
- Generation: max_new_tokens, min_new_tokens, temperature, top_p, do_sample.
- Provider: HF adapter/base model, Ollama options, prompt template.

## Adding a New Hugging Face Model

You can add a new HF model via API or directly in code. The only required fields are
model_id, model_type, provider_id, model_name, roles, and parser_id.

### Example: register a new HF model (API)

```bash
curl -X POST http://localhost:5000/api/models/registry \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "hf_local",
    "provider_id": "huggingface",
    "model_name": "your-org/your-model",
    "display_name": "Your HF Model",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "settings": {
      "task_type": "text-generation",
      "runtime": {
        "device_preference": ["cuda", "cpu"],
        "dtype": "bf16"
      },
      "generation_kwargs": {
        "max_new_tokens": 512,
        "min_new_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": true
      }
    }
  }'
```

### Example: register a new HF model (code)

```python
from aegis.models.registry import ModelRegistryV2
from aegis.models.schema import ModelType, ModelRole

registry = ModelRegistryV2()
registry.register_model(
    model_id="hf:your_model",
    model_type=ModelType.HF_LOCAL,
    provider_id="huggingface",
    model_name="your-org/your-model",
    display_name="Your HF Model",
    roles=[ModelRole.DEEP_SCAN],
    parser_id="json_schema",
    settings={
        "task_type": "text-generation",
        "generation_kwargs": {"max_new_tokens": 256},
        "runtime": {"device_preference": ["cuda", "cpu"]},
    },
)
```

### Custom output parsing

If your model does not return JSON or standard HF labels, implement a parser and reference it by ID:

- Add a parser in `aegis/models/parsers/`.
- Register it in `aegis/models/parser_factory.py` or use a dotted path
  like `your_package.your_parser.CustomParser`.

## Scanning

### Standard scan (upload ZIP)

```bash
curl -X POST http://localhost:5000/api/scan \
  -F "file=@path/to/source.zip" \
  -F "models=ollama:qwen2.5-coder:7b,hf:codeastra_7b" \
  -F "consensus_strategy=union"
```

### Pipeline scan

```bash
curl -X POST http://localhost:5000/api/scan/pipeline \
  -F "file=@path/to/source.zip" \
  -F "pipeline=classic"
```

### Exports

```bash
curl http://localhost:5000/api/scan/<scan_id>/sarif
curl http://localhost:5000/api/scan/<scan_id>/csv
```

## Runtime Settings (CPU/GPU)

Runtime configuration lives in `settings.runtime`:

```json
{
  "runtime": {
    "device_preference": ["cuda", "cpu"],
    "device_map": "auto",
    "dtype": "bf16",
    "quantization": "4bit",
    "max_concurrency": 1,
    "keep_alive_seconds": 0,
    "allow_fallback": true
  }
}
```

Notes:
- For GPU, install a CUDA-enabled PyTorch build.
- If CUDA is unavailable, Aegis can auto-fallback to CPU (unless require_device is set).

## Configuration (Seed)

`config/models.yaml` is used to seed providers/models into the registry.
To apply changes, rerun:

```bash
python scripts/migrate_to_v2.py
```

The registry in `data/aegis.db` remains the source of truth for scans.

## API Reference (current)

- `GET /api/models/discovered/ollama`
- `POST /api/models/registry`
- `GET /api/models/registry`
- `PATCH /api/models/registry/<model_id>`
- `POST /api/models/ollama/pull`
- `POST /api/models/test`
- `GET /api/models/hf/presets`
- `POST /api/models/hf/register_preset`
- `POST /api/scan`
- `GET /api/scan/<scan_id>`
- `GET /api/scan/<scan_id>/status`
- `POST /api/scan/<scan_id>/cancel`
- `POST /api/scan/pipeline`
- `GET /api/scan/<scan_id>/sarif`
- `GET /api/scan/<scan_id>/csv`

## Extending Aegis with Other Tools

Aegis is designed as a framework. You can add:
- Traditional ML classifiers.
- Static analyzers or heuristics.
- Custom pre-filters or post-processors.

Use the same registry, runner, and parser pattern to integrate new tools.

## Troubleshooting

- Keras/TensorFlow warnings on startup: set `TRANSFORMERS_NO_TF=1`.
- GPU not used: check `torch.cuda.is_available()` and install a CUDA-enabled torch wheel.
- HF CodeAstra output not JSON: increase min_new_tokens and enforce a JSON-only prompt template.
- HF missing dependencies: install `transformers torch peft accelerate`.

## Project Structure

```
aegis/
├─ aegis/                 # Core server and model system
├─ aegis/models/          # Registry, parsers, runners, runtime manager
├─ aegis/providers/       # Local providers (HF)
├─ aegis/connectors/      # External connectors (Ollama/OpenAI)
├─ aegis/templates/       # UI
├─ aegis/static/          # UI assets
├─ config/models.yaml     # Seed config
├─ data/aegis.db          # Registry + scan history (SQLite)
├─ scripts/               # Migration and helper scripts
```

## Tests

```bash
pytest -q
```

## License

MIT. See `LICENSE`.
