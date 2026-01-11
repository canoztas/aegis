<div align="center">
  <img src="aegis/static/img/aegis-logo.svg" alt="aegis logo" width="200"/>
  <h1>Aegis - AI-Powered SAST Framework</h1>
  <p>Use LLMs to find security vulnerabilities in your code</p>
</div>

---

## Demos

[![Demo Video (Ollama)](https://img.youtube.com/vi/StXTwdxQyQI/0.jpg)](https://youtu.be/StXTwdxQyQI)

- **Ollama**: https://youtu.be/StXTwdxQyQI
- **Cloud AI**: Coming soon
- **HuggingFace Models**: Coming soon
- **ML Models**: Coming soon

---

## Features

- **Multi-Provider Support**: Ollama, HuggingFace, OpenAI, Anthropic, Google Gemini
- **Registry-Driven**: All models registered in SQLite - no hidden magic
- **Role-Based Scanning**: Triage, deep scan, judge, explain
- **Runtime Control**: Choose CPU/GPU, quantization, concurrency per model
- **Consensus Engine**: Combine results from multiple models
- **Web UI + API**: Manage models, run scans, view history
- **Export**: SARIF and CSV formats

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python scripts/migrate_to_v2.py
```

### 3. Run Server

```bash
python app.py
```

Open: http://localhost:5000

---

## Using Models

### Ollama (Local)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3`
3. In Aegis UI → **Models** → **DISCOVER_OLLAMA** → Register
4. Run a scan!

### HuggingFace (Local)

1. In Aegis UI → **Models** → **HUGGING_FACE**
2. Click **Register** on CodeBERT or CodeAstra
3. Models download automatically on first use

### Cloud LLMs

Set API key as environment variable:

```bash
# Windows
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
set GOOGLE_API_KEY=...

# Linux/Mac
export OPENAI_API_KEY=sk-...
```

Register via API:

```bash
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
```

**Supported Cloud Providers:**
- OpenAI (GPT-4, GPT-3.5-Turbo)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini Pro, Gemini 1.5)

---

## Running Scans

### Via UI

1. Go to **Home** → Upload files or paste code
2. Select registered models
3. Click **Start Scan**
4. View results in real-time

### Via API

```bash
curl -X POST http://localhost:5000/api/scans/upload \
  -F "files=@vulnerable.py" \
  -F "model_ids=ollama:llama3"
```

---

## Model Settings

Edit any registered model to configure:

**Runtime:**
- Device (CPU/GPU)
- Quantization (int4, int8)
- Max concurrency
- Keep-alive time

**Generation:**
- Temperature
- Max tokens
- Top-p, top-k

**Provider-Specific:**
- HuggingFace: adapter model, device_map, dtype
- Ollama: options dict
- Cloud: API keys, base URL

---

## Architecture

```
User Upload → Pipeline Executor → Models (via Registry)
                                  ↓
                    Parsers (JSON/Classification)
                                  ↓
                    Findings → SARIF/CSV Export
```

- **Model Registry**: SQLite database (`data/aegis.db`)
- **Providers**: Abstraction layer for Ollama/HF/Cloud
- **Runners**: Role-based prompt builders (triage, deep_scan, etc.)
- **Parsers**: Convert raw LLM output to structured findings
- **Consensus**: Merge results from multiple models

---

## Development

### Project Structure

```
aegis/
├── api/              # Flask routes
├── models/           # Registry, runtime, providers
├── pipeline/         # Scan execution engine
├── parsers/          # Output parsing
├── templates/        # Web UI
└── static/           # CSS/JS

config/models.yaml    # Model presets
data/aegis.db         # Registry database
```

### Requirements

- **Minimal** (cloud only): `pip install -r requirements-minimal.txt` (~100MB)
- **Standard** (CPU): `pip install -r requirements.txt` (~3.5GB)
- **GPU** (CUDA 11.8): `pip install -r requirements-gpu.txt` (~5GB)

See [REQUIREMENTS.md](REQUIREMENTS.md) for details.

---

## Contributing

Pull requests welcome! Please:
- Follow existing code style
- Add tests for new features
- Update documentation

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Troubleshooting

**Cloud LLMs return prose instead of JSON?**
- Make sure you're using the latest version (system prompts fix)
- Temperature should be 0.1 or lower

**Models not loading?**
- Check `logs/aegis.log` for errors
- Verify API keys are set correctly
- HuggingFace models need ~8GB RAM minimum

**Scan stuck?**
- Check model status in UI
- Some models take time to load (first run)
- Check console for rate limit errors (cloud APIs)

---

**Made with ❤️ for security researchers**
