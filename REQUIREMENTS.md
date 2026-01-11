# Requirements Files Guide

This project provides multiple requirements files for different deployment scenarios.

## Quick Start

### For Development (Most Common)
```bash
pip install -r requirements.txt
```

### For Cloud-Only Deployment (No Local Models)
```bash
pip install -r requirements-minimal.txt
```

### For GPU-Accelerated Local Models
```bash
pip install -r requirements-gpu.txt
```

---

## Requirements Files

### `requirements.txt`
**General purpose requirements with flexible version constraints**
- ✅ Use for: Local development, testing
- 📦 Size: ~3.5GB (includes PyTorch CPU)
- 🚀 Includes: Flask, PyTorch, Transformers, Cloud APIs

### `requirements-minimal.txt`
**Lightweight cloud-only deployment**
- ✅ Use for: Production servers using only OpenAI/Anthropic/Google
- 📦 Size: ~100MB (no PyTorch/Transformers)
- ⚡ Fast installation
- ❌ Cannot run local HuggingFace models

### `requirements-gpu.txt`
**GPU-accelerated local model inference**
- ✅ Use for: High-performance local scanning with HuggingFace models
- 📦 Size: ~5GB (includes PyTorch CUDA 11.8)
- 🎮 Requires: NVIDIA GPU with CUDA 11.8+
- 🚀 Includes: accelerate, bitsandbytes for quantization

### `requirements-prod.txt`
**Pinned versions for production stability**
- ✅ Use for: Production deployments requiring reproducible builds
- 🔒 All versions locked to specific releases
- 📌 Update these versions periodically for security patches

### `requirements-dev.txt`
**Development tools and testing**
- ✅ Use for: Contributing to the project
- 🧪 Includes: pytest, black, flake8, mypy
- 📚 Includes: Sphinx for documentation

---

## Installation Guide

### Step 1: Choose Your Setup

#### **Option A: Full Local Setup (Default)**
```bash
# Install base requirements
pip install -r requirements.txt

# Optional: Install dev tools
pip install -r requirements-dev.txt
```

#### **Option B: Cloud-Only Setup**
```bash
pip install -r requirements-minimal.txt
```

#### **Option C: GPU Setup**
```bash
# Check CUDA version first
nvcc --version  # Should show CUDA 11.8 or higher

# Install GPU requirements
pip install -r requirements-gpu.txt
```

### Step 2: Verify Installation
```bash
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Ollama (Optional - for local Ollama models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud

# Cloud API Keys (Optional - only if using cloud LLMs)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Database
DATABASE_PATH=./data/aegis.db
```

---

## Docker Support

### CPU-Only Docker
```dockerfile
FROM python:3.11-slim
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
```

### GPU Docker
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt
```

---

## Troubleshooting

### PyTorch Installation Issues

**Problem**: PyTorch is huge (~2GB)
```bash
# Install CPU-only version (smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: CUDA version mismatch
```bash
# Check your CUDA version
nvcc --version

# Install matching PyTorch
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Transformers Installation Issues

**Problem**: Takes forever to download
```bash
# Set HuggingFace cache to faster disk
export HF_HOME=/path/to/fast/disk
```

**Problem**: Out of disk space
```bash
# Use cloud-only setup instead
pip install -r requirements-minimal.txt
```

### python-magic Issues (Windows)

**Problem**: `python-magic` requires libmagic
```bash
# Windows: Install python-magic-bin instead
pip uninstall python-magic
pip install python-magic-bin
```

---

## Platform-Specific Notes

### Windows
- Use `python-magic-bin` instead of `python-magic`
- CUDA installation requires Visual Studio Build Tools

### macOS
- PyTorch has MPS (Metal) support for Apple Silicon
- Install with: `pip install torch torchvision`

### Linux
- Most straightforward installation
- For GPU: Ensure NVIDIA drivers and CUDA toolkit are installed

---

## Dependency Size Comparison

| Requirements File | Installed Size | Key Libraries |
|-------------------|----------------|---------------|
| `requirements-minimal.txt` | ~100MB | Flask, OpenAI, Anthropic |
| `requirements.txt` | ~3.5GB | + PyTorch CPU, Transformers |
| `requirements-gpu.txt` | ~5GB | + PyTorch CUDA, Accelerate |

---

## Updating Dependencies

```bash
# Check outdated packages
pip list --outdated

# Update specific package
pip install --upgrade flask

# Regenerate requirements with current versions
pip freeze > requirements-locked.txt
```

---

## License

All dependencies are subject to their respective licenses. Check individual package licenses before commercial use.
