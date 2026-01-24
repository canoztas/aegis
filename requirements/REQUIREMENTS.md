# Requirements Files Guide

This project provides multiple requirements files for different deployment scenarios.

## Quick Start

### For Development (Most Common)
```bash
# Standard CPU-based installation
pip install -r requirements.txt
```

### For Cloud-Only Deployment (No Local Models)
```bash
# Minimal installation (~100MB) - OpenAI, Anthropic, Google APIs only
pip install -r requirements-minimal.txt
```

### For GPU-Accelerated Local Models
```bash
# GPU installation with CUDA 11.8 support (~5GB)
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

#### **Option A: Minimal (Cloud-Only)**
**Best for**: Production servers using only OpenAI/Anthropic/Google APIs
**Size**: ~100MB
**No PyTorch or Transformers** - Cannot run local HuggingFace models

```bash
# Install cloud-only dependencies
pip install -r requirements-minimal.txt
```

This installs:
- Flask and core dependencies from `requirements-base.txt`
- OpenAI, Anthropic, Google API clients
- No PyTorch, no Transformers, no local model support

#### **Option B: CPU (Standard)**
**Best for**: Local development and CPU-based local model inference
**Size**: ~3.5GB
**Includes PyTorch CPU** - Can run local HuggingFace models on CPU

```bash
# Install standard CPU dependencies
pip install -r requirements.txt
```

This installs:
- Everything from `requirements-minimal.txt`
- PyTorch (CPU-only build)
- Transformers, Sentence-Transformers
- Full local model support (CPU inference)

#### **Option C: GPU (CUDA 11.8)**
**Best for**: High-performance local model inference with NVIDIA GPUs
**Size**: ~5GB
**Requires**: NVIDIA GPU with CUDA 11.8+

```bash
# Step 1: Verify CUDA installation
nvcc --version  # Should show CUDA 11.8 or higher
nvidia-smi      # Verify GPU is detected

# Step 2: Install base dependencies first
pip install -r requirements-base.txt

# Step 3: Install GPU-optimized PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install remaining GPU dependencies
pip install -r requirements-gpu.txt
```

This installs:
- Everything from `requirements-minimal.txt`
- PyTorch with CUDA 11.8 support
- Transformers, Sentence-Transformers
- GPU acceleration libraries (accelerate, bitsandbytes)

#### **Option D: Development Setup**
```bash
# Install standard requirements
pip install -r requirements.txt

# Install development tools
pip install -r requirements-dev.txt
```

This adds:
- pytest, black, flake8, mypy
- Sphinx for documentation
- Development and testing tools

### Step 2: Verify Installation

#### Basic Verification
```bash
# Test core imports
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import aegis; print('Aegis: OK')"
```

#### For CPU/GPU Installations (requirements.txt or requirements-gpu.txt)
```bash
# Test PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__)"

# Test CUDA availability (GPU only)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Test Transformers
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Complete verification script
python -c "
import torch
import transformers
print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU Device:', torch.cuda.get_device_name(0))
print('Transformers:', transformers.__version__)
print('All dependencies verified successfully!')
"
```

#### Test Aegis Device Detection
```bash
# Test automatic device selection
python -c "
from aegis.utils.device import get_device
device = get_device()
print(f'Selected device: {device}')
print(f'Device type: {device.type}')
"
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Device Configuration (NEW!)
# Controls which device PyTorch uses for local model inference
# Options: cpu, cuda, mps, auto
# Default: auto (automatically detects best available device)
AEGIS_DEVICE=auto

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

### Environment Variable Reference

#### `AEGIS_DEVICE` (Optional)
Controls which device PyTorch uses for local model inference.

**Valid values**:
- `auto` (default): Automatically selects best available device (cuda > mps > cpu)
- `cpu`: Force CPU inference (works with all installations)
- `cuda`: Force CUDA GPU inference (requires GPU installation)
- `mps`: Force Apple Metal Performance Shaders (macOS only)

**Examples**:
```bash
# Let Aegis auto-detect (recommended)
AEGIS_DEVICE=auto

# Force CPU even if GPU is available (useful for testing)
AEGIS_DEVICE=cpu

# Force GPU inference
AEGIS_DEVICE=cuda

# macOS: Use Apple Silicon GPU
AEGIS_DEVICE=mps
```

**When to use**:
- **Leave unset or use `auto`**: For most users (automatic detection works well)
- **Use `cpu`**: To force CPU inference for debugging or when GPU has issues
- **Use `cuda`**: To ensure GPU is being used (will fail if GPU unavailable)
- **Use `mps`**: On Apple Silicon Macs to use Metal acceleration

#### API Keys
Only required if using cloud LLM providers:

- `OPENAI_API_KEY`: Required for OpenAI models (GPT-4, GPT-3.5, etc.)
- `ANTHROPIC_API_KEY`: Required for Anthropic models (Claude 3.5, etc.)
- `GOOGLE_API_KEY`: Required for Google models (Gemini, etc.)

**Note**: If using only local models (HuggingFace or Ollama), API keys are not needed.

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

### GPU Requirements Error (FIXED!)

**Problem**: `ImportError: GPU requirements not installed` when using local HuggingFace models

**Cause**: This error previously occurred when local models were used without PyTorch/Transformers installed. This has been fixed in recent versions!

**Solution**: Install the appropriate requirements file:
```bash
# For CPU inference
pip install -r requirements.txt

# For GPU inference
pip install -r requirements-gpu.txt
```

**New behavior**: As of the latest version, Aegis gracefully handles missing dependencies:
- Cloud-only installations (`requirements-minimal.txt`) will work fine with cloud LLMs
- Local model requests without PyTorch will show clear error messages
- Use `AEGIS_DEVICE` environment variable to control device selection

### Device Selection Verification

**Problem**: Want to verify which device Aegis is using

**Solution**: Check device detection:
```bash
# Test device selection
python -c "
from aegis.utils.device import get_device
device = get_device()
print(f'Selected device: {device}')
print(f'Device type: {device.type}')
"

# Force specific device for testing
AEGIS_DEVICE=cpu python -c "from aegis.utils.device import get_device; print(get_device())"
AEGIS_DEVICE=cuda python -c "from aegis.utils.device import get_device; print(get_device())"
```

### CUDA Version Detection

**Problem**: Need to verify CUDA is properly installed

**Solution**: Check CUDA availability:
```bash
# Check CUDA toolkit version
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Test PyTorch CUDA detection
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

**Problem**: CUDA version mismatch between PyTorch and system

**Solution**: Install matching PyTorch version:
```bash
# Check your CUDA version
nvcc --version

# Install matching PyTorch
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### PyTorch Installation Issues

**Problem**: PyTorch is huge (~2GB)

**Solution**: Use CPU-only version (smaller) or minimal installation:
```bash
# Option 1: CPU-only PyTorch (smaller than full installation)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Option 2: Skip PyTorch entirely (cloud-only)
pip install -r requirements-minimal.txt
```

**Problem**: Want to verify PyTorch installation

**Solution**: Test PyTorch:
```bash
# Basic PyTorch test
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Test tensor creation
python -c "import torch; x = torch.rand(3, 3); print('Tensor created:', x.shape)"

# Test device availability
python -c "
import torch
print('CPU:', torch.device('cpu'))
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
"
```

### Transformers Installation Issues

**Problem**: Takes forever to download

**Solution**: Set HuggingFace cache to faster disk:
```bash
# Linux/macOS
export HF_HOME=/path/to/fast/disk

# Windows PowerShell
$env:HF_HOME="C:\path\to\fast\disk"

# Windows Command Prompt
set HF_HOME=C:\path\to\fast\disk
```

**Problem**: Out of disk space

**Solution**: Use cloud-only setup instead:
```bash
pip install -r requirements-minimal.txt
```

**Problem**: Models download slowly or fail

**Solution**: Use HuggingFace mirror or cache:
```bash
# Set HuggingFace endpoint (e.g., for China users)
export HF_ENDPOINT=https://hf-mirror.com

# Pre-download models
python -c "
from transformers import AutoTokenizer, AutoModel
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print('Model downloaded successfully!')
"
```

### python-magic Issues (Windows)

**Problem**: `python-magic` requires libmagic

**Solution**: Install python-magic-bin instead:
```bash
# Windows: Install python-magic-bin instead
pip uninstall python-magic
pip install python-magic-bin
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'aegis'`

**Solution**: Ensure Aegis package is in Python path:
```bash
# Option 1: Install in development mode
pip install -e .

# Option 2: Add to Python path temporarily
export PYTHONPATH="${PYTHONPATH}:/path/to/aegis"

# Option 3: Run from project root
cd /path/to/aegis
python -m aegis.cli
```

### Memory Issues

**Problem**: Out of memory when loading models

**Solution**: Use smaller models or quantization:
```bash
# Use smaller embedding model
# Edit config to use: sentence-transformers/all-MiniLM-L6-v2 (80MB)
# Instead of: sentence-transformers/all-mpnet-base-v2 (420MB)

# For GPU: Enable 8-bit quantization (requires bitsandbytes)
# This is automatically included in requirements-gpu.txt
pip install bitsandbytes
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
