# HuggingFace Integration - Quick Start Guide

## What Was Fixed

### ✅ Model Caching
- **Problem**: Models re-downloaded on every app restart (499MB each time!)
- **Solution**: Automatic local caching to `aegis/huggingface/cache/`
- **Result**: First run downloads, subsequent runs load from cache instantly

### ✅ Simple Output Configuration
- **Problem**: Complex JSON configs hard to write for every model
- **Solution**: Example-based learning system - just provide sample outputs
- **Result**: Add any HuggingFace model in 5 minutes without writing code

## Getting Started (3 Steps)

### Step 1: Seed Built-in Models

Navigate to `http://localhost:5000/models` and click the **HuggingFace** tab, then click **"Seed Built-in Models"**

This adds:
- **CodeBERT** - Fast binary classifier (~500MB, CPU-friendly)
- **CodeAstra-7B** - Advanced LLM scanner (~4GB with quantization, GPU recommended)

### Step 2: Enable Models

Toggle the switch on the models you want to use.

**Recommendation**:
- Start with **CodeBERT only** for fast scans
- Enable **CodeAstra-7B** for deeper analysis (requires ~4GB RAM)

### Step 3: Run a Scan

Go to the main scan page and upload code as usual. Enabled HuggingFace models will automatically participate in the scan!

## Adding Custom Models

### Example: Adding a New Vulnerability Scanner

1. **Click "Add Model"**

2. **Fill Basic Info**:
   - Model ID: `hf_my_scanner`
   - Display Name: `My Vulnerability Scanner`
   - HF Model ID: `username/model-name` (from HuggingFace Hub)
   - Task Type: Select based on model type
     - Text Classification → Binary/multi-class classifiers
     - Text Generation → LLM models

3. **Configure Output Parsing** (Simple Mode):

```json
{
  "type": "auto",
  "examples": [
    {
      "output": 0,
      "is_vulnerable": false
    },
    {
      "output": 1,
      "is_vulnerable": true,
      "name": "Security Vulnerability",
      "severity": "high",
      "cwe": "CWE-1035"
    }
  ]
}
```

**How it works**:
- System sees your examples
- Learns what output `0` and `1` mean
- Automatically applies pattern to new scans

4. **Save** and **Enable** the model

## Example Configurations

### Classification Model (Like CodeBERT)

```json
{
  "type": "auto",
  "examples": [
    {
      "output": 0,
      "is_vulnerable": false,
      "message": "Safe code - class 0"
    },
    {
      "output": 1,
      "is_vulnerable": true,
      "name": "Insecure Code Detected",
      "severity": "high",
      "cwe": "CWE-1035",
      "message": "Vulnerable code - class 1"
    }
  ]
}
```

### Text Generation Model (Like CodeAstra-7B)

```json
{
  "type": "auto",
  "examples": [
    {
      "output": "No vulnerabilities found in this code.",
      "is_vulnerable": false
    },
    {
      "output": "This code contains a SQL injection vulnerability.",
      "is_vulnerable": true,
      "name": "SQL Injection",
      "severity": "critical",
      "cwe": "CWE-89"
    },
    {
      "output": "Potential XSS detected. User input rendered without escaping.",
      "is_vulnerable": true,
      "name": "Cross-Site Scripting (XSS)",
      "severity": "high",
      "cwe": "CWE-79"
    }
  ]
}
```

### JSON Output Model

```json
{
  "type": "auto",
  "examples": [
    {
      "output": {
        "vulnerabilities": [
          {
            "vuln_name": "SQL Injection",
            "risk_level": "critical",
            "cwe_id": "CWE-89",
            "description": "Unsafe SQL query"
          }
        ]
      },
      "is_vulnerable": true
    }
  ]
}
```

## How Model Caching Works

### First Run:
```
1. User starts scan
2. Model downloads from HuggingFace (499MB)
3. Saved to: aegis/huggingface/cache/
4. Loaded into memory
5. Scan runs
```

### Subsequent Runs:
```
1. User starts scan
2. Model loads from local cache (instant!)
3. Loaded into memory
4. Scan runs
```

### Cache Location:
```
aegis/
├── huggingface/
│   └── cache/
│       ├── models--mrm8488--codebert-base-finetuned-detect-insecure-code/
│       │   └── (model files: 499MB)
│       └── models--rootxhacker--CodeAstra-7B/
│           └── (model files: ~4GB with quantization)
```

**Cache is automatically managed** - No manual cleanup needed!

## Understanding Model Output

### What Your Model Outputs

Different models output different formats:

| Model Type | Output Format | Example |
|-----------|-----------------|---------|
| Binary Classifier | Class index or logits | `1` or `[0.2, 0.8]` |
| Text Generator | Generated text | `"SQL injection detected"` |
| JSON Model | JSON object | `{"vulns": [...]}` |

### How to Configure

**Just provide examples!** The system learns:

```json
{
  "type": "auto",
  "examples": [
    {"output": <what model outputs>, "is_vulnerable": <true/false>, ...}
  ]
}
```

The mapper will:
1. Compare new outputs to your examples
2. Match patterns automatically
3. Generate findings in Aegis format

## Testing Your Configuration

### Quick Test Workflow

1. **Add model with 2 examples** (safe + vulnerable)
2. **Enable model**
3. **Run scan on test code**
4. **Check findings** - Do they make sense?
5. **If not, add more examples** and update config

### Example Test Code (Python)

```python
# Safe code
def add(a, b):
    return a + b

# Vulnerable code
def query_user(user_id):
    sql = f"SELECT * FROM users WHERE id = {user_id}"  # SQL Injection!
    return execute(sql)
```

Expected:
- Model should flag the vulnerable function
- Model should not flag the safe function

## Performance Tips

### Memory Management

| Model Size | RAM Needed | Quantization | Device |
|-----------|------------|--------------|--------|
| < 500MB | ~1GB | None | CPU OK |
| 1-3GB | ~4GB | None | CPU/GPU |
| 7B params | ~14GB | None | GPU only |
| 7B params | ~4GB | 4-bit | GPU recommended |

### Speed Tips

1. **Use smaller models for quick scans**
   - CodeBERT: ~1-2 seconds per file
   - CodeAstra-7B: ~5-10 seconds per file

2. **Adjust chunk size**
   - Smaller chunks (50 lines) = faster
   - Larger chunks (200 lines) = more context, better accuracy

3. **Disable unused models**
   - Toggle off models you're not using
   - Saves memory and speeds up scans

## Troubleshooting

### Model Won't Load

**Error**: `Out of memory`
- **Fix 1**: Enable quantization (4-bit or 8-bit)
- **Fix 2**: Use smaller model
- **Fix 3**: Set device to CPU (slower but works)

**Error**: `Model not found`
- **Fix**: Check HF model ID is correct
- Visit: `https://huggingface.co/<model-id>` to verify

### No Findings Generated

**Issue**: Model runs but no vulnerabilities found

**Debug Steps**:
1. Check mapper examples - are they correct?
2. Look at model output format - does it match examples?
3. Try lowering confidence threshold
4. Add more example cases

### Cache Taking Too Much Space

**Location**: `aegis/huggingface/cache/`

**Solutions**:
- Delete cache folder (will re-download on next run)
- Use quantization (4-bit = 75% smaller)
- Disable large models you don't use

## Advanced: Manual Cache Configuration

Set custom cache directory in model config:

```json
{
  "hf_config": {
    "model_id": "username/model-name",
    "cache_dir": "/path/to/custom/cache",
    ...
  }
}
```

## Real-World Example: Adding CodeQL Model

Let's say you found a HuggingFace model `awesome/codeql-scanner`:

1. **Test the model** (optional but recommended):
```python
from transformers import pipeline

pipe = pipeline("text-classification", model="awesome/codeql-scanner")
result = pipe("def sql_query(id): return f'SELECT * FROM users WHERE id={id}'")
print(result)  # [{'label': 'VULNERABLE', 'score': 0.95}]
```

2. **Add to Aegis UI**:
   - Model ID: `hf_codeql`
   - HF Model ID: `awesome/codeql-scanner`
   - Task: Text Classification
   - Mapper Type: **Simple (Recommended)**

3. **Configure with examples**:
```json
{
  "type": "auto",
  "examples": [
    {
      "output": [{"label": "SAFE", "score": 0.9}],
      "is_vulnerable": false
    },
    {
      "output": [{"label": "VULNERABLE", "score": 0.95}],
      "is_vulnerable": true,
      "name": "CodeQL Vulnerability",
      "severity": "high",
      "cwe": "CWE-1035"
    }
  ]
}
```

4. **Done!** Model will cache on first run, then work offline.

## Benefits

✅ **No Re-downloads** - Models cached locally
✅ **No Code Needed** - Example-based configuration
✅ **Works Offline** - After first download
✅ **Fast** - Instant load from cache
✅ **Flexible** - Supports any HuggingFace model
✅ **Simple** - 5 minutes to add new model

## Next Steps

1. ✅ Seed built-in models
2. ✅ Run first scan with CodeBERT
3. ⭐ Browse HuggingFace Hub for more models
4. ⭐ Add custom models for your tech stack
5. ⭐ Fine-tune confidence thresholds

## Need Help?

- Check `aegis/huggingface/README.md` for detailed docs
- View example configs in database after seeding
- Test models incrementally (one at a time)

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Model Download | Every restart | Once, then cached |
| Add New Model | Write code | Provide examples |
| Configuration | Complex JSON | Simple examples |
| Setup Time | 30+ minutes | 5 minutes |
| Offline Work | ❌ No | ✅ Yes (after first download) |
