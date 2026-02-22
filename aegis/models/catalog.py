"""
Built-in Model Catalog - Single source of truth for pre-registered model presets.

This module defines all curated models that appear in the UI catalog.
Models can be registered and downloaded from the catalog without needing
to manually configure all settings.
"""

from typing import Dict, List, Any, Optional
from enum import Enum


class CatalogCategory(str, Enum):
    """Categories for catalog models."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CLOUD = "cloud"
    CLASSIC_ML = "classic_ml"
    AGENTIC = "agentic"


class CatalogStatus(str, Enum):
    """Status of a catalog model."""
    AVAILABLE = "available"      # In catalog, not registered
    ADDED = "added"              # Registered but not downloaded/ready
    DOWNLOADING = "downloading"  # Download in progress
    READY = "ready"              # Registered and ready to use
    NEEDS_KEY = "needs_key"      # Cloud model needs API key
    NEEDS_ARTIFACT = "needs_artifact"  # Classic ML needs artifact path


# =============================================================================
# HuggingFace Model Definitions (moved from hf_local.py presets)
# =============================================================================

HF_CODEBERT_INSECURE = {
    "catalog_id": "codebert_insecure",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "CodeBERT Insecure Code Detector",
    "description": "Fast binary classifier for detecting potentially insecure code patterns. Good for triage.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "mrm8488/codebert-base-finetuned-detect-insecure-code",
    "task_type": "text-classification",
    "roles": ["triage"],
    "parser_id": "hf_classification",
    "parser_config": {
        "positive_labels": ["LABEL_1", "VULNERABLE", "INSECURE"],
        "negative_labels": ["LABEL_0", "SAFE", "SECURE"],
        "threshold": 0.5,
    },
    "hf_kwargs": {},
    "generation_kwargs": {},
    "size_mb": 500,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["classifier", "fast", "triage", "python", "catalog"],
}

HF_CODEBERT_PRIMEVUL = {
    "catalog_id": "codebert_primevul",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "CodeBERT PrimeVul-BigVul",
    "description": "Multi-task vulnerability detector with CWE classification. Custom architecture.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "mahdin70/CodeBERT-PrimeVul-BigVul",
    "task_type": "text-classification",
    "roles": ["triage"],
    "parser_id": "hf_classification",
    "parser_config": {
        "positive_labels": ["LABEL_1", "1", "vulnerable"],
        "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
        "threshold": 0.5,
    },
    "hf_kwargs": {
        "trust_remote_code": True,
        "tokenizer_id": "microsoft/codebert-base",
    },
    "generation_kwargs": {},
    "custom_loading": True,
    "size_mb": 500,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["classifier", "cwe", "triage", "custom-arch", "catalog"],
}

HF_VULBERTA_DEVIGN = {
    "catalog_id": "vulberta_devign",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "VulBERTa MLP Devign",
    "description": "RoBERTa-based C/C++ vulnerability detector trained on CodeXGLUE Devign. Requires libclang for tokenizer.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "claudios/VulBERTa-MLP-Devign",
    "task_type": "text-classification",
    "roles": ["triage"],
    "parser_id": "hf_classification",
    "parser_config": {
        "positive_labels": ["LABEL_1"],
        "negative_labels": ["LABEL_0"],
        "threshold": 0.5,
    },
    "hf_kwargs": {
        "trust_remote_code": True,
        # VulBERTa uses a custom C/C++ tokenizer that requires libclang
        # Fallback to roberta-base tokenizer if libclang not available
        "tokenizer_id": "roberta-base",
    },
    "generation_kwargs": {},
    "custom_loading": True,
    "size_mb": 500,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["classifier", "c", "cpp", "triage", "custom-arch", "catalog"],
}

HF_UNIXCODER_PRIMEVUL = {
    "catalog_id": "unixcoder_primevul",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "UnixCoder PrimeVul-BigVul",
    "description": "UniXcoder-based vulnerability detector trained on BigVul and PrimeVul datasets. Custom architecture.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "mahdin70/UnixCoder-Primevul-BigVul",
    "task_type": "text-classification",
    "roles": ["triage"],
    "parser_id": "hf_classification",
    "parser_config": {
        "positive_labels": ["LABEL_1", "1", "vulnerable"],
        "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
        "threshold": 0.5,
    },
    "hf_kwargs": {
        "trust_remote_code": True,
        "tokenizer_id": "microsoft/unixcoder-base",
    },
    "generation_kwargs": {},
    "custom_loading": True,
    "size_mb": 500,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["classifier", "triage", "custom-arch", "catalog"],
}

HF_CODEASTRA_7B = {
    "catalog_id": "codeastra_7b",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "CodeAstra 7B",
    "description": "Generative model for code analysis and vulnerability detection. LoRA adapter on Mistral.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "rootxhacker/CodeAstra-7B",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "load_in_4bit": True,
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "min_new_tokens": 32,
        "max_new_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    },
    "adapter_id": "rootxhacker/CodeAstra-7B",
    "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
    "size_mb": 4000,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "lora", "gpu", "catalog"],
}

HF_QWEN25_CODER_7B = {
    "catalog_id": "qwen25_coder_7b",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "Qwen 2.5 Coder 7B Instruct",
    "description": "Qwen 2.5 code-specialized 7B model for code analysis and vulnerability detection.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bf16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    },
    "size_mb": 14000,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "gpu", "catalog"],
}

HF_VULNLLM_R_7B = {
    "catalog_id": "vulnllm_r_7b",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "VulnLLM-R 7B",
    "description": "Qwen2.5-7B fine-tuned for vulnerability detection with chain-of-thought reasoning. Supports C, C++, Python, Java.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "UCSB-SURFI/VulnLLM-R-7B",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bf16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    },
    "size_mb": 16000,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "vulnerability", "cot", "gpu", "catalog"],
}

HF_DEEPSEEK_CODER_V2_LITE = {
    "catalog_id": "deepseek_coder_v2_lite",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "DeepSeek Coder V2 Lite Instruct",
    "description": "DeepSeek Coder V2 Lite (16B) - efficient code model with MoE architecture.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bf16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    },
    "size_mb": 32000,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "moe", "gpu", "catalog"],
}

HF_STARCODER2_15B = {
    "catalog_id": "starcoder2_15b",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "StarCoder2 15B Instruct",
    "description": "StarCoder2 15B instruction-tuned model for code generation and analysis.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "bigcode/starcoder2-15b-instruct-v0.1",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bf16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
    },
    "size_mb": 30000,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "gpu", "catalog"],
}

HF_PHI35_MINI = {
    "catalog_id": "phi35_mini",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "Phi-3.5 Mini Instruct",
    "description": "Microsoft Phi-3.5 Mini (3.8B) - compact but capable instruction model.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "microsoft/Phi-3.5-mini-instruct",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "bf16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    },
    "size_mb": 7600,
    "requires_gpu": True,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "deep_scan", "compact", "gpu", "catalog"],
}

HF_QWEN25_05B = {
    "catalog_id": "qwen25_05b",
    "category": CatalogCategory.HUGGINGFACE,
    "display_name": "Qwen 2.5 0.5B Instruct",
    "description": "Qwen 2.5 tiny (0.5B) - fast, lightweight model for quick analysis. Works on CPU.",
    "provider_id": "huggingface",
    "model_type": "hf_local",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "fp16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
    },
    "small_model": True,
    "size_mb": 1000,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["generative", "triage", "deep_scan", "tiny", "cpu", "fast", "catalog"],
}


# =============================================================================
# Cloud Model Definitions
# =============================================================================

CLOUD_OPENAI_GPT4O = {
    "catalog_id": "openai_gpt4o",
    "category": CatalogCategory.CLOUD,
    "display_name": "OpenAI GPT-4o",
    "description": "OpenAI's most capable model for deep security analysis. Requires API key.",
    "provider_id": "openai",
    "model_type": "openai_cloud",
    "model_name": "gpt-4o",
    "task_type": "text-generation",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,  # Cloud - no download
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "openai", "deep_scan", "judge", "catalog"],
}

CLOUD_OPENAI_GPT4O_MINI = {
    "catalog_id": "openai_gpt4o_mini",
    "category": CatalogCategory.CLOUD,
    "display_name": "OpenAI GPT-4o Mini",
    "description": "Cost-effective OpenAI model for triage and quick analysis. Requires API key.",
    "provider_id": "openai",
    "model_type": "openai_cloud",
    "model_name": "gpt-4o-mini",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "openai", "triage", "deep_scan", "fast", "catalog"],
}

CLOUD_ANTHROPIC_SONNET = {
    "catalog_id": "anthropic_claude_sonnet",
    "category": CatalogCategory.CLOUD,
    "display_name": "Claude Sonnet 4",
    "description": "Anthropic Claude Sonnet - excellent for security analysis and judging. Requires API key.",
    "provider_id": "anthropic",
    "model_type": "anthropic_cloud",
    "model_name": "claude-sonnet-4-20250514",
    "task_type": "text-generation",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "anthropic", "deep_scan", "judge", "catalog"],
}

CLOUD_ANTHROPIC_HAIKU_35 = {
    "catalog_id": "anthropic_claude_haiku_35",
    "category": CatalogCategory.CLOUD,
    "display_name": "Claude Haiku 3.5",
    "description": "Anthropic Claude 3.5 Haiku - fast and cost-effective for triage. Requires API key.",
    "provider_id": "anthropic",
    "model_type": "anthropic_cloud",
    "model_name": "claude-3-5-haiku-20241022",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "anthropic", "triage", "deep_scan", "fast", "catalog"],
}

CLOUD_ANTHROPIC_OPUS = {
    "catalog_id": "anthropic_claude_opus",
    "category": CatalogCategory.CLOUD,
    "display_name": "Claude Opus 4",
    "description": "Anthropic Claude Opus 4 - most capable model for deep analysis and judging. Requires API key.",
    "provider_id": "anthropic",
    "model_type": "anthropic_cloud",
    "model_name": "claude-opus-4-20250514",
    "task_type": "text-generation",
    "roles": ["deep_scan", "judge"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "anthropic", "deep_scan", "judge", "catalog"],
}

CLOUD_GOOGLE_GEMINI_PRO = {
    "catalog_id": "google_gemini_pro",
    "category": CatalogCategory.CLOUD,
    "display_name": "Google Gemini 1.5 Pro",
    "description": "Google Gemini 1.5 Pro for deep security analysis. Requires API key.",
    "provider_id": "google",
    "model_type": "google_cloud",
    "model_name": "gemini-1.5-pro",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "google", "deep_scan", "catalog"],
}

CLOUD_GOOGLE_GEMINI_FLASH = {
    "catalog_id": "google_gemini_flash",
    "category": CatalogCategory.CLOUD,
    "display_name": "Google Gemini 1.5 Flash",
    "description": "Google Gemini 1.5 Flash - fast and cost-effective. Requires API key.",
    "provider_id": "google",
    "model_type": "google_cloud",
    "model_name": "gemini-1.5-flash",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 4096,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "google", "triage", "deep_scan", "fast", "catalog"],
}

CLOUD_GOOGLE_GEMINI_25_FLASH = {
    "catalog_id": "google_gemini_25_flash",
    "category": CatalogCategory.CLOUD,
    "display_name": "Google Gemini 2.5 Flash",
    "description": "Google Gemini 2.5 Flash - latest generation, fast and capable. Requires API key.",
    "provider_id": "google",
    "model_type": "google_cloud",
    "model_name": "gemini-2.5-flash",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "max_tokens": 65536,
        "temperature": 0.1,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": True,
    "requires_artifact": False,
    "tags": ["cloud", "google", "triage", "deep_scan", "fast", "catalog"],
}


# =============================================================================
# Classic ML Model Definitions (sklearn, joblib)
# =============================================================================

ML_KAGGLE_RF_CFUNCTIONS = {
    "catalog_id": "kaggle_rf_cfunctions",
    "category": CatalogCategory.CLASSIC_ML,
    "display_name": "Kaggle RF C-Functions Predictor",
    "description": "Random Forest classifier trained on C/C++ functions for security vulnerability prediction. Fast CPU-based inference.",
    "provider_id": "tool_ml",
    "model_type": "tool_ml",
    "model_name": "kaggle_rf_cfunctions",
    "tool_id": "kaggle_rf_cfunctions",
    "task_type": "classification",
    "roles": ["triage"],
    "parser_id": "tool_result",
    "parser_config": {},
    "settings": {
        "model_url": "https://github.com/canoztas/aegis-models/releases/download/c_security_model/c_security_model.pkl",
        "model_type": "kaggle_rf_cfunctions",
        "threshold": 0.5,
    },
    "size_mb": 50,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["ml", "sklearn", "random_forest", "triage", "fast", "cpu", "catalog"],
}


# =============================================================================
# Ollama Model Definitions (local LLM via Ollama)
# =============================================================================

OLLAMA_QWEN25_CODER_7B = {
    "catalog_id": "ollama_qwen25_coder_7b",
    "category": CatalogCategory.OLLAMA,
    "display_name": "Qwen 2.5 Coder 7B (Ollama)",
    "description": "Qwen 2.5 Coder 7B via Ollama. Requires Ollama installed and model pulled.",
    "provider_id": "ollama",
    "model_type": "ollama_local",
    "model_name": "qwen2.5-coder:7b",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "temperature": 0.2,
        "num_predict": 1024,
    },
    "size_mb": 4500,
    "requires_gpu": False,  # Can run on CPU
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["ollama", "local", "deep_scan", "catalog"],
}

OLLAMA_CODELLAMA_7B = {
    "catalog_id": "ollama_codellama_7b",
    "category": CatalogCategory.OLLAMA,
    "display_name": "CodeLlama 7B (Ollama)",
    "description": "Meta CodeLlama 7B via Ollama. Good for code understanding tasks.",
    "provider_id": "ollama",
    "model_type": "ollama_local",
    "model_name": "codellama:7b",
    "task_type": "text-generation",
    "roles": ["deep_scan"],
    "parser_id": "json_schema",
    "parser_config": {},
    "settings": {
        "temperature": 0.2,
        "num_predict": 1024,
    },
    "size_mb": 3800,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["ollama", "local", "deep_scan", "catalog"],
}


# =============================================================================
# Agentic Model Definitions (Claude Code CLI-based scanning)
# =============================================================================

AGENTIC_CLAUDE_CODE_SECURITY_QUICK = {
    "catalog_id": "claude_code_security_quick",
    "category": CatalogCategory.AGENTIC,
    "display_name": "Claude Code Security (Quick)",
    "description": "Fast AI-powered vulnerability scan using Claude Code CLI with Sonnet model. Lower latency, good for triage.",
    "provider_id": "claude_code_security",
    "model_type": "claude_code",
    "model_name": "sonnet",
    "task_type": "text-generation",
    "roles": ["triage", "deep_scan"],
    "parser_id": "claude_code_security",
    "parser_config": {},
    "settings": {
        "claude_model": "sonnet",
        "max_turns": 1,
        "timeout": 300,
        "skip_permissions": True,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["agentic", "claude-code", "triage", "deep_scan", "fast", "catalog"],
}

AGENTIC_CLAUDE_CODE_SECURITY_DEEP = {
    "catalog_id": "claude_code_security_deep",
    "category": CatalogCategory.AGENTIC,
    "display_name": "Claude Code Security (Deep)",
    "description": "Thorough AI-powered vulnerability scan using Claude Code CLI with Opus model. More turns, deeper analysis.",
    "provider_id": "claude_code_security",
    "model_type": "claude_code",
    "model_name": "opus",
    "task_type": "text-generation",
    "roles": ["deep_scan", "judge"],
    "parser_id": "claude_code_security",
    "parser_config": {},
    "settings": {
        "claude_model": "opus",
        "max_turns": 1,
        "timeout": 300,
        "skip_permissions": True,
    },
    "size_mb": 0,
    "requires_gpu": False,
    "requires_api_key": False,
    "requires_artifact": False,
    "tags": ["agentic", "claude-code", "deep_scan", "judge", "thorough", "catalog"],
}


# =============================================================================
# Master Catalog List
# =============================================================================

MODEL_CATALOG: List[Dict[str, Any]] = [
    # HuggingFace Classifiers (Triage)
    HF_CODEBERT_INSECURE,
    HF_CODEBERT_PRIMEVUL,
    HF_VULBERTA_DEVIGN,
    HF_UNIXCODER_PRIMEVUL,
    # HuggingFace Generative (Deep Scan)
    HF_CODEASTRA_7B,
    HF_QWEN25_CODER_7B,
    HF_VULNLLM_R_7B,
    HF_DEEPSEEK_CODER_V2_LITE,
    HF_STARCODER2_15B,
    HF_PHI35_MINI,
    HF_QWEN25_05B,
    # Cloud Models
    CLOUD_OPENAI_GPT4O,
    CLOUD_OPENAI_GPT4O_MINI,
    CLOUD_ANTHROPIC_SONNET,
    CLOUD_ANTHROPIC_HAIKU_35,
    CLOUD_ANTHROPIC_OPUS,
    CLOUD_GOOGLE_GEMINI_PRO,
    CLOUD_GOOGLE_GEMINI_FLASH,
    CLOUD_GOOGLE_GEMINI_25_FLASH,
    # Ollama Models
    OLLAMA_QWEN25_CODER_7B,
    OLLAMA_CODELLAMA_7B,
    # Classic ML Models
    ML_KAGGLE_RF_CFUNCTIONS,
    # Agentic Models
    AGENTIC_CLAUDE_CODE_SECURITY_QUICK,
    AGENTIC_CLAUDE_CODE_SECURITY_DEEP,
]

# Index by catalog_id for fast lookup
CATALOG_BY_ID: Dict[str, Dict[str, Any]] = {
    entry["catalog_id"]: entry for entry in MODEL_CATALOG
}

# Alias mapping for backward compatibility with existing preset IDs
CATALOG_ALIASES: Dict[str, str] = {
    # CodeBERT Insecure
    "codebert_insecure": "codebert_insecure",
    "mrm8488/codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
    "codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
    # CodeBERT PrimeVul
    "codebert_primevul": "codebert_primevul",
    "mahdin70/codebert-primevul-bigvul": "codebert_primevul",
    "codebert-primevul-bigvul": "codebert_primevul",
    # VulBERTa
    "vulberta_devign": "vulberta_devign",
    "claudios/vulberta-mlp-devign": "vulberta_devign",
    # UnixCoder
    "unixcoder_primevul": "unixcoder_primevul",
    "mahdin70/unixcoder-primevul-bigvul": "unixcoder_primevul",
    # CodeAstra
    "codeastra_7b": "codeastra_7b",
    "rootxhacker/codeastra-7b": "codeastra_7b",
    # Qwen Coder
    "qwen25_coder_7b": "qwen25_coder_7b",
    "qwen/qwen2.5-coder-7b-instruct": "qwen25_coder_7b",
    # DeepSeek
    "deepseek_coder_v2_lite": "deepseek_coder_v2_lite",
    "deepseek-ai/deepseek-coder-v2-lite-instruct": "deepseek_coder_v2_lite",
    # StarCoder2
    "starcoder2_15b": "starcoder2_15b",
    "bigcode/starcoder2-15b-instruct-v0.1": "starcoder2_15b",
    # Phi
    "phi35_mini": "phi35_mini",
    "microsoft/phi-3.5-mini-instruct": "phi35_mini",
    # Qwen 0.5B
    "qwen25_05b": "qwen25_05b",
    "qwen/qwen2.5-0.5b-instruct": "qwen25_05b",
    # VulnLLM-R
    "vulnllm_r_7b": "vulnllm_r_7b",
    "ucsb-surfi/vulnllm-r-7b": "vulnllm_r_7b",
    "vulnllm-r-7b": "vulnllm_r_7b",
    # Anthropic Haiku 3.5
    "anthropic_claude_haiku_35": "anthropic_claude_haiku_35",
    "claude-3-5-haiku-20241022": "anthropic_claude_haiku_35",
    "claude-3.5-haiku": "anthropic_claude_haiku_35",
    # Anthropic Opus 4
    "anthropic_claude_opus": "anthropic_claude_opus",
    "claude-opus-4-20250514": "anthropic_claude_opus",
    "claude-opus-4": "anthropic_claude_opus",
    # Google Gemini 2.5 Flash
    "google_gemini_25_flash": "google_gemini_25_flash",
    "gemini-2.5-flash": "google_gemini_25_flash",
    # Classic ML
    "kaggle_rf_cfunctions": "kaggle_rf_cfunctions",
    "kaggle-rf-cfunctions": "kaggle_rf_cfunctions",
    "rf_cfunctions": "kaggle_rf_cfunctions",
    # Claude Code Security
    "claude_code_security_quick": "claude_code_security_quick",
    "claude-code-security-quick": "claude_code_security_quick",
    "claude_code_security_deep": "claude_code_security_deep",
    "claude-code-security-deep": "claude_code_security_deep",
    "claude_code_security": "claude_code_security_deep",
}


def get_catalog_entry(catalog_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a catalog entry by ID or alias.

    Args:
        catalog_id: Catalog ID or alias

    Returns:
        Catalog entry dict or None if not found
    """
    # Normalize to lowercase
    normalized = catalog_id.strip().lower()

    # Check aliases first
    canonical_id = CATALOG_ALIASES.get(normalized, normalized)

    return CATALOG_BY_ID.get(canonical_id)


def get_catalog_by_category(category: CatalogCategory) -> List[Dict[str, Any]]:
    """Get all catalog entries for a specific category."""
    return [
        entry for entry in MODEL_CATALOG
        if entry["category"] == category or entry["category"] == category.value
    ]


def get_hf_catalog_entries() -> List[Dict[str, Any]]:
    """Get HuggingFace catalog entries (for backward compatibility with /hf/presets)."""
    return get_catalog_by_category(CatalogCategory.HUGGINGFACE)


def get_ml_catalog_entries() -> List[Dict[str, Any]]:
    """Get Classic ML catalog entries."""
    return get_catalog_by_category(CatalogCategory.CLASSIC_ML)


def get_agentic_catalog_entries() -> List[Dict[str, Any]]:
    """Get Agentic catalog entries (Claude Code Security, etc.)."""
    return get_catalog_by_category(CatalogCategory.AGENTIC)
