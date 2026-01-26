"""HuggingFace Local Provider for running models locally."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Configure HuggingFace cache directory BEFORE importing transformers
# This ensures models are downloaded under the Aegis project directory
def _configure_hf_cache():
    """Set HuggingFace cache directory to Aegis project models_cache folder."""
    # Try to get from config first, fall back to project-relative path
    cache_dir = os.environ.get("AEGIS_MODELS_CACHE")
    if not cache_dir:
        # Default: project_root/models_cache
        project_root = Path(__file__).parent.parent.parent
        cache_dir = str(project_root / "models_cache")

    # Set HuggingFace environment variables
    # HF_HOME is the primary cache location (TRANSFORMERS_CACHE is deprecated in v5)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))

    # Create directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    return cache_dir

_HF_CACHE_DIR = _configure_hf_cache()

# Lazy imports for transformers/torch
_transformers_available = False
_pipeline = None
_torch = None
_torch_cuda_available = False
_accelerate_available = False
_bitsandbytes_available = False
_peft_available = False
_AutoTokenizer = None
_AutoModelForCausalLM = None
_AutoModel = None
_PeftModel = None
_PeftConfig = None

# Force transformers to prefer PyTorch; avoid pulling in TF/Keras when not needed
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from transformers import pipeline as _hf_pipeline
    from transformers import AutoTokenizer as _auto_tok
    from transformers import AutoModelForCausalLM as _auto_causal
    from transformers import AutoModel as _auto_model
    import torch as _torch_module
    _transformers_available = True
    _pipeline = _hf_pipeline
    _AutoTokenizer = _auto_tok
    _AutoModelForCausalLM = _auto_causal
    _AutoModel = _auto_model
    _torch = _torch_module
    try:
        _torch_cuda_available = bool(
            hasattr(_torch, "cuda")
            and hasattr(_torch.cuda, "is_available")
            and _torch.cuda.is_available()
        )
    except Exception:
        _torch_cuda_available = False
    try:
        import accelerate  # noqa: F401
        _accelerate_available = True
    except Exception:
        _accelerate_available = False
    try:
        import bitsandbytes  # noqa: F401
        _bitsandbytes_available = True
    except Exception:
        _bitsandbytes_available = False
    try:
        from peft import PeftModel as _peft_model, PeftConfig as _peft_config
        _peft_available = True
        _PeftModel = _peft_model
        _PeftConfig = _peft_config
    except Exception:
        _peft_available = False
except ImportError:
    logger.warning(
        "transformers or torch not installed. "
        "HuggingFace models will not be available. "
        "Install with: pip install transformers torch"
    )


class HFLocalProvider:
    """
    Provider for running HuggingFace models locally.

    Supports:
    - text-classification (e.g., CodeBERT insecure code detector)
    - text-generation (e.g., CodeAstra)
    """

    def __init__(
        self,
        model_id: str,
        task_type: str = "text-classification",
        device: Optional[str] = None,
        adapter_id: Optional[str] = None,
        base_model_id: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        custom_loading: bool = False,
        **kwargs
    ):
        """
        Initialize HF Local Provider.

        Args:
            model_id: HuggingFace model ID (e.g., 'mrm8488/codebert-base-finetuned-detect-insecure-code')
            task_type: Pipeline task ('text-classification', 'text-generation')
            device: Device to run on ('cpu', 'cuda', or None for auto)
            adapter_id: Optional PEFT adapter ID
            base_model_id: Optional base model ID for adapters
            generation_kwargs: Generation parameters for text-generation
            max_workers: Max thread pool workers
            custom_loading: If True, use AutoModel.from_pretrained instead of pipeline
                           for models with custom architectures (e.g., MultiTaskCodeBERT)
            **kwargs: Additional arguments for pipeline/model loading
        """
        self.model_id = model_id
        self.task_type = task_type
        self.device = device or ("cuda" if _transformers_available and _torch_cuda_available else "cpu")
        self.kwargs = kwargs
        self.adapter_id = adapter_id
        self.base_model_id = base_model_id
        self.generation_kwargs = generation_kwargs or {}
        self.custom_loading = custom_loading

        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._force_manual_generate = bool(adapter_id)
        self._custom_model_loaded = False
        worker_count = 1
        try:
            if max_workers is not None:
                worker_count = max(1, int(max_workers))
        except (TypeError, ValueError):
            worker_count = 1
        self._executor = ThreadPoolExecutor(max_workers=worker_count)
        self._dependency_error = None
        self.dependency_error: Optional[str] = None

        if not _transformers_available:
            self._dependency_error = (
                "transformers/torch not installed. "
                "Install with: pip install transformers torch --upgrade"
            )
            self.dependency_error = self._dependency_error

        logger.info(f"Initialized HFLocalProvider: {model_id} on {self.device} (custom_loading={custom_loading})")

    @property
    def is_loaded(self) -> bool:
        """Check if the model pipeline or custom model is loaded."""
        return self._pipeline is not None or self._custom_model_loaded

    def is_model_cached(self) -> bool:
        """
        Check if the model files are already downloaded to the HuggingFace cache.

        Returns:
            True if model is cached locally, False if download is needed.
        """
        try:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.utils import EntryNotFoundError

            # Check for common model files that indicate model is cached
            # For most models, config.json is always present
            check_files = ["config.json", "model.safetensors", "pytorch_model.bin"]

            for filename in check_files:
                try:
                    result = try_to_load_from_cache(self.model_id, filename)
                    if result is not None:
                        # Found at least one model file in cache
                        return True
                except (EntryNotFoundError, Exception):
                    continue

            return False
        except ImportError:
            # huggingface_hub not available, assume not cached
            return False
        except Exception as e:
            logger.debug(f"Cache check failed for {self.model_id}: {e}")
            return False

    def prefetch(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Download model files to cache without loading to GPU/CPU.

        This allows pre-downloading models before a scan so they are ready
        when needed. Does NOT instantiate transformers.pipeline() or load
        weights into memory.

        Args:
            progress_callback: Optional callback function(downloaded_bytes, total_bytes)
                              called during download progress.

        Returns:
            Dict with prefetch results:
                - success: bool
                - cached: bool (whether model was already cached)
                - cache_dir: str (path to cached model files)
                - error: str or None
        """
        result = {
            "success": False,
            "cached": False,
            "cache_dir": None,
            "error": None,
        }

        # Check if already cached
        if self.is_model_cached():
            result["success"] = True
            result["cached"] = True
            result["cache_dir"] = _HF_CACHE_DIR
            logger.info(f"Model already cached: {self.model_id}")
            return result

        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Prefetching model: {self.model_id}")

            # Build kwargs for snapshot_download
            download_kwargs = {
                "repo_id": self.model_id,
                "cache_dir": os.path.join(_HF_CACHE_DIR, "hub"),
            }

            # Add progress callback if provided
            if progress_callback is not None:
                # huggingface_hub uses tqdm-style progress
                # We wrap in a simple class to convert to callback
                class ProgressCallback:
                    def __init__(self, callback):
                        self._callback = callback

                    def __call__(self, progress):
                        # progress is a dict with 'downloaded' and 'total'
                        if isinstance(progress, dict):
                            downloaded = progress.get("downloaded", 0)
                            total = progress.get("total", 0)
                            self._callback(downloaded, total)

                # Note: snapshot_download doesn't directly support callback,
                # but we can use the returned path to verify download
                pass

            # Download all model files to cache
            cache_path = snapshot_download(**download_kwargs)

            # Verify download succeeded using existing is_model_cached logic
            if self.is_model_cached():
                result["success"] = True
                result["cached"] = False  # Was not cached before, now is
                result["cache_dir"] = cache_path
                logger.info(f"Model prefetched successfully: {self.model_id} -> {cache_path}")
            else:
                result["error"] = "Download completed but model not found in cache"
                logger.warning(f"Prefetch verification failed for {self.model_id}")

        except ImportError:
            result["error"] = "huggingface_hub not installed. Install with: pip install huggingface_hub"
            logger.error(result["error"])
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Prefetch failed for {self.model_id}: {e}")

        return result

    async def prefetch_async(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Async version of prefetch - downloads model files in background.

        Args:
            progress_callback: Optional callback function(downloaded_bytes, total_bytes)

        Returns:
            Dict with prefetch results (same as prefetch())
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.prefetch,
            progress_callback
        )

    def get_model_size_estimate(self) -> int:
        """
        Estimate the model download size in MB.

        Returns:
            Estimated download size in MB, or 0 if unknown.
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            model_info = api.model_info(self.model_id)

            if hasattr(model_info, 'siblings') and model_info.siblings:
                total_size = sum(
                    (s.size or 0) for s in model_info.siblings
                    if s.rfilename and s.rfilename.endswith(
                        ('.bin', '.safetensors', '.onnx', '.pt', '.pth')
                    )
                )
                return int(total_size / (1024 * 1024))

            return 0
        except Exception as e:
            logger.debug(f"Could not get model size for {self.model_id}: {e}")
            return 0

    def warmup(self, dummy_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Pre-load the model and run a warmup inference to compile CUDA kernels.

        Args:
            dummy_input: Optional test input. If None, uses a default test string.

        Returns:
            Dict with warmup results: device, vram_mb, load_time_ms, warmup_time_ms
        """
        import time

        result = {
            "success": False,
            "device": self.device,
            "vram_mb": 0,
            "load_time_ms": 0,
            "warmup_time_ms": 0,
            "error": None,
        }

        try:
            # Time the model loading
            load_start = time.time()
            self._ensure_pipeline()
            result["load_time_ms"] = int((time.time() - load_start) * 1000)

            # Run warmup inference to compile CUDA kernels
            if dummy_input is None:
                dummy_input = "def test(): pass"

            warmup_start = time.time()
            if self.task_type == "text-classification":
                # Use _analyze_sync to handle both pipeline and custom classification
                if self.custom_loading and self._custom_model_loaded:
                    _ = self._custom_classify(dummy_input)
                else:
                    _ = self._pipeline(dummy_input, truncation=True)
            elif self.task_type == "text-generation":
                gen_kwargs = {"max_new_tokens": 10, "do_sample": False}
                _ = self._pipeline(dummy_input, **gen_kwargs)
            result["warmup_time_ms"] = int((time.time() - warmup_start) * 1000)

            # Get telemetry
            telemetry = self.get_telemetry()
            result["device"] = telemetry.get("device", self.device)
            result["vram_mb"] = telemetry.get("vram_mb", 0)
            result["success"] = True

            logger.info(
                f"Model warmup complete: {self.model_id} on {result['device']} "
                f"(load={result['load_time_ms']}ms, warmup={result['warmup_time_ms']}ms, "
                f"VRAM={result['vram_mb']}MB)"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Model warmup failed for {self.model_id}: {e}")

        return result

    def unload(self, clear_cuda_cache: bool = True) -> Dict[str, Any]:
        """
        Unload the model from memory to free GPU/CPU resources.

        Args:
            clear_cuda_cache: If True, call torch.cuda.empty_cache() after unload

        Returns:
            Dict with unload results: freed_vram_mb, success
        """
        result = {
            "success": False,
            "freed_vram_mb": 0,
            "error": None,
        }

        try:
            vram_before = 0
            if _torch and _torch_cuda_available:
                try:
                    vram_before = _torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    pass

            # Delete model and pipeline
            if self._pipeline is not None:
                if hasattr(self._pipeline, "model") and self._pipeline.model is not None:
                    # Move model to CPU first to free GPU memory
                    try:
                        self._pipeline.model.to("cpu")
                    except Exception:
                        pass
                del self._pipeline
                self._pipeline = None

            if self._model is not None:
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
                del self._model
                self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            # Reset custom model loaded flag
            self._custom_model_loaded = False

            # Clear CUDA cache
            if clear_cuda_cache and _torch and _torch_cuda_available:
                try:
                    _torch.cuda.empty_cache()
                    _torch.cuda.synchronize()
                except Exception as e:
                    logger.debug(f"CUDA cache clear failed: {e}")

            # Calculate freed memory
            if _torch and _torch_cuda_available:
                try:
                    vram_after = _torch.cuda.memory_allocated() / (1024 * 1024)
                    result["freed_vram_mb"] = int(vram_before - vram_after)
                except Exception:
                    pass

            result["success"] = True
            logger.info(f"Model unloaded: {self.model_id} (freed {result['freed_vram_mb']}MB VRAM)")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Model unload failed for {self.model_id}: {e}")

        return result

    def _validate_device_available(self) -> None:
        """Validate that the configured device is still available before loading."""
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            # Re-check CUDA availability right before loading
            cuda_available = False
            if _torch is not None:
                try:
                    cuda_available = bool(
                        hasattr(_torch, "cuda")
                        and hasattr(_torch.cuda, "is_available")
                        and _torch.cuda.is_available()
                    )
                except Exception:
                    cuda_available = False

            if not cuda_available:
                logger.warning(
                    f"CUDA device '{self.device}' requested but CUDA is not available. "
                    "Falling back to CPU."
                )
                self.device = "cpu"

    def _load_custom_model(self) -> None:
        """
        Load model using AutoModel for custom architectures.

        This is used for models that don't work with the standard pipeline() API,
        such as models with custom config classes (e.g., MultiTaskCodeBERT).
        """
        if self._custom_model_loaded:
            return

        self._validate_device_available()
        logger.info(f"Loading custom model: {self.model_id}")

        try:
            safe_kwargs = dict(self.kwargs) if self.kwargs else {}

            # Ensure trust_remote_code is set for custom architectures
            safe_kwargs["trust_remote_code"] = True

            # Normalize torch dtype if configured
            torch_dtype = safe_kwargs.get("torch_dtype")
            if torch_dtype and _torch is not None:
                if isinstance(torch_dtype, str):
                    dtype_key = torch_dtype.lower()
                    if dtype_key in ("bf16", "bfloat16"):
                        safe_kwargs["torch_dtype"] = _torch.bfloat16
                    elif dtype_key in ("fp16", "float16"):
                        safe_kwargs["torch_dtype"] = _torch.float16
                    elif dtype_key in ("fp32", "float32"):
                        safe_kwargs["torch_dtype"] = _torch.float32

            # Load tokenizer - try custom model first, fall back to base tokenizer
            tokenizer_id = safe_kwargs.pop("tokenizer_id", None)
            try:
                self._tokenizer = _AutoTokenizer.from_pretrained(
                    tokenizer_id or self.model_id,
                    trust_remote_code=True,
                )
            except Exception as tok_err:
                # For custom architectures, tokenizer may not be loadable from model
                # Fall back to common base tokenizers based on model name
                logger.warning(f"Could not load tokenizer from {self.model_id}: {tok_err}")
                fallback_tokenizers = [
                    "microsoft/codebert-base",  # For CodeBERT-based models
                    "roberta-base",             # Generic RoBERTa fallback
                ]
                tokenizer_loaded = False
                for fallback_id in fallback_tokenizers:
                    try:
                        logger.info(f"Trying fallback tokenizer: {fallback_id}")
                        self._tokenizer = _AutoTokenizer.from_pretrained(fallback_id)
                        tokenizer_loaded = True
                        logger.info(f"Loaded fallback tokenizer: {fallback_id}")
                        break
                    except Exception:
                        continue
                if not tokenizer_loaded:
                    raise RuntimeError(
                        f"Could not load tokenizer for {self.model_id}. "
                        f"Original error: {tok_err}"
                    )

            # Load model with AutoModel (for custom architectures)
            model_kwargs = {k: v for k, v in safe_kwargs.items()
                           if k not in ("trust_remote_code", "tokenizer_id")}
            model_kwargs["trust_remote_code"] = True

            self._model = _AutoModel.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

            # Move to device if not using device_map
            if "device_map" not in safe_kwargs and self.device:
                try:
                    self._model = self._model.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move model to {self.device}: {e}")

            self._model.eval()
            self._custom_model_loaded = True
            logger.info(f"Custom model loaded successfully: {self.model_id}")

        except Exception as e:
            logger.error(f"Failed to load custom model {self.model_id}: {e}")
            raise

    def _custom_classify(self, text: str) -> List[Dict[str, Any]]:
        """
        Run classification inference on a custom-loaded model.

        This handles models with non-standard architectures that can't use pipeline().
        For multi-task models (like CodeBERT-PrimeVul), extracts both vulnerability
        and CWE predictions.

        Args:
            text: Input text to classify

        Returns:
            List of classification results in pipeline-compatible format.
            For multi-task models, includes 'cwe' and 'cwe_score' fields.
        """
        if not self._custom_model_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Custom model not loaded")

        try:
            # Tokenize input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Move to model device
            target_device = None
            if hasattr(self._model, "device"):
                target_device = self._model.device
            try:
                param_device = next(self._model.parameters()).device
                target_device = param_device or target_device
            except Exception:
                pass

            if target_device is not None:
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            # Run inference
            with _torch.inference_mode():
                outputs = self._model(**inputs)

            # Check for multi-task output format (CodeBERT-PrimeVul, UnixCoder-PrimeVul)
            # These models return vul_logits and cwe_logits
            # Handle both dict and ModelOutput formats
            if isinstance(outputs, dict):
                if "vul_logits" in outputs and "cwe_logits" in outputs:
                    return self._extract_multitask_output(outputs)
            elif hasattr(outputs, "vul_logits") and hasattr(outputs, "cwe_logits"):
                return self._extract_multitask_output(outputs)

            # Extract logits - handle different output formats (dict or ModelOutput)
            logits = None
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            elif hasattr(outputs, "logits"):
                logits = outputs.logits

            # Check for last_hidden_state (dict or object)
            last_hidden_state = None
            if logits is None:
                if isinstance(outputs, dict):
                    last_hidden_state = outputs.get("last_hidden_state")
                elif hasattr(outputs, "last_hidden_state"):
                    last_hidden_state = outputs.last_hidden_state

            if logits is None and last_hidden_state is not None:
                # Some models return hidden states; use the [CLS] token
                cls_hidden = last_hidden_state[:, 0, :]
                # Check if model has a classifier head
                if hasattr(self._model, "classifier"):
                    logits = self._model.classifier(cls_hidden)
                elif hasattr(self._model, "pooler") and hasattr(self._model, "classifier"):
                    pooled = self._model.pooler(last_hidden_state)
                    logits = self._model.classifier(pooled)
                else:
                    # Fallback: assume binary classification from hidden state
                    # Take mean and sigmoid as a vulnerability score
                    score = _torch.sigmoid(cls_hidden.mean()).item()
                    return [{
                        "label": "LABEL_1" if score > 0.5 else "LABEL_0",
                        "score": score if score > 0.5 else 1 - score,
                    }]
            elif logits is None and isinstance(outputs, _torch.Tensor):
                logits = outputs

            if logits is None:
                logger.warning(f"Could not extract logits from model output: {type(outputs)}")
                return [{"label": "LABEL_0", "score": 0.5}]

            # Convert logits to probabilities
            probs = _torch.softmax(logits, dim=-1)

            # Get predicted class and score
            pred_class = probs.argmax(dim=-1).item()
            pred_score = probs[0, pred_class].item()

            # Return in pipeline-compatible format
            return [{
                "label": f"LABEL_{pred_class}",
                "score": pred_score,
            }]

        except Exception as e:
            logger.error(f"Custom classification inference failed: {e}")
            raise

    def _extract_multitask_output(self, outputs) -> List[Dict[str, Any]]:
        """
        Extract predictions from multi-task model output (CodeBERT-PrimeVul).

        Args:
            outputs: Model outputs with vul_logits and cwe_logits (dict or ModelOutput)

        Returns:
            List with classification result including CWE prediction
        """
        # Extract vulnerability logits - handle both dict and ModelOutput
        if isinstance(outputs, dict):
            vul_logits = outputs.get("vul_logits")
            cwe_logits = outputs.get("cwe_logits")
        else:
            vul_logits = outputs.vul_logits
            cwe_logits = outputs.cwe_logits

        if vul_logits is None:
            logger.warning("No vul_logits found in multi-task output")
            return [{"label": "LABEL_0", "score": 0.5}]

        vul_probs = _torch.softmax(vul_logits, dim=-1)
        vul_pred = vul_probs.argmax(dim=-1).item()
        vul_score = vul_probs[0, vul_pred].item()

        # Build result
        result = {
            "label": f"LABEL_{vul_pred}",
            "score": vul_score,
        }

        # Extract CWE prediction if vulnerable
        if vul_pred == 1 and cwe_logits is not None:  # Vulnerable
            cwe_probs = _torch.softmax(cwe_logits, dim=-1)
            cwe_pred_idx = cwe_probs.argmax(dim=-1).item()
            cwe_score = cwe_probs[0, cwe_pred_idx].item()

            # Map CWE index to CWE ID
            cwe_id = PRIMEVUL_CWE_MAPPING.get(cwe_pred_idx)
            if cwe_id:
                result["cwe"] = cwe_id
                result["cwe_score"] = cwe_score
                result["cwe_index"] = cwe_pred_idx
            else:
                # Unknown CWE index - report as generic
                result["cwe"] = f"CWE-Unknown-{cwe_pred_idx}"
                result["cwe_score"] = cwe_score
                result["cwe_index"] = cwe_pred_idx

            logger.debug(
                f"Multi-task prediction: vul={vul_pred} ({vul_score:.3f}), "
                f"cwe={result.get('cwe')} ({cwe_score:.3f})"
            )

        return [result]

    def _ensure_pipeline(self):
        """Lazy load the pipeline on first use."""
        if self._dependency_error:
            raise RuntimeError(self._dependency_error)

        # Use custom loading for models with non-standard architectures
        if self.custom_loading:
            if not self._custom_model_loaded:
                self._load_custom_model()
            return

        if self._pipeline is None:
            # Validate device availability before attempting load
            self._validate_device_available()
            logger.info(f"Loading HF model: {self.model_id} ({self.task_type})")
            try:
                safe_kwargs = dict(self.kwargs) if self.kwargs else {}

                # Normalize torch dtype if configured
                torch_dtype = safe_kwargs.get("torch_dtype")
                if torch_dtype and _torch is not None:
                    if isinstance(torch_dtype, str):
                        dtype_key = torch_dtype.lower()
                        if dtype_key in ("bf16", "bfloat16"):
                            safe_kwargs["torch_dtype"] = _torch.bfloat16
                        elif dtype_key in ("fp16", "float16"):
                            safe_kwargs["torch_dtype"] = _torch.float16
                        elif dtype_key in ("fp32", "float32"):
                            safe_kwargs["torch_dtype"] = _torch.float32

                if (
                    isinstance(self.device, str)
                    and self.device.startswith("cuda")
                    and not _torch_cuda_available
                ):
                    raise RuntimeError("CUDA requested but torch reports no CUDA support.")

                # Strip device_map if accelerate is missing
                if not _accelerate_available and "device_map" in safe_kwargs:
                    logger.warning("Accelerate not installed; removing device_map to avoid load failure.")
                    safe_kwargs.pop("device_map", None)

                if str(self.device).startswith("cpu"):
                    if safe_kwargs.pop("load_in_4bit", None) is not None:
                        logger.warning("CPU runtime detected; removed load_in_4bit.")
                    if safe_kwargs.pop("load_in_8bit", None) is not None:
                        logger.warning("CPU runtime detected; removed load_in_8bit.")
                    if safe_kwargs.pop("quantization_config", None) is not None:
                        logger.warning("CPU runtime detected; removed quantization_config.")

                # Strip quantization flags if bitsandbytes missing
                if not _bitsandbytes_available:
                    for key in ["load_in_4bit", "load_in_8bit"]:
                        if safe_kwargs.pop(key, None) is not None:
                            logger.warning("bitsandbytes not installed; removed quantization flag %s", key)
                    # Also drop quantization_config if present
                    if safe_kwargs.pop("quantization_config", None) is not None:
                        logger.warning("bitsandbytes not installed; removed quantization_config")

                # Adapter flow (PEFT)
                if self.adapter_id:
                    if not _peft_available:
                        raise RuntimeError("PEFT not installed. Install with: pip install peft")

                    base_model_id = self.base_model_id
                    if not base_model_id:
                        cfg = _PeftConfig.from_pretrained(self.adapter_id)
                        base_model_id = cfg.base_model_name_or_path

                    tokenizer = _AutoTokenizer.from_pretrained(
                        base_model_id,
                        trust_remote_code=True,
                    )

                    device_arg = None if "device_map" in safe_kwargs else self.device
                    base_kwargs = {
                        "return_dict": True,
                        "trust_remote_code": True,
                        **safe_kwargs,
                    }
                    if device_arg and "device_map" not in safe_kwargs:
                        base_kwargs["device_map"] = device_arg

                    base_model = _AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        **base_kwargs,
                    )
                    peft_model = _PeftModel.from_pretrained(base_model, self.adapter_id)

                    self._pipeline = _pipeline(
                        self.task_type,
                        model=peft_model,
                        tokenizer=tokenizer,
                        framework="pt",
                    )
                else:
                    # If device_map is provided (Accelerate flow), do not pass device argument
                    device_arg = None if "device_map" in safe_kwargs else self.device

                    pipeline_kwargs = {
                        "task": self.task_type,
                        "model": self.model_id,
                        "framework": "pt",  # force PyTorch to avoid Keras/TensorFlow path
                        **safe_kwargs,
                    }
                    if device_arg:
                        pipeline_kwargs["device"] = device_arg

                    self._pipeline = _pipeline(**pipeline_kwargs)
                self._model = getattr(self._pipeline, "model", None)
                self._tokenizer = getattr(self._pipeline, "tokenizer", None)
                logger.info(f"Model loaded successfully: {self.model_id}")
            except Exception as e:
                err_str = str(e)
                if "Keras is Keras 3" in err_str or "tf-keras" in err_str:
                    friendly = (
                        "Transformers attempted to load TensorFlow/Keras. "
                        "Install a compatible TF shim with: pip install tf-keras "
                        "or force PyTorch by ensuring torch is installed."
                    )
                    logger.error(f"Failed to load model {self.model_id}: {friendly}")
                    raise RuntimeError(friendly)
                if "torch" in err_str.lower() and "not found" in err_str.lower():
                    friendly = (
                        "PyTorch is required for local HuggingFace models. "
                        "Install CPU wheel: pip install torch --index-url "
                        "https://download.pytorch.org/whl/cpu"
                    )
                    logger.error(f"Failed to load model {self.model_id}: {friendly}")
                    raise RuntimeError(friendly)
                logger.error(f"Failed to load model {self.model_id}: {e}")
                raise

    def _normalize_generation_kwargs(self, generation_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        }
        if self.generation_kwargs:
            gen_kwargs.update(self.generation_kwargs)
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        max_new = gen_kwargs.get("max_new_tokens")
        if not isinstance(max_new, int) or max_new <= 0:
            gen_kwargs["max_new_tokens"] = 256
        min_new = gen_kwargs.get("min_new_tokens")
        if isinstance(min_new, (int, float)):
            min_new = int(min_new)
            if min_new <= 0:
                gen_kwargs.pop("min_new_tokens", None)
            else:
                if min_new > gen_kwargs["max_new_tokens"]:
                    min_new = max(1, gen_kwargs["max_new_tokens"] // 2)
                gen_kwargs["min_new_tokens"] = min_new

        temperature = gen_kwargs.get("temperature")
        if isinstance(temperature, (int, float)) and temperature < 0:
            gen_kwargs["temperature"] = 0.0

        top_p = gen_kwargs.get("top_p")
        if isinstance(top_p, (int, float)) and (top_p <= 0 or top_p > 1):
            gen_kwargs.pop("top_p", None)

        return gen_kwargs

    def _apply_chat_template(self, prompt: str) -> str:
        """
        Apply chat template for instruct models.

        For instruct/chat models, the tokenizer's chat template must be used
        to properly format the prompt so the model understands it's an instruction.

        Args:
            prompt: Raw prompt text

        Returns:
            Formatted prompt with chat template applied, or original if no template
        """
        if self._tokenizer is None:
            return prompt

        try:
            # Check if tokenizer has a chat template
            if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
                # Format as a simple user message
                messages = [{"role": "user", "content": prompt}]
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            elif hasattr(self._tokenizer, 'apply_chat_template'):
                # Try applying even without explicit template (uses default)
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return formatted
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not apply chat template: {e}")

        return prompt

    def _manual_generate(self, prompt: str, gen_kwargs: Dict[str, Any]) -> Optional[str]:
        if not _torch or self._model is None or self._tokenizer is None:
            return None

        try:
            # Apply chat template for instruct models
            formatted_prompt = self._apply_chat_template(prompt)
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            target_device = None
            if hasattr(self._model, "device"):
                target_device = self._model.device
            try:
                param_device = next(self._model.parameters()).device
                target_device = param_device or target_device
            except Exception:
                pass
            if target_device is not None:
                inputs = {k: v.to(target_device) for k, v in inputs.items()}

            safe_kwargs = dict(gen_kwargs)
            for key in ("return_full_text", "return_text", "clean_up_tokenization_spaces"):
                safe_kwargs.pop(key, None)

            if self._tokenizer.eos_token_id is not None:
                safe_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)
                safe_kwargs.setdefault("pad_token_id", self._tokenizer.eos_token_id)

            if "max_new_tokens" not in safe_kwargs or safe_kwargs["max_new_tokens"] is None:
                safe_kwargs["max_new_tokens"] = 256

            with _torch.inference_mode():
                output_ids = self._model.generate(**inputs, **safe_kwargs)

            input_len = inputs["input_ids"].shape[-1]
            generated = output_ids[0][input_len:]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            if not text.strip():
                text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text
        except Exception as e:
            logger.warning("Manual generate fallback failed: %s", e)
            return None

    @staticmethod
    def _looks_like_json(text: Optional[str]) -> bool:
        if not text or not isinstance(text, str):
            return False
        stripped = text.strip()
        if not stripped:
            return False
        return "{" in stripped and "}" in stripped

    @staticmethod
    def _json_retry_prompt(prompt: str) -> str:
        suffix = (
            "\n\nReturn ONLY valid JSON now. The output must start with '{' and end with '}'.\n"
            "If there are no findings, return {\"findings\": []}."
        )
        return f"{prompt.rstrip()}{suffix}"

    def _generate_with_json_retry(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """
        Generate text with automatic JSON retry logic.

        Attempts generation with fallbacks:
        1. Initial generation
        2. Strict mode (no sampling)
        3. JSON repair prompt with strict mode

        Args:
            prompt: Input prompt
            gen_kwargs: Generation kwargs

        Returns:
            Generated text (may be empty if all attempts fail)
        """
        text = self._manual_generate(prompt, gen_kwargs)

        # If empty or not JSON, retry with strict settings
        if text is None or not str(text).strip() or not self._looks_like_json(text):
            strict_kwargs = dict(gen_kwargs)
            strict_kwargs["do_sample"] = False
            strict_kwargs["temperature"] = 0.0
            strict_kwargs.pop("top_p", None)

            retry_text = self._manual_generate(prompt, strict_kwargs)
            if retry_text and str(retry_text).strip():
                text = retry_text

            # If still not JSON, try repair prompt
            if text is None or not str(text).strip() or not self._looks_like_json(text):
                repair_prompt = self._json_retry_prompt(prompt)
                repair_text = self._manual_generate(repair_prompt, strict_kwargs)
                if repair_text and str(repair_text).strip():
                    text = repair_text

        return text if text is not None else ""

    def _pipeline_with_json_retry(self, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
        """
        Run pipeline generation with JSON retry fallbacks.

        Attempts:
        1. Pipeline generation
        2. Pipeline with min_new_tokens
        3. Manual generate fallback
        4. JSON repair prompt

        Args:
            prompt: Input prompt
            gen_kwargs: Generation kwargs

        Returns:
            Generated text
        """
        # Apply chat template for instruct models
        formatted_prompt = self._apply_chat_template(prompt)
        result = self._pipeline(formatted_prompt, **gen_kwargs)
        text = self._extract_generated_text(result, prompt)

        # Retry with min_new_tokens if empty
        if text is None or not str(text).strip():
            fallback_kwargs = dict(gen_kwargs)
            min_new_tokens = fallback_kwargs.get("min_new_tokens")
            if not isinstance(min_new_tokens, int) or min_new_tokens <= 0:
                max_new = fallback_kwargs.get("max_new_tokens")
                fallback_kwargs["min_new_tokens"] = min(32, max_new) if isinstance(max_new, int) and max_new > 0 else 32
            result = self._pipeline(formatted_prompt, **fallback_kwargs)
            text = self._extract_generated_text(result, formatted_prompt)

        # Fallback to manual generate with JSON retry
        if text is None or not str(text).strip() or not self._looks_like_json(text):
            manual = self._manual_generate(prompt, gen_kwargs)
            if manual and str(manual).strip():
                text = manual

            if text is None or not str(text).strip() or not self._looks_like_json(text):
                strict_kwargs = dict(gen_kwargs)
                strict_kwargs["do_sample"] = False
                strict_kwargs["temperature"] = 0.0
                strict_kwargs.pop("top_p", None)
                repair_prompt = self._json_retry_prompt(prompt)
                manual = self._manual_generate(repair_prompt, strict_kwargs)
                if manual and str(manual).strip():
                    text = manual

        return text if text is not None else result

    def _extract_generated_text(self, output: Any, prompt: str) -> Optional[str]:
        """Extract text from HF outputs and avoid over-stripping."""
        def _extract_text(result: Any) -> Optional[str]:
            if isinstance(result, list) and result:
                item = result[0]
                if isinstance(item, dict):
                    if "generated_text" in item:
                        return item.get("generated_text")
                    if "text" in item:
                        return item.get("text")
                if isinstance(item, str):
                    return item
            if isinstance(result, dict):
                return result.get("generated_text") or result.get("text")
            if isinstance(result, str):
                return result
            return None

        text = _extract_text(output)
        if not text:
            return text

        prompt_stripped = prompt.strip()
        text_stripped = text.strip()
        if text_stripped == prompt_stripped:
            return text
        if text_stripped.startswith(prompt_stripped):
            remainder = text_stripped[len(prompt_stripped):].lstrip()
            if remainder:
                return remainder
        return text

    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **generation_kwargs
    ) -> Any:
        """
        Run analysis asynchronously.

        Args:
            prompt: Input text to analyze
            context: Optional context
            **generation_kwargs: Additional generation parameters

        Returns:
            Model output (format depends on task_type)
        """
        # Run pipeline in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._analyze_sync,
            prompt,
            generation_kwargs
        )

    def _analyze_sync(self, prompt: str, generation_kwargs: Dict[str, Any]) -> Any:
        """Synchronous analysis (runs in thread pool)."""
        self._ensure_pipeline()

        try:
            if self.task_type == "text-classification":
                # Use custom classification for models with custom architectures
                if self.custom_loading and self._custom_model_loaded:
                    return self._custom_classify(prompt)
                # Enable truncation to handle inputs longer than model's max length
                return self._pipeline(prompt, truncation=True)

            elif self.task_type == "text-generation":
                gen_kwargs = self._normalize_generation_kwargs(generation_kwargs)

                if self._force_manual_generate:
                    return self._generate_with_json_retry(prompt, gen_kwargs)

                return self._pipeline_with_json_retry(prompt, gen_kwargs)

            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

        except Exception as e:
            logger.error(f"HF model inference failed: {e}")
            raise

    def analyze_sync(self, prompt: str, **generation_kwargs) -> Any:
        """Synchronous version of analyze."""
        return self._analyze_sync(prompt, generation_kwargs)

    async def analyze_batch(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
        **generation_kwargs
    ) -> Any:
        """Batch analysis for multiple prompts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._analyze_batch_sync,
            prompts,
            generation_kwargs
        )

    def _analyze_batch_sync(self, prompts: List[str], generation_kwargs: Dict[str, Any]) -> Any:
        """Synchronous batch analysis (runs in thread pool)."""
        self._ensure_pipeline()

        if not isinstance(prompts, list):
            return [self._analyze_sync(prompts, generation_kwargs)]

        try:
            if self.task_type == "text-classification":
                # Use custom classification for models with custom architectures
                if self.custom_loading and self._custom_model_loaded:
                    return [self._custom_classify(p) for p in prompts]
                # Enable truncation to handle inputs longer than model's max length
                outputs = self._pipeline(prompts, truncation=True)
                if isinstance(outputs, dict):
                    return [[outputs]]
                if isinstance(outputs, list):
                    normalized = []
                    for item in outputs:
                        if isinstance(item, dict):
                            normalized.append([item])
                        else:
                            normalized.append(item)
                    return normalized
                return outputs

            if self.task_type == "text-generation":
                gen_kwargs = self._normalize_generation_kwargs(generation_kwargs)

                if self._force_manual_generate:
                    return [self._generate_with_json_retry(p, gen_kwargs) for p in prompts]

                # Batch pipeline call with individual fallbacks
                result = self._pipeline(prompts, **gen_kwargs)
                outputs = []
                for prompt, output in zip(prompts, result):
                    text = self._extract_generated_text(output, prompt)
                    # If initial extraction failed, use full retry logic for this prompt
                    if text is None or not str(text).strip() or not self._looks_like_json(text):
                        text = self._pipeline_with_json_retry(prompt, gen_kwargs)
                    outputs.append(text if text is not None else output)
                return outputs

            raise ValueError(f"Unsupported task type: {self.task_type}")

        except Exception as e:
            logger.error(f"HF model batch inference failed: {e}")
            raise

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get model telemetry information.

        Returns:
            Dictionary with device, VRAM, quantization, precision, etc.
        """
        telemetry = {
            "device": self.device,
            "vram_mb": 0,
            "quantization": None,
            "precision": None,
            "load_time_ms": 0,
        }

        try:
            # Get actual device from pipeline if loaded
            if self._pipeline and hasattr(self._pipeline, "device"):
                device = self._pipeline.device
                telemetry["device"] = str(device)
            elif self._pipeline and hasattr(self._pipeline, "model"):
                model = self._pipeline.model
                if hasattr(model, "device"):
                    telemetry["device"] = str(model.device)

            # Get VRAM usage if CUDA available
            if _torch and _torch_cuda_available and "cuda" in str(telemetry["device"]):
                try:
                    if self._pipeline and hasattr(self._pipeline, "model"):
                        model = self._pipeline.model
                        # Get memory allocated to this model
                        if hasattr(_torch, "cuda"):
                            # Get current allocated memory
                            allocated_mb = _torch.cuda.memory_allocated() / (1024 * 1024)
                            telemetry["vram_mb"] = int(allocated_mb)
                except Exception as e:
                    logger.debug(f"Failed to get VRAM usage: {e}")

            # Detect quantization from kwargs
            if self.kwargs:
                if self.kwargs.get("load_in_4bit"):
                    telemetry["quantization"] = "int4"
                elif self.kwargs.get("load_in_8bit"):
                    telemetry["quantization"] = "int8"
                elif self.kwargs.get("quantization_config"):
                    telemetry["quantization"] = "quantized"

                # Detect precision from torch_dtype
                torch_dtype = self.kwargs.get("torch_dtype")
                if torch_dtype:
                    if _torch:
                        if torch_dtype == _torch.bfloat16:
                            telemetry["precision"] = "bfloat16"
                        elif torch_dtype == _torch.float16:
                            telemetry["precision"] = "float16"
                        elif torch_dtype == _torch.float32:
                            telemetry["precision"] = "float32"
                    elif isinstance(torch_dtype, str):
                        telemetry["precision"] = torch_dtype.lower()

        except Exception as e:
            logger.debug(f"Error collecting telemetry: {e}")

        return telemetry

    def close(self) -> None:
        """Release background executor resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Factory function
def create_hf_provider(
    model_id: str,
    task_type: str = "text-classification",
    **kwargs
) -> Optional[HFLocalProvider]:
    """
    Factory to create HF provider with error handling.

    Args:
        model_id: HuggingFace model ID
        task_type: Pipeline task type
        **kwargs: Additional arguments

    Returns:
        HFLocalProvider instance or None on error
    """
    try:
        return HFLocalProvider(model_id, task_type, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create HF provider: {e}")
        return None


# Pre-configured models
CODEBERT_INSECURE = {
    "model_id": "mrm8488/codebert-base-finetuned-detect-insecure-code",
    "task_type": "text-classification",
    "name": "CodeBERT Insecure Code Detector",
    "description": "Binary classifier for detecting potentially insecure code",
}

CODEASTRA_7B = {
    "model_id": "rootxhacker/CodeAstra-7B",
    "task_type": "text-generation",
    "name": "CodeAstra 7B",
    "description": "Generative model for code analysis and vulnerability detection",
    "adapter_id": "rootxhacker/CodeAstra-7B",
    "base_model_id": "mistralai/Mistral-7B-Instruct-v0.2",
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
}

# CWE Index to ID mapping for CodeBERT-PrimeVul-BigVul and UnixCoder-PrimeVul models
# These multi-task models output a CWE class index (0-134) instead of CWE ID strings.
# This mapping converts model output indices to actual CWE IDs.
#
# IMPORTANT: This mapping was created based on common CWEs from BigVul/PrimeVul datasets.
# The actual order depends on how sklearn.LabelEncoder sorted the CWE IDs during training.
# To regenerate the accurate mapping, run: python scripts/extract_cwe_mapping.py
#
# For indices not in this mapping, the parser will return "CWE-Unknown-{index}"
PRIMEVUL_CWE_MAPPING = {
    0: None,  # Not vulnerable / no CWE
    1: "CWE-119",  # Improper Restriction of Operations within the Bounds of a Memory Buffer
    2: "CWE-120",  # Buffer Copy without Checking Size of Input (Classic Buffer Overflow)
    3: "CWE-125",  # Out-of-bounds Read
    4: "CWE-787",  # Out-of-bounds Write
    5: "CWE-416",  # Use After Free
    6: "CWE-476",  # NULL Pointer Dereference
    7: "CWE-190",  # Integer Overflow or Wraparound
    8: "CWE-20",   # Improper Input Validation
    9: "CWE-200",  # Exposure of Sensitive Information
    10: "CWE-401", # Missing Release of Memory after Effective Lifetime (Memory Leak)
    11: "CWE-122", # Heap-based Buffer Overflow
    12: "CWE-121", # Stack-based Buffer Overflow
    13: "CWE-399", # Resource Management Errors
    14: "CWE-264", # Permissions, Privileges, and Access Controls
    15: "CWE-189", # Numeric Errors
    16: "CWE-362", # Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)
    17: "CWE-772", # Missing Release of Resource after Effective Lifetime
    18: "CWE-617", # Reachable Assertion
    19: "CWE-369", # Divide By Zero
    20: "CWE-400", # Uncontrolled Resource Consumption
    21: "CWE-415", # Double Free
    22: "CWE-22",  # Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)
    23: "CWE-79",  # Cross-site Scripting (XSS)
    24: "CWE-89",  # SQL Injection
    25: "CWE-78",  # OS Command Injection
    26: "CWE-94",  # Code Injection
    27: "CWE-287", # Improper Authentication
    28: "CWE-284", # Improper Access Control
    29: "CWE-310", # Cryptographic Issues
    30: "CWE-311", # Missing Encryption of Sensitive Data
    31: "CWE-327", # Use of a Broken or Risky Cryptographic Algorithm
    32: "CWE-330", # Use of Insufficiently Random Values
    33: "CWE-352", # Cross-Site Request Forgery (CSRF)
    34: "CWE-434", # Unrestricted Upload of File with Dangerous Type
    35: "CWE-502", # Deserialization of Untrusted Data
    36: "CWE-601", # URL Redirection to Untrusted Site (Open Redirect)
    37: "CWE-611", # Improper Restriction of XML External Entity Reference
    38: "CWE-732", # Incorrect Permission Assignment for Critical Resource
    39: "CWE-798", # Use of Hard-coded Credentials
    40: "CWE-862", # Missing Authorization
    41: "CWE-863", # Incorrect Authorization
    42: "CWE-918", # Server-Side Request Forgery (SSRF)
    43: "CWE-77",  # Command Injection
    44: "CWE-74",  # Injection
    45: "CWE-129", # Improper Validation of Array Index
    46: "CWE-131", # Incorrect Calculation of Buffer Size
    47: "CWE-134", # Use of Externally-Controlled Format String
    48: "CWE-191", # Integer Underflow
    49: "CWE-193", # Off-by-one Error
    50: "CWE-252", # Unchecked Return Value
    51: "CWE-295", # Improper Certificate Validation
    52: "CWE-306", # Missing Authentication for Critical Function
    53: "CWE-319", # Cleartext Transmission of Sensitive Information
    54: "CWE-326", # Inadequate Encryption Strength
    55: "CWE-347", # Improper Verification of Cryptographic Signature
    56: "CWE-384", # Session Fixation
    57: "CWE-404", # Improper Resource Shutdown or Release
    58: "CWE-426", # Untrusted Search Path
    59: "CWE-427", # Uncontrolled Search Path Element
    60: "CWE-457", # Use of Uninitialized Variable
    61: "CWE-459", # Incomplete Cleanup
    62: "CWE-471", # Modification of Assumed-Immutable Data
    63: "CWE-532", # Insertion of Sensitive Information into Log File
    64: "CWE-552", # Files or Directories Accessible to External Parties
    65: "CWE-565", # Reliance on Cookies without Validation and Integrity Checking
    66: "CWE-674", # Uncontrolled Recursion
    67: "CWE-682", # Incorrect Calculation
    68: "CWE-704", # Incorrect Type Conversion or Cast
    69: "CWE-754", # Improper Check for Unusual or Exceptional Conditions
    70: "CWE-755", # Improper Handling of Exceptional Conditions
    71: "CWE-763", # Release of Invalid Pointer or Reference
    72: "CWE-770", # Allocation of Resources Without Limits or Throttling
    73: "CWE-824", # Access of Uninitialized Pointer
    74: "CWE-835", # Loop with Unreachable Exit Condition (Infinite Loop)
    75: "CWE-908", # Use of Uninitialized Resource
    76: "CWE-909", # Missing Initialization of Resource
    # Remaining indices map to less common CWEs or unknown
    # The parser will use this as a best-effort mapping
}

# Reverse mapping for lookup
CWE_TO_PRIMEVUL_INDEX = {v: k for k, v in PRIMEVUL_CWE_MAPPING.items() if v is not None}


# CodeBERT-PrimeVul-BigVul has a custom MultiTaskCodeBERT architecture
# that requires custom loading (AutoModel with trust_remote_code=True)
# The tokenizer must be loaded from the base CodeBERT model
CODEBERT_PRIMEVUL = {
    "model_id": "mahdin70/CodeBERT-PrimeVul-BigVul",
    "task_type": "text-classification",
    "name": "CodeBERT PrimeVul-BigVul",
    "description": "CodeBERT-based vulnerability detector trained on PrimeVul and BigVul datasets (custom architecture)",
    "custom_loading": True,  # Requires AutoModel loading, not pipeline()
    "hf_kwargs": {
        "trust_remote_code": True,
        "tokenizer_id": "microsoft/codebert-base",  # Custom model can't load its own tokenizer
    },
}

VULBERTA_DEVIGN = {
    "model_id": "claudios/VulBERTa-MLP-Devign",
    "task_type": "text-classification",
    "name": "VulBERTa MLP Devign",
    "description": "RoBERTa-based C/C++ vulnerability detector trained on CodeXGLUE Devign (F1: 0.57)",
    "custom_loading": True,  # Custom model architecture
    "hf_kwargs": {
        "trust_remote_code": True,  # Required for custom tokenizer
        "tokenizer_id": "roberta-base",  # Fallback tokenizer if libclang not available
    },
}

UNIXCODER_PRIMEVUL = {
    "model_id": "mahdin70/UnixCoder-Primevul-BigVul",
    "task_type": "text-classification",
    "name": "UnixCoder PrimeVul-BigVul",
    "description": "UniXcoder-based vulnerability detector trained on BigVul and PrimeVul datasets (custom architecture)",
    "custom_loading": True,  # Requires AutoModel loading, not pipeline()
    "hf_kwargs": {
        "trust_remote_code": True,
        "tokenizer_id": "microsoft/unixcoder-base",  # Custom model can't load its own tokenizer
    },
}

QWEN25_CODER_7B = {
    "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "task_type": "text-generation",
    "name": "Qwen 2.5 Coder 7B Instruct",
    "description": "Qwen 2.5 code-specialized 7B model for code analysis and vulnerability detection",
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
}

DEEPSEEK_CODER_V2_LITE = {
    "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "task_type": "text-generation",
    "name": "DeepSeek Coder V2 Lite Instruct",
    "description": "DeepSeek Coder V2 Lite (16B) - efficient code model with MoE architecture",
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
}

STARCODER2_15B = {
    "model_id": "bigcode/starcoder2-15b-instruct-v0.1",
    "task_type": "text-generation",
    "name": "StarCoder2 15B Instruct",
    "description": "StarCoder2 15B instruction-tuned model for code generation and analysis",
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
}

PHI35_MINI = {
    "model_id": "microsoft/Phi-3.5-mini-instruct",
    "task_type": "text-generation",
    "name": "Phi-3.5 Mini Instruct",
    "description": "Microsoft Phi-3.5 Mini (3.8B) - compact but capable instruction model",
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
}

QWEN25_05B = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "task_type": "text-generation",
    "name": "Qwen 2.5 0.5B Instruct",
    "description": "Qwen 2.5 tiny (0.5B) - fast, lightweight model for quick analysis",
    "hf_kwargs": {
        "device_map": "auto",
        "torch_dtype": "fp16",
        "trust_remote_code": True,
    },
    "generation_kwargs": {
        "max_new_tokens": 256,  # Smaller output for tiny model
        "temperature": 0.0,     # Deterministic for better JSON compliance
        "do_sample": False,     # Greedy decoding for consistency
    },
    # Small model flag - uses simplified prompts
    "small_model": True,
}
