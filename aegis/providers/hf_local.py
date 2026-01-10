"""HuggingFace Local Provider for running models locally."""

import logging
import os
from typing import Any, Dict, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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
_PeftModel = None
_PeftConfig = None

# Force transformers to prefer PyTorch; avoid pulling in TF/Keras when not needed
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

try:
    from transformers import pipeline as _hf_pipeline
    from transformers import AutoTokenizer as _auto_tok
    from transformers import AutoModelForCausalLM as _auto_causal
    import torch as _torch_module
    _transformers_available = True
    _pipeline = _hf_pipeline
    _AutoTokenizer = _auto_tok
    _AutoModelForCausalLM = _auto_causal
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
        **kwargs
    ):
        """
        Initialize HF Local Provider.

        Args:
            model_id: HuggingFace model ID (e.g., 'mrm8488/codebert-base-finetuned-detect-insecure-code')
            task_type: Pipeline task ('text-classification', 'text-generation')
            device: Device to run on ('cpu', 'cuda', or None for auto)
            **kwargs: Additional arguments for pipeline
        """
        self.model_id = model_id
        self.task_type = task_type
        self.device = device or ("cuda" if _transformers_available and _torch_cuda_available else "cpu")
        self.kwargs = kwargs
        self.adapter_id = adapter_id
        self.base_model_id = base_model_id

        self._pipeline = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single worker for model inference
        self._dependency_error = None
        self.dependency_error: Optional[str] = None

        if not _transformers_available:
            self._dependency_error = (
                "transformers/torch not installed. "
                "Install with: pip install transformers torch --upgrade"
            )
            self.dependency_error = self._dependency_error

        logger.info(f"Initialized HFLocalProvider: {model_id} on {self.device}")

    def _ensure_pipeline(self):
        """Lazy load the pipeline on first use."""
        if self._dependency_error:
            raise RuntimeError(self._dependency_error)
        if self._pipeline is None:
            logger.info(f"Loading HF model: {self.model_id} ({self.task_type})")
            try:
                safe_kwargs = dict(self.kwargs) if self.kwargs else {}

                # Strip device_map if accelerate is missing
                if not _accelerate_available and "device_map" in safe_kwargs:
                    logger.warning("Accelerate not installed; removing device_map to avoid load failure.")
                    safe_kwargs.pop("device_map", None)

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
                # Classification returns list of label/score dicts
                result = self._pipeline(prompt)
                return result

            elif self.task_type == "text-generation":
                # Merge default and provided kwargs
                gen_kwargs = {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "do_sample": True,
                    "return_full_text": False,
                    **generation_kwargs
                }

                result = self._pipeline(prompt, **gen_kwargs)

                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return result

            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

        except Exception as e:
            logger.error(f"HF model inference failed: {e}")
            raise

    def analyze_sync(self, prompt: str, **generation_kwargs) -> Any:
        """Synchronous version of analyze."""
        return self._analyze_sync(prompt, generation_kwargs)

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
}
