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
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
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
        self.generation_kwargs = generation_kwargs or {}

        self._pipeline = None
        self._model = None
        self._tokenizer = None
        self._force_manual_generate = bool(adapter_id)
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

        logger.info(f"Initialized HFLocalProvider: {model_id} on {self.device}")

    def _ensure_pipeline(self):
        """Lazy load the pipeline on first use."""
        if self._dependency_error:
            raise RuntimeError(self._dependency_error)
        if self._pipeline is None:
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

    def _manual_generate(self, prompt: str, gen_kwargs: Dict[str, Any]) -> Optional[str]:
        if not _torch or self._model is None or self._tokenizer is None:
            return None

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
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
                # Classification returns list of label/score dicts
                result = self._pipeline(prompt)
                return result

            elif self.task_type == "text-generation":
                # Merge default and provided kwargs
                gen_kwargs = self._normalize_generation_kwargs(generation_kwargs)

                if self._force_manual_generate:
                    text = self._manual_generate(prompt, gen_kwargs)
                    if text is None or not str(text).strip() or not self._looks_like_json(text):
                        strict_kwargs = dict(gen_kwargs)
                        strict_kwargs["do_sample"] = False
                        strict_kwargs["temperature"] = 0.0
                        strict_kwargs.pop("top_p", None)
                        retry_text = self._manual_generate(prompt, strict_kwargs)
                        if retry_text and str(retry_text).strip():
                            text = retry_text
                        if text is None or not str(text).strip() or not self._looks_like_json(text):
                            repair_prompt = self._json_retry_prompt(prompt)
                            repair_text = self._manual_generate(repair_prompt, strict_kwargs)
                            if repair_text and str(repair_text).strip():
                                text = repair_text
                    return text if text is not None else ""

                # Some models echo the prompt even with return_full_text=False
                result = self._pipeline(prompt, **gen_kwargs)
                text = self._extract_generated_text(result, prompt)

                # Retry once with min_new_tokens if we got an empty response
                if text is None or not str(text).strip():
                    fallback_kwargs = dict(gen_kwargs)
                    min_new_tokens = fallback_kwargs.get("min_new_tokens")
                    if not isinstance(min_new_tokens, int) or min_new_tokens <= 0:
                        max_new = fallback_kwargs.get("max_new_tokens")
                        fallback_kwargs["min_new_tokens"] = min(32, max_new) if isinstance(max_new, int) and max_new > 0 else 32
                    result = self._pipeline(prompt, **fallback_kwargs)
                    text = self._extract_generated_text(result, prompt)

                if text is None or not str(text).strip() or not self._looks_like_json(text):
                    manual = self._manual_generate(prompt, gen_kwargs)
                    if manual and str(manual).strip():
                        text = manual
                    if text is None or not str(text).strip() or not self._looks_like_json(text):
                        repair_prompt = self._json_retry_prompt(prompt)
                        strict_kwargs = dict(gen_kwargs)
                        strict_kwargs["do_sample"] = False
                        strict_kwargs["temperature"] = 0.0
                        strict_kwargs.pop("top_p", None)
                        manual = self._manual_generate(repair_prompt, strict_kwargs)
                        if manual and str(manual).strip():
                            text = manual

                return text if text is not None else result

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
                outputs = self._pipeline(prompts)
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
                    outputs = []
                    for prompt in prompts:
                        text = self._manual_generate(prompt, gen_kwargs)
                        if text is None or not str(text).strip() or not self._looks_like_json(text):
                            strict_kwargs = dict(gen_kwargs)
                            strict_kwargs["do_sample"] = False
                            strict_kwargs["temperature"] = 0.0
                            strict_kwargs.pop("top_p", None)
                            retry_text = self._manual_generate(prompt, strict_kwargs)
                            if retry_text and str(retry_text).strip():
                                text = retry_text
                            if text is None or not str(text).strip() or not self._looks_like_json(text):
                                repair_prompt = self._json_retry_prompt(prompt)
                                repair_text = self._manual_generate(repair_prompt, strict_kwargs)
                                if repair_text and str(repair_text).strip():
                                    text = repair_text
                        outputs.append(text if text is not None else "")
                    return outputs

                result = self._pipeline(prompts, **gen_kwargs)
                outputs = []
                for prompt, output in zip(prompts, result):
                    text = self._extract_generated_text(output, prompt)
                    if text is None or not str(text).strip():
                        fallback_kwargs = dict(gen_kwargs)
                        min_new_tokens = fallback_kwargs.get("min_new_tokens")
                        if not isinstance(min_new_tokens, int) or min_new_tokens <= 0:
                            max_new = fallback_kwargs.get("max_new_tokens")
                            fallback_kwargs["min_new_tokens"] = min(32, max_new) if isinstance(max_new, int) and max_new > 0 else 32
                        fallback_output = self._pipeline(prompt, **fallback_kwargs)
                        text = self._extract_generated_text(fallback_output, prompt)
                    if text is None or not str(text).strip() or not self._looks_like_json(text):
                        manual = self._manual_generate(prompt, gen_kwargs)
                        if manual and str(manual).strip():
                            text = manual
                        if text is None or not str(text).strip() or not self._looks_like_json(text):
                            repair_prompt = self._json_retry_prompt(prompt)
                            strict_kwargs = dict(gen_kwargs)
                            strict_kwargs["do_sample"] = False
                            strict_kwargs["temperature"] = 0.0
                            strict_kwargs.pop("top_p", None)
                            manual = self._manual_generate(repair_prompt, strict_kwargs)
                            if manual and str(manual).strip():
                                text = manual
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
