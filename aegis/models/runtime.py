"""Runtime selection and settings normalization for model execution."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RuntimeConfigError(ValueError):
    """Raised when runtime requirements cannot be satisfied."""


@dataclass(frozen=True)
class RuntimeSpec:
    """Resolved runtime settings for a model."""

    device: str
    device_map: Optional[str]
    dtype: Optional[str]
    quantization: Optional[str]
    max_concurrency: int
    keep_alive_seconds: int
    allow_fallback: bool
    require_device: Optional[str]


def _detect_cuda() -> bool:
    try:
        import torch

        return bool(hasattr(torch, "cuda") and torch.cuda.is_available())
    except Exception:
        return False


def _normalize_list(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return list(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_runtime(settings: Dict[str, Any], cuda_available: Optional[bool] = None) -> RuntimeSpec:
    """
    Resolve runtime settings from model settings.

    Supported settings:
      settings.runtime.device: explicit device (cpu, cuda, cuda:0)
      settings.runtime.device_preference: ordered list (["cuda", "cpu"])
      settings.runtime.device_map: transformers device_map value
      settings.runtime.dtype: bf16|fp16|fp32
      settings.runtime.quantization: 4bit|8bit|none
      settings.runtime.max_concurrency: integer
      settings.runtime.keep_alive_seconds: integer (0 = keep forever)
      settings.runtime.allow_fallback: bool (default True)
      settings.runtime.require_device: "cuda" to enforce GPU availability
    """
    settings = settings or {}
    runtime = settings.get("runtime") or {}

    allow_fallback = bool(runtime.get("allow_fallback", True))
    require_device = runtime.get("require_device") or settings.get("require_device")
    device_override = runtime.get("device") or settings.get("device")
    device_preference = _normalize_list(
        runtime.get("device_preference") or settings.get("device_preference"),
        ["cuda", "cpu"],
    )
    device_map = runtime.get("device_map") if "device_map" in runtime else None
    dtype = runtime.get("dtype") if "dtype" in runtime else settings.get("dtype")
    quantization = runtime.get("quantization") if "quantization" in runtime else settings.get("quantization")
    max_concurrency = _to_int(
        runtime.get("max_concurrency") or settings.get("max_concurrency"),
        1,
    )
    keep_alive_seconds = _to_int(
        runtime.get("keep_alive_seconds") or settings.get("keep_alive_seconds"),
        0,
    )

    if cuda_available is None:
        cuda_available = _detect_cuda()

    if require_device:
        if require_device.startswith("cuda") and not cuda_available:
            raise RuntimeConfigError("CUDA required but not available on this host.")

    device = None
    if device_override:
        if isinstance(device_override, str) and device_override.lower() in ("gpu", "cuda"):
            device_override = "cuda"
        device = str(device_override)
        if device.startswith("cuda") and not cuda_available:
            if allow_fallback and not require_device:
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                device = "cpu"
            else:
                raise RuntimeConfigError("CUDA requested but not available on this host.")
    else:
        for pref in device_preference:
            pref_str = str(pref).lower()
            if pref_str in ("gpu", "cuda") and cuda_available:
                device = "cuda"
                break
            if pref_str.startswith("cuda") and cuda_available:
                device = str(pref)
                break
            if pref_str == "cpu":
                device = "cpu"
                break
        if not device:
            device = "cpu"

    if device.startswith("cpu") and device_map:
        if not runtime.get("device_map_on_cpu", False):
            device_map = None

    if dtype:
        dtype = str(dtype).lower()
        if device.startswith("cpu") and dtype in ("bf16", "bfloat16", "fp16", "float16"):
            if runtime.get("allow_half_cpu", False):
                dtype = "fp16" if dtype in ("fp16", "float16") else "bf16"
            else:
                logger.warning("Half-precision requested on CPU. Falling back to fp32.")
                dtype = "fp32"

    if quantization:
        quantization = str(quantization).lower()
        if quantization == "none":
            quantization = None
        elif quantization in ("4bit", "8bit"):
            if device.startswith("cpu"):
                if allow_fallback:
                    logger.warning("Quantization requested on CPU. Disabling quantization.")
                    quantization = None
                else:
                    raise RuntimeConfigError("Quantization requires CUDA.")

    return RuntimeSpec(
        device=device,
        device_map=device_map,
        dtype=dtype,
        quantization=quantization,
        max_concurrency=max(1, max_concurrency),
        keep_alive_seconds=max(0, keep_alive_seconds),
        allow_fallback=allow_fallback,
        require_device=require_device,
    )
