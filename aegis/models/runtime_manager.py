"""Provider/runtime caching and concurrency control."""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, List

from aegis.models.parser_factory import get_parser
from aegis.models.provider_factory import ProviderCreationError, create_provider
from aegis.models.runtime import resolve_runtime
from aegis.models.runners import TriageRunner, DeepScanRunner, JudgeRunner, ExplainRunner
from aegis.models.schema import ModelRecord, ModelRole, ModelType

logger = logging.getLogger(__name__)


class ModelRuntime:
    """Holds a cached provider + parser + runner set for a model."""

    def __init__(self, model: ModelRecord):
        self.model = model
        self.settings = model.settings or {}
        self.runtime_spec = resolve_runtime(self.settings)

        # Track provider load time for telemetry
        provider_load_start = time.time()
        self.provider = create_provider(model)
        self.provider_load_time_ms = int((time.time() - provider_load_start) * 1000)

        # Collect provider telemetry if available (for HF models)
        self.telemetry = {}
        if hasattr(self.provider, 'get_telemetry'):
            try:
                self.telemetry = self.provider.get_telemetry()
                self.telemetry["load_time_ms"] = self.provider_load_time_ms
            except Exception as e:
                logger.warning(f"Failed to collect provider telemetry: {e}")

        parser_cfg = self.settings.get("parser_config", {})
        parser_id = model.parser_id
        if not parser_id and model.model_type == ModelType.TOOL_ML:
            parser_id = "tool_result"
        self.parser = get_parser(parser_id or "json_schema", parser_cfg)
        self.runners: Dict[ModelRole, Any] = {}
        self.keep_alive_seconds = self.runtime_spec.keep_alive_seconds
        self.last_used = time.time()
        self._semaphore = threading.Semaphore(self.runtime_spec.max_concurrency)

        # Rate limiter and cost tracker for cloud providers
        self.rate_limiter = None
        self.cost_tracker = None
        self._setup_cloud_features()

    def _setup_cloud_features(self):
        """Setup rate limiter and cost tracker for cloud providers."""
        # Check if this is a cloud provider
        is_cloud = self.model.model_type in (
            ModelType.OPENAI_CLOUD,
            ModelType.ANTHROPIC_CLOUD,
            ModelType.GOOGLE_CLOUD,
        )

        if is_cloud:
            try:
                from aegis.models.rate_limiter import DEFAULT_RATE_LIMITER, configure_rate_limiter
                from aegis.models.cost_tracker import DEFAULT_COST_TRACKER

                # Configure rate limiter
                model_type_str = self.model.model_type.value if hasattr(self.model.model_type, 'value') else self.model.model_type
                provider_type = self.model.provider_id or model_type_str.replace("_cloud", "")
                custom_rpm = self.settings.get("rate_limit", {}).get("rpm")

                configure_rate_limiter(
                    DEFAULT_RATE_LIMITER,
                    provider_type,
                    self.model.model_name,
                    rpm=custom_rpm,
                )

                self.rate_limiter = DEFAULT_RATE_LIMITER
                self.cost_tracker = DEFAULT_COST_TRACKER

                logger.info(f"Enabled rate limiting and cost tracking for {provider_type}:{self.model.model_name}")

            except Exception as e:
                logger.warning(f"Failed to setup cloud features: {e}")

    def touch(self) -> None:
        self.last_used = time.time()

    def _build_runner(self, role: ModelRole):
        # Include model info in runner config for auto-detection features
        runner_config = dict(self.settings)
        runner_config["model_name"] = self.model.model_name
        runner_config["model_id"] = self.model.model_id

        if role == ModelRole.TRIAGE:
            return TriageRunner(self.provider, self.parser, config=runner_config)
        if role == ModelRole.JUDGE:
            return JudgeRunner(self.provider, self.parser, config=runner_config)
        if role == ModelRole.EXPLAIN:
            return ExplainRunner(self.provider, self.parser, config=runner_config)
        return DeepScanRunner(self.provider, self.parser, config=runner_config)

    def _normalize_role(self, role: Any) -> ModelRole:
        if isinstance(role, ModelRole):
            return role
        if isinstance(role, str):
            try:
                return ModelRole(role)
            except ValueError:
                legacy = {
                    "scan": ModelRole.DEEP_SCAN,
                    "deep_scan": ModelRole.DEEP_SCAN,
                    "triage": ModelRole.TRIAGE,
                    "judge": ModelRole.JUDGE,
                    "explain": ModelRole.EXPLAIN,
                    "custom": ModelRole.CUSTOM,
                }
                mapped = legacy.get(role.lower())
                if mapped:
                    return mapped
        return ModelRole.DEEP_SCAN

    def get_runner(self, role: ModelRole):
        role = self._normalize_role(role)
        runner = self.runners.get(role)
        if runner:
            return runner
        runner = self._build_runner(role)
        self.runners[role] = runner
        return runner

    async def run(self, prompt: str, context: Dict[str, Any], role: Optional[ModelRole] = None, scan_id: Optional[str] = None, **kwargs):
        target_role = role or (self.model.roles[0] if self.model.roles else ModelRole.DEEP_SCAN)
        target_role = self._normalize_role(target_role)
        runner = self.get_runner(target_role)
        self.touch()

        # Rate limiting for cloud providers
        if self.rate_limiter:
            provider_key = f"{self.model.provider_id}:{self.model.model_name}"
            try:
                await self.rate_limiter.acquire(provider_key, tokens=1.0, timeout=60.0)
            except Exception as e:
                logger.warning(f"Rate limit acquire failed: {e}")

        await asyncio.to_thread(self._semaphore.acquire)
        try:
            # Track start time for cost calculation
            start_time = time.time()

            # Run the model
            result = await runner.run(prompt, context, **kwargs)

            # Cost tracking for cloud providers
            if self.cost_tracker and hasattr(self.provider, "provider"):
                self._log_api_usage(prompt, result, scan_id, start_time)

            return result
        finally:
            self._semaphore.release()

    async def run_batch(
        self,
        prompts: List[str],
        contexts: List[Dict[str, Any]],
        role: Optional[ModelRole] = None,
        scan_id: Optional[str] = None,
        **kwargs
    ):
        """Run a batch of prompts through the provider and parser."""
        target_role = role or (self.model.roles[0] if self.model.roles else ModelRole.DEEP_SCAN)
        target_role = self._normalize_role(target_role)
        runner = self.get_runner(target_role)
        self.touch()

        # Rate limiting for cloud providers (batch counts as one call)
        if self.rate_limiter:
            provider_key = f"{self.model.provider_id}:{self.model.model_name}"
            try:
                await self.rate_limiter.acquire(provider_key, tokens=1.0, timeout=60.0)
            except Exception as e:
                logger.warning(f"Rate limit acquire failed: {e}")

        await asyncio.to_thread(self._semaphore.acquire)
        try:
            # Track start time for cost calculation
            start_time = time.time()

            if hasattr(self.provider, "analyze_batch"):
                raw_outputs = await self.provider.analyze_batch(prompts, contexts, **kwargs)
                results = []
                for raw_output, context, prompt in zip(raw_outputs, contexts, prompts):
                    if isinstance(context, dict) and "prompt" not in context:
                        context["prompt"] = prompt
                    results.append(runner.parser.parse(raw_output, context))
            else:
                results = []
                for prompt, context in zip(prompts, contexts):
                    results.append(await runner.run(prompt, context, **kwargs))

            # Cost tracking for cloud providers (best-effort)
            if self.cost_tracker and hasattr(self.provider, "provider"):
                self._log_api_usage(" ".join(prompts[:1]), results[-1] if results else None, scan_id, start_time)

            return results
        finally:
            self._semaphore.release()

    def _log_api_usage(self, prompt: str, result: Any, scan_id: Optional[str], start_time: float):
        """Log API usage and cost for cloud providers."""
        try:
            # Extract token usage from result metadata if available
            input_tokens = getattr(result, "input_tokens", 0) or 0
            output_tokens = getattr(result, "output_tokens", 0) or 0

            # Estimate tokens if not provided (rough approximation)
            if input_tokens == 0:
                input_tokens = len(prompt) // 4  # ~4 chars per token

            if output_tokens == 0 and hasattr(result, "findings"):
                # Estimate based on findings
                output_tokens = len(str(result.findings)) // 4

            # Calculate cost
            model_type_str = self.model.model_type.value if hasattr(self.model.model_type, 'value') else self.model.model_type
            provider_type = self.model.provider_id or model_type_str.replace("_cloud", "")
            cost_usd = 0.0

            if provider_type == "openai":
                from aegis.providers.openai_provider import calculate_cost
                cost_usd = calculate_cost(self.model.model_name, input_tokens, output_tokens)
            elif provider_type == "anthropic":
                from aegis.providers.anthropic_provider import calculate_cost
                cost_usd = calculate_cost(self.model.model_name, input_tokens, output_tokens)
            elif provider_type == "google":
                from aegis.providers.google_provider import calculate_cost
                cost_usd = calculate_cost(self.model.model_name, input_tokens, output_tokens)

            # Log to cost tracker
            if cost_usd > 0:
                self.cost_tracker.log_usage(
                    provider=provider_type,
                    model_name=self.model.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    scan_id=scan_id,
                )

        except Exception as e:
            logger.warning(f"Failed to log API usage: {e}")

    def close(self) -> None:
        close_fn = getattr(self.provider, "close", None)
        if callable(close_fn):
            close_fn()
        shutdown_fn = getattr(self.provider, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()


class ModelRuntimeManager:
    """Caches runtimes keyed by model+settings signature."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runtimes: Dict[str, ModelRuntime] = {}

    def _runtime_key(self, model: ModelRecord) -> str:
        role_values = []
        for role in model.roles:
            value = getattr(role, "value", None)
            role_values.append(value if value is not None else str(role))
        signature = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "parser_id": model.parser_id,
            "roles": role_values,
            "settings": model.settings,
        }
        raw = json.dumps(signature, sort_keys=True, default=str)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"{model.model_id}:{digest}"

    def _prune_locked(self, now: float) -> None:
        to_remove = []
        for key, runtime in self._runtimes.items():
            ttl = runtime.keep_alive_seconds
            if ttl and ttl > 0 and (now - runtime.last_used) > ttl:
                to_remove.append(key)

        for key in to_remove:
            runtime = self._runtimes.pop(key, None)
            if runtime:
                runtime.close()

    def get_runtime(self, model: ModelRecord) -> ModelRuntime:
        key = self._runtime_key(model)
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            runtime = self._runtimes.get(key)
            if runtime:
                runtime.touch()
                return runtime
            try:
                runtime = ModelRuntime(model)
            except ProviderCreationError:
                raise
            except Exception as exc:
                logger.error("Failed to initialize runtime for %s: %s", model.model_id, exc)
                raise
            self._runtimes[key] = runtime
            return runtime

    def clear_model(self, model_id: str) -> None:
        with self._lock:
            keys = [k for k in self._runtimes.keys() if k.startswith(f"{model_id}:")]
            for key in keys:
                runtime = self._runtimes.pop(key, None)
                if runtime:
                    runtime.close()

    def clear_all(self) -> None:
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.close()
            self._runtimes.clear()

    def warmup_model(self, model: ModelRecord, dummy_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Pre-load a model and run warmup inference to compile CUDA kernels.

        Args:
            model: ModelRecord to warm up
            dummy_input: Optional test input for warmup inference

        Returns:
            Dict with warmup results including device, VRAM, load times
        """
        result = {
            "model_id": model.model_id,
            "success": False,
            "cached": False,
            "device": None,
            "vram_mb": 0,
            "load_time_ms": 0,
            "warmup_time_ms": 0,
            "error": None,
        }

        try:
            key = self._runtime_key(model)

            with self._lock:
                # Check if already cached
                existing = self._runtimes.get(key)
                if existing:
                    result["cached"] = True
                    # If provider supports warmup, run it anyway to get telemetry
                    if hasattr(existing.provider, "warmup"):
                        warmup_result = existing.provider.warmup(dummy_input)
                        result.update({
                            "success": warmup_result.get("success", True),
                            "device": warmup_result.get("device"),
                            "vram_mb": warmup_result.get("vram_mb", 0),
                            "warmup_time_ms": warmup_result.get("warmup_time_ms", 0),
                            "error": warmup_result.get("error"),
                        })
                    else:
                        result["success"] = True
                        if existing.telemetry:
                            result["device"] = existing.telemetry.get("device")
                            result["vram_mb"] = existing.telemetry.get("vram_mb", 0)
                    return result

            # Create runtime (this initializes the provider)
            runtime = self.get_runtime(model)
            result["load_time_ms"] = runtime.provider_load_time_ms

            # Run warmup if provider supports it
            if hasattr(runtime.provider, "warmup"):
                warmup_result = runtime.provider.warmup(dummy_input)
                result.update({
                    "success": warmup_result.get("success", False),
                    "device": warmup_result.get("device"),
                    "vram_mb": warmup_result.get("vram_mb", 0),
                    "warmup_time_ms": warmup_result.get("warmup_time_ms", 0),
                    "error": warmup_result.get("error"),
                })
            else:
                # Provider doesn't support warmup, but we've created the runtime
                result["success"] = True
                if runtime.telemetry:
                    result["device"] = runtime.telemetry.get("device")
                    result["vram_mb"] = runtime.telemetry.get("vram_mb", 0)

            logger.info(
                f"Model warmup complete: {model.model_id} "
                f"(device={result['device']}, vram={result['vram_mb']}MB, "
                f"load={result['load_time_ms']}ms, warmup={result['warmup_time_ms']}ms)"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Model warmup failed for {model.model_id}: {e}")

        return result

    def unload_model(self, model_id: str, clear_cuda_cache: bool = True) -> Dict[str, Any]:
        """
        Unload a model from memory to free GPU/CPU resources.

        Args:
            model_id: Model ID to unload
            clear_cuda_cache: If True, clear CUDA cache after unload

        Returns:
            Dict with unload results
        """
        result = {
            "model_id": model_id,
            "success": False,
            "freed_vram_mb": 0,
            "runtimes_removed": 0,
            "error": None,
        }

        try:
            with self._lock:
                keys = [k for k in self._runtimes.keys() if k.startswith(f"{model_id}:")]

                for key in keys:
                    runtime = self._runtimes.pop(key, None)
                    if runtime:
                        # Call provider unload if available
                        if hasattr(runtime.provider, "unload"):
                            unload_result = runtime.provider.unload(clear_cuda_cache=clear_cuda_cache)
                            result["freed_vram_mb"] += unload_result.get("freed_vram_mb", 0)
                        else:
                            runtime.close()
                        result["runtimes_removed"] += 1

            result["success"] = True
            logger.info(
                f"Model unloaded: {model_id} "
                f"(freed {result['freed_vram_mb']}MB VRAM, "
                f"{result['runtimes_removed']} runtimes removed)"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Model unload failed for {model_id}: {e}")

        return result

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get the current status and telemetry for a loaded model.

        Args:
            model_id: Model ID to check

        Returns:
            Dict with model status including loaded state, device, VRAM
        """
        status = {
            "model_id": model_id,
            "loaded": False,
            "device": None,
            "vram_mb": 0,
            "last_used": None,
            "runtime_count": 0,
        }

        with self._lock:
            keys = [k for k in self._runtimes.keys() if k.startswith(f"{model_id}:")]
            status["runtime_count"] = len(keys)

            if keys:
                status["loaded"] = True
                # Get telemetry from first runtime
                runtime = self._runtimes.get(keys[0])
                if runtime:
                    status["last_used"] = runtime.last_used

                    # Try to get current telemetry
                    if hasattr(runtime.provider, "get_telemetry"):
                        try:
                            telemetry = runtime.provider.get_telemetry()
                            status["device"] = telemetry.get("device")
                            status["vram_mb"] = telemetry.get("vram_mb", 0)
                        except Exception:
                            pass

                    # Fallback to cached telemetry
                    if not status["device"] and runtime.telemetry:
                        status["device"] = runtime.telemetry.get("device")
                        status["vram_mb"] = runtime.telemetry.get("vram_mb", 0)

        return status

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List all currently loaded models with their status.

        Returns:
            List of model status dictionaries
        """
        models = []
        seen_model_ids = set()

        with self._lock:
            for key, runtime in self._runtimes.items():
                model_id = runtime.model.model_id
                if model_id in seen_model_ids:
                    continue
                seen_model_ids.add(model_id)

                # model_type may be enum or string depending on pydantic config
                model_type = runtime.model.model_type
                if hasattr(model_type, 'value'):
                    model_type = model_type.value

                status = {
                    "model_id": model_id,
                    "model_name": runtime.model.model_name,
                    "model_type": model_type,
                    "device": None,
                    "vram_mb": 0,
                    "last_used": runtime.last_used,
                }

                if runtime.telemetry:
                    status["device"] = runtime.telemetry.get("device")
                    status["vram_mb"] = runtime.telemetry.get("vram_mb", 0)

                models.append(status)

        return models

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU memory information.

        Returns:
            Dict with total_vram_mb, used_vram_mb, free_vram_mb, gpu_name
        """
        info = {
            "available": False,
            "gpu_name": None,
            "total_vram_mb": 0,
            "used_vram_mb": 0,
            "free_vram_mb": 0,
        }

        try:
            import torch
            if torch.cuda.is_available():
                info["available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)

                # Get memory info
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)

                info["total_vram_mb"] = int(total / (1024 * 1024))
                info["used_vram_mb"] = int(allocated / (1024 * 1024))
                info["reserved_vram_mb"] = int(reserved / (1024 * 1024))
                info["free_vram_mb"] = info["total_vram_mb"] - info["used_vram_mb"]
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")

        return info

    def is_model_cached(self, model: ModelRecord) -> Dict[str, Any]:
        """
        Check if a HuggingFace model is already downloaded to cache.

        This allows the UI to show download progress when a model needs to be
        downloaded before scanning.

        Args:
            model: ModelRecord to check

        Returns:
            Dict with: cached (bool), size_mb (int estimate), model_type (str)
        """
        result = {
            "model_id": model.model_id,
            "cached": True,  # Assume cached for non-HF models
            "size_mb": 0,
            "model_type": str(model.model_type.value if hasattr(model.model_type, 'value') else model.model_type),
            "needs_download": False,
        }

        # Only check HF local models
        if model.model_type != ModelType.HF_LOCAL:
            return result

        try:
            # Check if already loaded in runtime cache
            key = self._runtime_key(model)
            with self._lock:
                if key in self._runtimes:
                    result["cached"] = True
                    return result

            # Create a temporary provider to check cache status
            from aegis.providers.hf_local import HFLocalProvider

            settings = model.settings or {}
            temp_provider = HFLocalProvider(
                model_id=model.model_name,
                task_type=settings.get("task_type", "text-classification"),
                device="cpu",  # Don't actually load on GPU
                custom_loading=settings.get("custom_loading", False),
            )

            is_cached = temp_provider.is_model_cached()
            result["cached"] = is_cached
            result["needs_download"] = not is_cached

            if not is_cached:
                # Try to get estimated size
                result["size_mb"] = temp_provider.get_model_size_estimate()

        except Exception as e:
            logger.debug(f"Cache check failed for {model.model_id}: {e}")
            # On error, assume cached to avoid blocking scans
            result["cached"] = True

        return result


DEFAULT_RUNTIME_MANAGER = ModelRuntimeManager()
