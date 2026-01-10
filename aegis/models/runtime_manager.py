"""Provider/runtime caching and concurrency control."""

import asyncio
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

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
        self.provider = create_provider(model)
        parser_cfg = self.settings.get("parser_config", {})
        self.parser = get_parser(model.parser_id or "json_schema", parser_cfg)
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
                provider_type = self.model.provider_id or self.model.model_type.value.replace("_cloud", "")
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
        if role == ModelRole.TRIAGE:
            return TriageRunner(self.provider, self.parser, config=self.settings)
        if role == ModelRole.JUDGE:
            return JudgeRunner(self.provider, self.parser, config=self.settings)
        if role == ModelRole.EXPLAIN:
            return ExplainRunner(self.provider, self.parser, config=self.settings)
        return DeepScanRunner(self.provider, self.parser, config=self.settings)

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
            provider_type = self.model.provider_id or self.model.model_type.value.replace("_cloud", "")
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


DEFAULT_RUNTIME_MANAGER = ModelRuntimeManager()
