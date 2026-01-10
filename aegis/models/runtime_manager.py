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
from aegis.models.schema import ModelRecord, ModelRole

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

    async def run(self, prompt: str, context: Dict[str, Any], role: Optional[ModelRole] = None, **kwargs):
        target_role = role or (self.model.roles[0] if self.model.roles else ModelRole.DEEP_SCAN)
        target_role = self._normalize_role(target_role)
        runner = self.get_runner(target_role)
        self.touch()

        await asyncio.to_thread(self._semaphore.acquire)
        try:
            return await runner.run(prompt, context, **kwargs)
        finally:
            self._semaphore.release()

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
