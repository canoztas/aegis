"""Model execution helpers wiring providers, runners, and parsers together."""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Callable

from aegis.data_models import Finding
from aegis.models.provider_factory import ProviderCreationError
from aegis.models.registry import ModelRegistryV2
from aegis.models.schema import (
    ModelRecord,
    ModelRole,
    ParserResult,
    FindingCandidate,
)
from aegis.models.runtime_manager import DEFAULT_RUNTIME_MANAGER, ModelRuntimeManager


def _candidate_to_finding(candidate: FindingCandidate) -> Finding:
    """Convert FindingCandidate (parser-level) to core Finding dataclass."""
    fingerprint_src = (
        f"{candidate.file_path}|{candidate.line_start}|{candidate.line_end}|"
        f"{candidate.category}|{candidate.description}"
    )
    fingerprint = hashlib.sha1(fingerprint_src.encode("utf-8")).hexdigest()

    return Finding(
        name=candidate.title or candidate.category,
        severity=str(candidate.severity).lower(),
        cwe=candidate.cwe or candidate.metadata.get("cwe", "CWE-000"),
        file=candidate.file_path,
        start_line=int(candidate.line_start or 0),
        end_line=int(candidate.line_end or candidate.line_start or 0),
        message=candidate.description,
        confidence=float(candidate.confidence or 0.0),
        fingerprint=fingerprint,
    )


class ModelExecutionEngine:
    """Executes registered models using providers, runners, and parsers."""

    def __init__(
        self,
        registry: Optional[ModelRegistryV2] = None,
        default_timeout: int = 300,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        runtime_manager: Optional[ModelRuntimeManager] = None,
    ):
        """
        Initialize execution engine.

        Args:
            registry: Model registry instance
            default_timeout: Default timeout in seconds for model execution
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay in seconds between retries
        """
        self.registry = registry or ModelRegistryV2()
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.runtime_manager = runtime_manager or DEFAULT_RUNTIME_MANAGER

    def resolve_model(self, model_id: str) -> Optional[ModelRecord]:
        return self.registry.get_model(model_id)

    def resolve_role(self, role: ModelRole) -> Optional[ModelRecord]:
        return self.registry.get_best_model_for_role(role)

    async def _with_timeout(self, coro, timeout: Optional[int] = None):
        """
        Execute coroutine with timeout.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds (uses default_timeout if None)

        Returns:
            Result of coroutine

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
        """
        timeout = timeout or self.default_timeout
        return await asyncio.wait_for(coro, timeout=timeout)

    async def _with_retry(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        **kwargs
    ):
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            max_retries: Max retry attempts (uses default if None)
            retry_delay: Delay between retries (uses default if None)
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result of function

        Raises:
            Exception: Last exception if all retries fail
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        retry_delay = retry_delay if retry_delay is not None else self.retry_delay

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"All {max_retries + 1} attempts failed: {e}")
                    raise last_exception

    def _build_runner(self, model: ModelRecord, role: Optional[ModelRole] = None):
        """Legacy runner builder. Uses cached runtime when available."""
        try:
            runtime = self.runtime_manager.get_runtime(model)
            target_role = role or (model.roles[0] if model.roles else ModelRole.DEEP_SCAN)
            return runtime.get_runner(target_role)
        except Exception as exc:
            raise ProviderCreationError(str(exc))

    def run_model_sync(
        self,
        model: ModelRecord,
        code: str,
        file_path: str,
        role: Optional[ModelRole] = None,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> ParserResult:
        """Run a model synchronously on provided code."""
        context = {
            "code": code,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "snippet": code,
        }
        prompt = code  # Triage runner uses it directly; deep scan builds template internally
        runtime = self.runtime_manager.get_runtime(model)
        return asyncio.run(runtime.run(prompt, context, role=role))

    def run_model_to_findings(
        self,
        model: ModelRecord,
        code: str,
        file_path: str,
        role: Optional[ModelRole] = None,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> List[Finding]:
        """Execute model and convert parser output to Finding dataclasses."""
        result = self.run_model_sync(
            model=model,
            code=code,
            file_path=file_path,
            role=role,
            line_start=line_start,
            line_end=line_end,
        )
        return [_candidate_to_finding(c) for c in result.findings]

    async def run_models_concurrent(
        self,
        models: List[ModelRecord],
        code: str,
        file_path: str,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> Dict[str, List[Finding]]:
        """
        Execute multiple models concurrently on the same code.

        Args:
            models: List of ModelRecord objects to execute
            code: Source code to analyze
            file_path: Path to the source file
            line_start: Starting line number
            line_end: Ending line number

        Returns:
            Dictionary mapping model_id to list of findings
        """
        context = {
            "code": code,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "snippet": code,
        }

        async def run_single_model(model: ModelRecord) -> tuple[str, List[Finding]]:
            """Run a single model with timeout and retry support."""
            try:
                runtime = self.runtime_manager.get_runtime(model)
                prompt = code

                # Get model-specific timeout (fallback to default)
                timeout = model.settings.get("timeout", self.default_timeout)

                # Execute with timeout and retry
                async def execute():
                    return await runtime.run(prompt, context, role=model.roles[0] if model.roles else None)

                result = await self._with_timeout(
                    self._with_retry(execute),
                    timeout=timeout
                )

                findings = [_candidate_to_finding(c) for c in result.findings]
                return (model.model_id, findings)

            except asyncio.TimeoutError:
                print(f"Warning: Model {model.model_id} timed out after {timeout}s")
                return (model.model_id, [])
            except Exception as e:
                # Log error but don't fail entire scan
                print(f"Warning: Model {model.model_id} failed after retries: {e}")
                return (model.model_id, [])

        # Execute all models concurrently
        tasks = [run_single_model(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary, filtering out exceptions
        per_model_findings = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Error in concurrent execution: {result}")
                continue
            model_id, findings = result
            per_model_findings[model_id] = findings

        return per_model_findings

    def run_models_concurrent_sync(
        self,
        models: List[ModelRecord],
        code: str,
        file_path: str,
        line_start: int = 1,
        line_end: Optional[int] = None,
    ) -> Dict[str, List[Finding]]:
        """
        Synchronous wrapper for concurrent model execution.

        Args:
            models: List of ModelRecord objects to execute
            code: Source code to analyze
            file_path: Path to the source file
            line_start: Starting line number
            line_end: Ending line number

        Returns:
            Dictionary mapping model_id to list of findings
        """
        return asyncio.run(
            self.run_models_concurrent(
                models=models,
                code=code,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
            )
        )

    async def health_check_model(self, model: ModelRecord, timeout: int = 30) -> Dict[str, Any]:
        """
        Perform health check on a model.

        Args:
            model: Model to check
            timeout: Health check timeout in seconds

        Returns:
            Dictionary with health status:
            {
                "model_id": "...",
                "healthy": True/False,
                "response_time_ms": 123.45,
                "error": "error message" (if unhealthy)
            }
        """
        from aegis.models.schema import ModelAvailability

        health_status = {
            "model_id": model.model_id,
            "healthy": False,
            "response_time_ms": None,
            "error": None,
        }

        try:
            # Simple test code to verify model responds
            test_code = "def hello():\n    return 'world'"
            test_context = {
                "code": test_code,
                "file_path": "health_check.py",
                "line_start": 1,
                "line_end": 2,
                "snippet": test_code,
            }

            runtime = self.runtime_manager.get_runtime(model)

            # Measure response time
            start_time = time.time()
            result = await self._with_timeout(
                runtime.run(test_code, test_context, role=model.roles[0] if model.roles else None),
                timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000

            health_status["healthy"] = True
            health_status["response_time_ms"] = round(elapsed_ms, 2)

            # Update model availability in registry
            self.registry.update_availability(
                model.model_id,
                ModelAvailability.AVAILABLE,
                response_time_ms=elapsed_ms
            )

        except asyncio.TimeoutError:
            health_status["error"] = f"Health check timed out after {timeout}s"
            self.registry.update_availability(
                model.model_id,
                ModelAvailability.UNAVAILABLE,
                error_message=health_status["error"]
            )
        except Exception as e:
            health_status["error"] = str(e)
            self.registry.update_availability(
                model.model_id,
                ModelAvailability.UNAVAILABLE,
                error_message=str(e)
            )

        return health_status

    def health_check_model_sync(self, model: ModelRecord, timeout: int = 30) -> Dict[str, Any]:
        """Synchronous wrapper for model health check."""
        return asyncio.run(self.health_check_model(model, timeout))
