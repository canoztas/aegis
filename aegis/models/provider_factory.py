"""Factory helpers for creating provider instances based on ModelRecord."""

import asyncio
import os
from typing import Any, Dict

from aegis.connectors.ollama_connector import OllamaConnector
from aegis.connectors.openai_connector import OpenAIConnector
from aegis.models.schema import ModelRecord, ModelType
from aegis.models.runtime import RuntimeConfigError, resolve_runtime
from aegis.providers.hf_local import HFLocalProvider
from aegis.providers.tool_provider import ToolProvider


class ProviderCreationError(RuntimeError):
    """Raised when a provider cannot be instantiated."""


class OllamaLocalProvider:
    """Lightweight wrapper around OllamaConnector that returns raw text."""

    def __init__(self, connector: OllamaConnector, model_name: str, settings: Dict[str, Any]):
        self.connector = connector
        self.model_name = model_name
        self.settings = settings or {}

    def generate(self, prompt: str, **kwargs) -> str:
        opts = {
            "temperature": self.settings.get("temperature", 0.0),
            "max_tokens": self.settings.get("max_tokens"),
        }
        opts.update(self.settings.get("options", {}))
        opts.update(kwargs)
        result = self.connector.generate(
            prompt=prompt,
            model=self.model_name,
            temperature=opts.pop("temperature", 0.0),
            max_tokens=opts.pop("max_tokens", None),
            **opts,
        )
        if isinstance(result, dict):
            return result.get("text") or result.get("response") or ""
        return str(result)


class OpenAICompatibleProvider:
    """Wrapper around OpenAIConnector that exposes a generate() method."""

    def __init__(self, connector: OpenAIConnector, model_name: str, settings: Dict[str, Any]):
        self.connector = connector
        self.model_name = model_name
        self.settings = settings or {}

    def generate(self, prompt: str, **kwargs) -> str:
        opts = {
            "temperature": self.settings.get("temperature", 0.0),
            "max_tokens": self.settings.get("max_tokens"),
        }
        opts.update(kwargs)
        result = self.connector.generate(
            prompt=prompt,
            model=self.model_name,
            temperature=opts.pop("temperature", 0.0),
            max_tokens=opts.pop("max_tokens", None),
            **opts,
        )
        return result.get("text", "") if isinstance(result, dict) else str(result)


class CloudProviderAdapter:
    """Adapter for cloud API providers (OpenAI, Anthropic, Google) with sync interface."""

    def __init__(self, provider: Any, settings: Dict[str, Any]):
        """
        Initialize cloud provider adapter.

        Args:
            provider: Async provider instance (OpenAIProvider, AnthropicProvider, GoogleProvider)
            settings: Model settings
        """
        self.provider = provider
        self.settings = settings or {}

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Generate completion synchronously (wraps async provider).

        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        opts = {
            "temperature": self.settings.get("temperature", 0.1),
            "max_tokens": self.settings.get("max_tokens", 2048),
            "top_p": self.settings.get("top_p", 1.0),
        }
        opts.update(kwargs)

        # Run async generate in sync context
        # Use a thread pool to avoid "asyncio.run() cannot be called from a running event loop" error
        # This approach works whether or not an event loop is already running
        import concurrent.futures

        def _run_async():
            """Helper to run async code in a fresh event loop."""
            return asyncio.run(
                self.provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **opts,
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_async)
            return future.result()

    def close(self):
        """Close provider resources."""
        if hasattr(self.provider, "close"):
            self.provider.close()


def create_provider(model: ModelRecord) -> Any:
    """
    Create a provider instance for a given registered model.

    Raises:
        ProviderCreationError: If the provider cannot be created.
    """
    settings = model.settings or {}
    provider_cfg = model.provider_config or {}

    if model.model_type == ModelType.OLLAMA_LOCAL:
        base_url = settings.get("base_url") or provider_cfg.get("base_url") or "http://localhost:11434"
        connector = OllamaConnector(base_url=base_url)
        return OllamaLocalProvider(connector, model.model_name, settings)

    if model.model_type == ModelType.HF_LOCAL:
        try:
            runtime = resolve_runtime(settings)
            hf_kwargs = dict(settings.get("hf_kwargs", {}) or {})
            if runtime.device_map is not None:
                hf_kwargs["device_map"] = runtime.device_map
            if runtime.dtype:
                hf_kwargs["torch_dtype"] = runtime.dtype
            if str(runtime.device).startswith("cpu"):
                hf_kwargs.pop("load_in_4bit", None)
                hf_kwargs.pop("load_in_8bit", None)
                hf_kwargs.pop("quantization_config", None)
            runtime_cfg = settings.get("runtime") or {}
            quantization_set = ("quantization" in runtime_cfg) or ("quantization" in settings)
            if quantization_set:
                if runtime.quantization == "4bit":
                    hf_kwargs["load_in_4bit"] = True
                    hf_kwargs.pop("load_in_8bit", None)
                elif runtime.quantization == "8bit":
                    hf_kwargs["load_in_8bit"] = True
                    hf_kwargs.pop("load_in_4bit", None)
                elif runtime.quantization is None:
                    hf_kwargs.pop("load_in_4bit", None)
                    hf_kwargs.pop("load_in_8bit", None)

            device = runtime.device if "device_map" not in hf_kwargs else None

            return HFLocalProvider(
                model_id=model.model_name,
                task_type=settings.get("task_type", "text-generation"),
                adapter_id=settings.get("adapter_id"),
                base_model_id=settings.get("base_model_id"),
                device=device,
                generation_kwargs=settings.get("generation_kwargs"),
                max_workers=runtime.max_concurrency,
                **hf_kwargs,
            )
        except RuntimeConfigError as exc:
            raise ProviderCreationError(str(exc))
        except Exception as exc:
            raise ProviderCreationError(str(exc))

    if model.model_type == ModelType.OPENAI_COMPATIBLE:
        base_url = settings.get("base_url") or provider_cfg.get("base_url")
        api_key = (
            settings.get("api_key")
            or provider_cfg.get("api_key")
            or os.environ.get(f"{model.provider_id.upper()}_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ProviderCreationError(f"API key missing for provider {model.provider_id}")
        connector = OpenAIConnector(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            provider_type=model.provider_id or "openai",
        )
        return OpenAICompatibleProvider(connector, model.model_name, settings)

    if model.model_type == ModelType.TOOL_ML:
        tool_id = settings.get("tool_id") or model.model_name
        tool_config = settings.get("tool_config") or {}
        try:
            return ToolProvider(tool_id=tool_id, tool_config=tool_config)
        except Exception as exc:
            raise ProviderCreationError(str(exc))

    # Cloud providers (OpenAI, Anthropic, Google)
    if model.model_type == ModelType.OPENAI_CLOUD:
        from aegis.providers.openai_provider import OpenAIProvider

        api_key = (
            settings.get("api_key")
            or provider_cfg.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )
        return CloudProviderAdapter(
            OpenAIProvider(
                model_name=model.model_name,
                api_key=api_key,
                base_url=settings.get("base_url"),
                organization=settings.get("organization"),
                timeout=settings.get("timeout", 120),
                max_retries=settings.get("max_retries", 3),
            ),
            settings,
        )

    if model.model_type == ModelType.ANTHROPIC_CLOUD:
        from aegis.providers.anthropic_provider import AnthropicProvider

        api_key = (
            settings.get("api_key")
            or provider_cfg.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        return CloudProviderAdapter(
            AnthropicProvider(
                model_name=model.model_name,
                api_key=api_key,
                base_url=settings.get("base_url"),
                timeout=settings.get("timeout", 120),
                max_retries=settings.get("max_retries", 3),
            ),
            settings,
        )

    if model.model_type == ModelType.GOOGLE_CLOUD:
        from aegis.providers.google_provider import GoogleProvider

        api_key = (
            settings.get("api_key")
            or provider_cfg.get("api_key")
            or os.environ.get("GOOGLE_API_KEY")
        )
        return CloudProviderAdapter(
            GoogleProvider(
                model_name=model.model_name,
                api_key=api_key,
                timeout=settings.get("timeout", 120),
            ),
            settings,
        )

    raise ProviderCreationError(f"Unsupported model type: {model.model_type}")
