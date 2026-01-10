"""Factory helpers for creating provider instances based on ModelRecord."""

import os
from typing import Any, Dict

from aegis.connectors.ollama_connector import OllamaConnector
from aegis.connectors.openai_connector import OpenAIConnector
from aegis.models.schema import ModelRecord, ModelType
from aegis.models.runtime import RuntimeConfigError, resolve_runtime
from aegis.providers.hf_local import HFLocalProvider


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

    raise ProviderCreationError(f"Unsupported model type: {model.model_type}")
