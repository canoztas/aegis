"""Enhanced model registry with database persistence and role-based selection."""
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from aegis.adapters.base import AbstractAdapter
from aegis.adapters.ollama_adapter import OllamaAdapter
from aegis.adapters.llm_adapter import LLMAdapter
from aegis.adapters.hf_adapter import HFAdapter
from aegis.adapters.classic_adapter import ClassicAdapter
from aegis.database.repositories import ProviderRepository, ModelRepository
from aegis.connectors import OllamaConnector, OpenAIConnector


class ModelRegistryV2:
    """
    Enhanced model registry with database persistence and role-based selection.

    Features:
    - Database-backed model storage
    - Role-based model assignment (triage, deep_scan, judge, explain, scan)
    - Dynamic adapter creation with caching
    - Provider metadata (rate limits, timeout, retry config)
    - Weight-based consensus
    """

    # Valid model roles
    VALID_ROLES = {"triage", "deep_scan", "judge", "explain", "scan"}

    def __init__(self):
        """Initialize enhanced model registry."""
        self.provider_repo = ProviderRepository()
        self.model_repo = ModelRepository()

        # Cache adapters to avoid recreating them
        self._adapter_cache: Dict[str, AbstractAdapter] = {}

    def get_adapter(self, model_id: str) -> Optional[AbstractAdapter]:
        """
        Get or create adapter for a model.

        Args:
            model_id: Model ID (e.g., 'ollama:qwen2.5-coder:7b')

        Returns:
            AbstractAdapter instance or None
        """
        # Check cache first
        if model_id in self._adapter_cache:
            return self._adapter_cache[model_id]

        # Load from database
        model_data = self.model_repo.get_by_model_id(model_id)
        if not model_data:
            return None

        # Get provider info
        provider_id = model_data['provider_id']
        provider_data = self.provider_repo.get_by_id(provider_id)
        if not provider_data:
            return None

        # Create adapter based on provider type
        adapter = self._create_adapter(model_data, provider_data)
        if adapter:
            self._adapter_cache[model_id] = adapter

        return adapter

    def _create_adapter(
        self, model_data: Dict[str, Any], provider_data: Dict[str, Any]
    ) -> Optional[AbstractAdapter]:
        """Create adapter from model and provider data."""
        provider_name = provider_data['name']
        provider_type = provider_data['type']
        base_url = provider_data.get('base_url')

        model_id = model_data['model_id']
        model_name = model_data['model_name']
        display_name = model_data['display_name']

        # Parse config
        import json
        model_config = json.loads(model_data.get('config_json', '{}'))
        provider_config = json.loads(provider_data.get('config_json', '{}'))

        if provider_name == "ollama":
            # Create Ollama adapter with connector
            connector = OllamaConnector(
                base_url=base_url,
                rate_limit_per_second=provider_data.get('rate_limit_per_second', 10.0),
                timeout_seconds=provider_data.get('timeout_seconds', 600),
                retry_max_attempts=provider_data.get('retry_max_attempts', 3),
                retry_backoff_factor=provider_data.get('retry_backoff_factor', 2.0),
            )
            return OllamaAdapter(
                model_name=model_name,
                base_url=base_url,
                display_name=display_name,
                connector=connector,
            )

        elif provider_type == "llm":
            # OpenAI-compatible providers (OpenAI, Anthropic, custom)
            api_key = model_config.get('api_key') or provider_config.get('api_key')
            if not api_key:
                # Try environment variable
                env_key = f"{provider_name.upper()}_API_KEY"
                api_key = os.environ.get(env_key)

            if not api_key:
                print(f"Warning: No API key found for {provider_name}")
                return None

            # Create LLM adapter with connector
            connector = OpenAIConnector(
                base_url=base_url,
                api_key=api_key,
                provider_type=provider_name,
                rate_limit_per_second=provider_data.get('rate_limit_per_second', 5.0),
                timeout_seconds=provider_data.get('timeout_seconds', 60),
                retry_max_attempts=provider_data.get('retry_max_attempts', 3),
                retry_backoff_factor=provider_data.get('retry_backoff_factor', 2.0),
            )
            return LLMAdapter(
                provider=provider_name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                display_name=display_name,
                connector=connector,
            )

        elif provider_type == "huggingface":
            # HuggingFace adapter
            task = model_config.get('task', 'text-generation')
            cache_dir = model_config.get('cache_dir')
            return HFAdapter(
                model_id=model_name,
                task=task,
                cache_dir=cache_dir,
                display_name=display_name,
            )

        elif provider_type == "classic":
            # Classic ML adapter
            model_path = model_config.get('model_path')
            model_type = model_config.get('model_type', 'sklearn')
            if not model_path:
                print(f"Warning: No model_path for classic model {model_id}")
                return None

            return ClassicAdapter(
                model_path=model_path,
                model_type=model_type,
                display_name=display_name,
            )

        return None

    def list_models_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        List models by role.

        Args:
            role: Model role (triage, deep_scan, judge, explain, scan)

        Returns:
            List of model dicts with adapter info
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Valid roles: {self.VALID_ROLES}")

        models = self.model_repo.list_by_role(role)
        result = []

        for model_data in models:
            provider_data = self.provider_repo.get_by_id(model_data['provider_id'])
            if not provider_data:
                continue

            result.append({
                "model_id": model_data['model_id'],
                "display_name": model_data['display_name'],
                "provider": provider_data['name'],
                "role": model_data['role'],
                "weight": model_data.get('weight', 1.0),
                "supports_streaming": model_data.get('supports_streaming', False),
                "supports_json": model_data.get('supports_json', True),
            })

        return result

    def list_all_models(self) -> List[Dict[str, Any]]:
        """List all enabled models."""
        models = self.model_repo.list_all()
        result = []

        for model_data in models:
            if not model_data.get('enabled', True):
                continue

            provider_data = self.provider_repo.get_by_id(model_data['provider_id'])
            if not provider_data or not provider_data.get('enabled', True):
                continue

            result.append({
                "model_id": model_data['model_id'],
                "display_name": model_data['display_name'],
                "provider": provider_data['name'],
                "role": model_data['role'],
                "weight": model_data.get('weight', 1.0),
            })

        return result

    def get_adapters_by_role(self, role: str) -> List[AbstractAdapter]:
        """
        Get all adapter instances for a given role.

        Args:
            role: Model role

        Returns:
            List of AbstractAdapter instances
        """
        models = self.list_models_by_role(role)
        adapters = []

        for model_info in models:
            adapter = self.get_adapter(model_info['model_id'])
            if adapter:
                adapters.append(adapter)

        return adapters

    def get_model_weight(self, model_id: str) -> float:
        """Get model weight for consensus voting."""
        model_data = self.model_repo.get_by_model_id(model_id)
        if not model_data:
            return 1.0
        return model_data.get('weight', 1.0)

    def add_model(
        self,
        provider_name: str,
        model_id: str,
        display_name: str,
        model_name: str,
        role: str = "scan",
        weight: float = 1.0,
        **config,
    ) -> bool:
        """
        Add a new model to the registry.

        Args:
            provider_name: Provider name (must exist in database)
            model_id: Unique model identifier
            display_name: Human-readable name
            model_name: Actual model name for API calls
            role: Model role
            weight: Model weight for consensus
            **config: Additional model configuration

        Returns:
            bool: Success
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")

        # Get provider
        provider = self.provider_repo.get_by_name(provider_name)
        if not provider:
            raise ValueError(f"Provider not found: {provider_name}")

        # Create model
        try:
            self.model_repo.create(
                provider_id=provider['id'],
                model_id=model_id,
                display_name=display_name,
                model_name=model_name,
                role=role,
                config={
                    "weight": weight,
                    "supports_streaming": config.get('supports_streaming', False),
                    "supports_json": config.get('supports_json', True),
                    **config,
                },
            )
            return True
        except Exception as e:
            print(f"Failed to add model: {e}")
            return False

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        # Clear from cache
        if model_id in self._adapter_cache:
            del self._adapter_cache[model_id]

        # Disable in database (soft delete)
        try:
            self.model_repo.update(model_id, enabled=False)
            return True
        except Exception as e:
            print(f"Failed to remove model: {e}")
            return False

    def clear_cache(self):
        """Clear adapter cache (useful after config changes)."""
        self._adapter_cache.clear()
