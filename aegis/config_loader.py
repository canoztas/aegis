"""YAML configuration loader with environment variable resolution."""
import os
import re
import yaml
from typing import Dict, Any, List
from pathlib import Path


class ConfigLoader:
    """Load and parse YAML configuration with environment variable support."""

    ENV_VAR_PATTERN = re.compile(r'\$\{([A-Z_][A-Z0-9_]*)\}')

    @classmethod
    def resolve_env_vars(cls, value: Any) -> Any:
        """
        Resolve environment variables in configuration values.

        Supports ${ENV_VAR} syntax.

        Args:
            value: Configuration value (str, dict, list, or other)

        Returns:
            Resolved value
        """
        if isinstance(value, str):
            # Replace ${VAR} with environment variable value
            def replace_env(match):
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    print(f"Warning: Environment variable '{var_name}' not found, using empty string")
                    return ""
                return env_value

            return cls.ENV_VAR_PATTERN.sub(replace_env, value)

        elif isinstance(value, dict):
            return {k: cls.resolve_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [cls.resolve_env_vars(item) for item in value]

        else:
            return value

    @classmethod
    def load_yaml(cls, config_path: Path) -> Dict[str, Any]:
        """
        Load YAML configuration file with environment variable resolution.

        Args:
            config_path: Path to YAML file

        Returns:
            Parsed configuration dict
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # Resolve environment variables
        config = cls.resolve_env_vars(raw_config)

        return config

    @classmethod
    def load_models_config(cls, config_path: Path = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load models configuration from YAML.

        Expected format:
        ```yaml
        providers:
          - name: ollama
            type: llm
            base_url: http://localhost:11434
            rate_limit_per_second: 10
            ...

        models:
          - provider: ollama
            model_id: "ollama:qwen2.5-coder:7b"
            display_name: "Qwen 2.5 Coder 7B"
            model_name: "qwen2.5-coder:7b"
            role: scan
            weight: 1.0
            ...
        ```

        Args:
            config_path: Path to models YAML file

        Returns:
            Dict with 'providers' and 'models' keys
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "models_v2.yaml"

        config = cls.load_yaml(config_path)

        providers = config.get('providers', [])
        models = config.get('models', [])

        # Validate configuration
        cls._validate_models_config(providers, models)

        return {
            'providers': providers,
            'models': models,
        }

    @classmethod
    def _validate_models_config(cls, providers: List[Dict], models: List[Dict]):
        """Validate models configuration."""
        # Check required provider fields
        provider_names = set()
        for provider in providers:
            if 'name' not in provider:
                raise ValueError("Provider missing 'name' field")
            if 'type' not in provider:
                raise ValueError(f"Provider '{provider['name']}' missing 'type' field")
            provider_names.add(provider['name'])

        # Check required model fields
        for model in models:
            if 'provider' not in model:
                raise ValueError("Model missing 'provider' field")
            if 'model_id' not in model:
                raise ValueError("Model missing 'model_id' field")
            if 'model_name' not in model:
                raise ValueError("Model missing 'model_name' field")

            # Check provider exists
            if model['provider'] not in provider_names:
                raise ValueError(
                    f"Model '{model.get('model_id')}' references unknown provider '{model['provider']}'"
                )

    @classmethod
    def bootstrap_from_yaml(cls, config_path: Path = None):
        """
        Bootstrap database from YAML configuration.

        Loads providers and models from YAML and creates them in database.

        Args:
            config_path: Path to models YAML file
        """
        from aegis.database.repositories import ProviderRepository, ModelRepository

        config = cls.load_models_config(config_path)
        provider_repo = ProviderRepository()
        model_repo = ModelRepository()

        provider_map = {}  # name -> id mapping

        # Create providers
        for provider_config in config['providers']:
            name = provider_config['name']

            # Check if already exists
            existing = provider_repo.get_by_name(name)
            if existing:
                provider_map[name] = existing['id']
                print(f"Provider '{name}' already exists (ID: {existing['id']})")
                continue

            # Create new provider
            try:
                provider_id = provider_repo.create(
                    name=name,
                    type=provider_config['type'],
                    config=provider_config,
                )
                provider_map[name] = provider_id
                print(f"Created provider '{name}' (ID: {provider_id})")
            except Exception as e:
                print(f"Failed to create provider '{name}': {e}")

        # Create models
        for model_config in config['models']:
            provider_name = model_config['provider']
            model_id = model_config['model_id']

            if provider_name not in provider_map:
                print(f"Skipping model '{model_id}' - provider '{provider_name}' not found")
                continue

            # Check if already exists
            existing = model_repo.get_by_model_id(model_id)
            if existing:
                print(f"Model '{model_id}' already exists")
                continue

            # Create new model
            try:
                model_repo.create(
                    provider_id=provider_map[provider_name],
                    model_id=model_id,
                    display_name=model_config.get('display_name', model_id),
                    model_name=model_config['model_name'],
                    role=model_config.get('role', 'scan'),
                    config=model_config.get('config', {}),
                )
                print(f"Created model '{model_id}'")
            except Exception as e:
                print(f"Failed to create model '{model_id}': {e}")

        print("\nBootstrap complete!")
