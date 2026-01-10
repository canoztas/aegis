"""Pipeline loader and validator.

Loads pipeline configurations from YAML files, validates them,
and provides access to preset and custom pipelines.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import ValidationError

from aegis.pipeline.schema import PipelineConfig, PipelineStep
from aegis.models.registry import ModelRegistryV2
from aegis.models.schema import ModelRole


class PipelineLoader:
    """Load and validate pipeline configurations."""

    def __init__(self, registry: Optional[ModelRegistryV2] = None):
        """
        Initialize pipeline loader.

        Args:
            registry: Optional ModelRegistryV2 for validating role/model references
        """
        self.registry = registry or ModelRegistryV2()
        self._preset_cache: Dict[str, PipelineConfig] = {}

    def load_from_yaml(self, yaml_path: Path) -> PipelineConfig:
        """
        Load pipeline from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Validated PipelineConfig

        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If pipeline is invalid
            yaml.YAMLError: If YAML is malformed
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # Resolve environment variables in config
        raw_config = self._resolve_env_vars(raw_config)

        # Validate with Pydantic
        try:
            pipeline = PipelineConfig(**raw_config)
        except ValidationError as e:
            raise ValidationError(f"Invalid pipeline configuration in {yaml_path}: {e}")

        # Additional validation: check role/model references
        self._validate_model_references(pipeline)

        return pipeline

    def load_preset(self, preset_name: str) -> PipelineConfig:
        """
        Load a preset pipeline by name.

        Args:
            preset_name: Name of preset pipeline (e.g., 'classic', 'triage_deep', 'judge_consensus')

        Returns:
            PipelineConfig

        Raises:
            FileNotFoundError: If preset doesn't exist
        """
        # Check cache first
        if preset_name in self._preset_cache:
            return self._preset_cache[preset_name]

        # Load from presets directory
        project_root = Path(__file__).parent.parent.parent
        preset_path = project_root / "config" / "pipelines" / f"{preset_name}.yaml"

        pipeline = self.load_from_yaml(preset_path)
        pipeline.is_preset = True

        # Cache it
        self._preset_cache[preset_name] = pipeline

        return pipeline

    def list_presets(self) -> List[str]:
        """
        List available preset pipelines.

        Returns:
            List of preset names
        """
        project_root = Path(__file__).parent.parent.parent
        presets_dir = project_root / "config" / "pipelines"

        if not presets_dir.exists():
            return []

        presets = []
        for yaml_file in presets_dir.glob("*.yaml"):
            preset_name = yaml_file.stem
            presets.append(preset_name)

        return sorted(presets)

    def load_from_dict(self, config_dict: Dict) -> PipelineConfig:
        """
        Load pipeline from dictionary (for API requests).

        Args:
            config_dict: Pipeline configuration as dict

        Returns:
            Validated PipelineConfig

        Raises:
            ValidationError: If pipeline is invalid
        """
        try:
            pipeline = PipelineConfig(**config_dict)
        except ValidationError as e:
            raise ValidationError(f"Invalid pipeline configuration: {e}")

        self._validate_model_references(pipeline)
        return pipeline

    def _resolve_env_vars(self, value):
        """Resolve environment variables in config values."""
        if isinstance(value, str):
            # Simple ${VAR} replacement
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                # Support default values: ${VAR:-default}
                if ":-" in var_name:
                    var_name, default = var_name.split(":-", 1)
                    return os.environ.get(var_name, default)
                return os.environ.get(var_name, value)
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        return value

    def _validate_model_references(self, pipeline: PipelineConfig):
        """
        Validate that all role/model references in pipeline exist in registry.

        Args:
            pipeline: Pipeline to validate

        Raises:
            ValueError: If role/model doesn't exist
        """
        for step in pipeline.steps:
            # Validate role steps
            if step.kind.value == "role" and step.role:
                role_enum = self._normalize_role(step.role)
                role_models = self.registry.get_models_for_role(role_enum) if role_enum else []
                if not role_models:
                    raise ValueError(f"Step '{step.id}': No models found for role '{step.role}'")

            # Validate model steps
            if step.kind.value == "model" and step.models:
                for model_id in step.models:
                    model = self.registry.get_model(model_id)
                    if not model:
                        raise ValueError(f"Step '{step.id}': Model '{model_id}' not found in registry")

    def validate_pipeline(self, pipeline: PipelineConfig) -> List[str]:
        """
        Validate pipeline and return list of warnings/issues.

        Args:
            pipeline: Pipeline to validate

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check for orphaned steps (no depends_on and not first step)
        for i, step in enumerate(pipeline.steps):
            if i > 0 and not step.depends_on:
                warnings.append(f"Step '{step.id}' has no depends_on (will execute after previous step)")

        # Check for unreachable steps (after a gate that might skip)
        for i, step in enumerate(pipeline.steps):
            if step.kind.value == "gate":
                if not step.on_true and not step.on_false:
                    warnings.append(f"Gate step '{step.id}' has no targets (pipeline may stop)")

        # Check consensus steps have valid sources
        for step in pipeline.steps:
            if step.kind.value == "consensus" and step.sources:
                for source_id in step.sources:
                    source_step = next((s for s in pipeline.steps if s.id == source_id), None)
                    if source_step and source_step.kind.value not in ["role", "model", "tool"]:
                        warnings.append(
                            f"Consensus step '{step.id}' sources from '{source_id}' which is not a scan step"
                        )

        return warnings

    def _normalize_role(self, role: str) -> Optional[ModelRole]:
        """Normalize legacy role strings to ModelRole."""
        mapping = {
            "scan": ModelRole.DEEP_SCAN,
            "deep_scan": ModelRole.DEEP_SCAN,
            "triage": ModelRole.TRIAGE,
            "judge": ModelRole.JUDGE,
            "explain": ModelRole.EXPLAIN,
        }
        try:
            return ModelRole(role)
        except Exception:
            return mapping.get(role.lower())


class PipelineRegistry:
    """Registry for managing preset and custom pipelines."""

    def __init__(self, loader: Optional[PipelineLoader] = None):
        """
        Initialize pipeline registry.

        Args:
            loader: Optional PipelineLoader instance
        """
        self.loader = loader or PipelineLoader()
        self._custom_pipelines: Dict[str, PipelineConfig] = {}

    def get_pipeline(self, name: str, is_preset: bool = True) -> Optional[PipelineConfig]:
        """
        Get pipeline by name.

        Args:
            name: Pipeline name
            is_preset: Whether to look in presets (True) or custom (False)

        Returns:
            PipelineConfig or None if not found
        """
        try:
            if is_preset:
                return self.loader.load_preset(name)
            else:
                return self._custom_pipelines.get(name)
        except FileNotFoundError:
            return None

    def register_custom(self, name: str, pipeline: PipelineConfig):
        """
        Register a custom pipeline.

        Args:
            name: Pipeline name
            pipeline: PipelineConfig instance
        """
        pipeline.is_preset = False
        self._custom_pipelines[name] = pipeline

    def list_all(self) -> Dict[str, List[str]]:
        """
        List all available pipelines.

        Returns:
            Dict with 'presets' and 'custom' keys containing pipeline names
        """
        return {
            "presets": self.loader.list_presets(),
            "custom": list(self._custom_pipelines.keys()),
        }
