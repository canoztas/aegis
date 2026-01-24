"""API routes for model management dashboard."""

import logging
from datetime import datetime
from typing import Any, List
from flask import Blueprint, jsonify, request, current_app

from aegis.models.schema import ModelType, ModelRole, ModelStatus, ModelAvailability, parse_role
from aegis.models.registry import ModelRegistryV2
from aegis.models.discovery.ollama import OllamaDiscoveryClient
from aegis.models.engine import ModelExecutionEngine
from aegis.models.runtime_manager import DEFAULT_RUNTIME_MANAGER
from aegis.providers.hf_local import (
    create_hf_provider,
    HFLocalProvider,
    CODEBERT_INSECURE,
    CODEASTRA_7B,
    CODEBERT_PRIMEVUL,
    VULBERTA_DEVIGN,
    UNIXCODER_PRIMEVUL,
    QWEN25_CODER_7B,
    DEEPSEEK_CODER_V2_LITE,
    STARCODER2_15B,
    PHI35_MINI,
    QWEN25_05B,
)
from aegis.connectors.ollama_connector import OllamaConnector
from aegis.models.catalog import (
    MODEL_CATALOG,
    CATALOG_BY_ID,
    CatalogCategory,
    CatalogStatus,
    get_catalog_entry,
    get_hf_catalog_entries,
    get_ml_catalog_entries,
)

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__, url_prefix="/api/models")


def parse_roles(role_strings: List[str]) -> List[ModelRole]:
    """Parse list of role strings with legacy mapping support."""
    return [parse_role(r) for r in role_strings]


def is_model_ready(catalog_entry: dict, registry: ModelRegistryV2) -> CatalogStatus:
    """
    Compute the current status of a catalog entry.

    Status transitions:
    - AVAILABLE: Not in registry, can be added
    - ADDED: In registry but not downloaded/ready
    - DOWNLOADING: Currently downloading (tracked externally)
    - READY: Downloaded and ready to use
    - NEEDS_KEY: Requires API key (cloud models)
    - NEEDS_ARTIFACT: Requires external artifact (classic_ml)

    Args:
        catalog_entry: Catalog entry dict from MODEL_CATALOG
        registry: ModelRegistryV2 instance

    Returns:
        CatalogStatus enum value
    """
    catalog_id = catalog_entry.get("catalog_id")
    category = catalog_entry.get("category")

    # Check if requires API key (cloud models)
    if catalog_entry.get("requires_api_key"):
        # Check if key is configured
        import os
        key_env_var = None
        provider_id = catalog_entry.get("provider_id", "")
        if provider_id == "openai":
            key_env_var = "OPENAI_API_KEY"
        elif provider_id == "anthropic":
            key_env_var = "ANTHROPIC_API_KEY"
        elif provider_id == "google":
            key_env_var = "GOOGLE_API_KEY"

        if key_env_var and not os.environ.get(key_env_var):
            return CatalogStatus.NEEDS_KEY

    # Check if requires external artifact
    if catalog_entry.get("requires_artifact"):
        return CatalogStatus.NEEDS_ARTIFACT

    # Build expected model_id based on category
    if category == CatalogCategory.HUGGINGFACE:
        model_id = f"hf:{catalog_id}"
    elif category == CatalogCategory.OLLAMA:
        model_id = f"ollama:{catalog_entry.get('model_name', catalog_id)}"
    elif category == CatalogCategory.CLOUD:
        model_id = f"{catalog_entry.get('provider_id', 'cloud')}:{catalog_id}"
    else:
        model_id = f"{catalog_entry.get('provider_id', 'unknown')}:{catalog_id}"

    # Check if registered in DB
    existing = registry.get_model(model_id)
    if not existing:
        return CatalogStatus.AVAILABLE

    # Registered - check if ready based on category
    if category == CatalogCategory.HUGGINGFACE:
        # Check if model files are cached
        hf_model_name = catalog_entry.get("model_name")
        if hf_model_name:
            try:
                provider = HFLocalProvider(
                    model_id=hf_model_name,
                    task_type=catalog_entry.get("task_type", "text-classification"),
                )
                if provider.is_model_cached():
                    return CatalogStatus.READY
            except Exception as e:
                logger.debug(f"Cache check failed for {hf_model_name}: {e}")
        return CatalogStatus.ADDED

    elif category == CatalogCategory.OLLAMA:
        # Check if Ollama model is pulled
        try:
            base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
            client = OllamaDiscoveryClient(base_url)
            models = client.discover_models_sync(force_refresh=False)
            model_name = catalog_entry.get("model_name", "")
            # Check if model name matches any discovered model
            for m in models:
                if m.name == model_name or m.name.startswith(model_name.split(":")[0]):
                    return CatalogStatus.READY
        except Exception as e:
            logger.debug(f"Ollama check failed for {catalog_id}: {e}")
        return CatalogStatus.ADDED

    elif category == CatalogCategory.CLOUD:
        # Cloud models are ready if API key is available
        return CatalogStatus.READY

    elif category == CatalogCategory.CLASSIC_ML:
        # Check if ML model is downloaded/cached
        try:
            from aegis.tools.builtin.sklearn_tool import SklearnTool
            settings = catalog_entry.get("settings", {})
            tool = SklearnTool(config=settings)
            if tool.is_model_cached():
                return CatalogStatus.READY
        except Exception as e:
            logger.debug(f"ML model cache check failed for {catalog_id}: {e}")
        return CatalogStatus.ADDED

    return CatalogStatus.ADDED


# ==============================================================================
# Catalog API routes
# ==============================================================================

@models_bp.route("/catalog", methods=["GET"])
def list_catalog() -> Any:
    """
    List all catalog entries with computed status.

    Query Parameters:
        category: Filter by category (huggingface, ollama, cloud, classic_ml)

    Returns:
        {
          "catalog": [
            {
              "catalog_id": "codebert_insecure",
              "category": "huggingface",
              "display_name": "CodeBERT Insecure Code Detector",
              "status": "available",
              ...
            }
          ]
        }
    """
    try:
        registry = ModelRegistryV2()
        category_filter = request.args.get("category")

        entries = []
        for entry in MODEL_CATALOG:
            # Filter by category if specified
            if category_filter:
                entry_category = entry.get("category")
                if isinstance(entry_category, CatalogCategory):
                    if entry_category.value != category_filter:
                        continue
                elif entry_category != category_filter:
                    continue

            # Compute current status
            status = is_model_ready(entry, registry)

            # Build response entry
            response_entry = {
                **entry,
                "status": status.value if isinstance(status, CatalogStatus) else status,
                "category": entry["category"].value if isinstance(entry.get("category"), CatalogCategory) else entry.get("category"),
            }
            entries.append(response_entry)

        return jsonify({"catalog": entries})

    except Exception as e:
        logger.error(f"List catalog failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/catalog/<catalog_id>/register", methods=["POST"])
def register_catalog_entry(catalog_id: str) -> Any:
    """
    Register a catalog entry to the model registry.

    This adds the model to the database so it appears in the registered models list.
    For HuggingFace models, this does NOT download the model files.

    Args:
        catalog_id: Catalog entry ID (URL parameter)

    Body (optional):
        {
          "display_name": "Custom Name",
          "settings": { ... }
        }

    Returns:
        {"model": <ModelRecord>, "status": "added"}
    """
    try:
        entry = get_catalog_entry(catalog_id)
        if not entry:
            return jsonify({"error": f"Unknown catalog entry: {catalog_id}"}), 404

        data = request.get_json(silent=True) or {}
        category = entry.get("category")

        # Build model_id based on category
        if category == CatalogCategory.HUGGINGFACE:
            model_id = f"hf:{catalog_id}"
            model_type = ModelType.HF_LOCAL
        elif category == CatalogCategory.OLLAMA:
            model_id = f"ollama:{entry.get('model_name', catalog_id)}"
            model_type = ModelType.OLLAMA_LOCAL
        elif category == CatalogCategory.CLOUD:
            provider_id = entry.get("provider_id", "cloud")
            model_id = f"{provider_id}:{catalog_id}"
            # Map provider to model type
            if provider_id == "openai":
                model_type = ModelType.OPENAI
            elif provider_id == "anthropic":
                model_type = ModelType.ANTHROPIC
            elif provider_id == "google":
                model_type = ModelType.GOOGLE
            else:
                model_type = ModelType.CLOUD
        else:
            model_id = f"{entry.get('provider_id', 'unknown')}:{catalog_id}"
            model_type = ModelType.CLASSIC_ML

        # Parse roles
        role_strings = entry.get("roles", ["triage"])
        roles = parse_roles(role_strings)

        # Build settings from catalog entry
        base_settings = {
            "task_type": entry.get("task_type"),
        }

        # Add HF-specific settings
        if category == CatalogCategory.HUGGINGFACE:
            if entry.get("hf_kwargs"):
                base_settings["hf_kwargs"] = entry["hf_kwargs"]
            if entry.get("generation_kwargs"):
                base_settings["generation_kwargs"] = entry["generation_kwargs"]
            if entry.get("custom_loading"):
                base_settings["custom_loading"] = entry["custom_loading"]
            if entry.get("small_model"):
                base_settings["small_model"] = entry["small_model"]
            if entry.get("adapter_id"):
                base_settings["adapter_id"] = entry["adapter_id"]
            if entry.get("base_model_id"):
                base_settings["base_model_id"] = entry["base_model_id"]

        # Add Classic ML-specific settings
        if category == CatalogCategory.CLASSIC_ML:
            if entry.get("tool_id"):
                base_settings["tool_id"] = entry["tool_id"]
            if entry.get("settings"):
                base_settings.update(entry["settings"])

        # Merge with user-provided settings
        user_settings = data.get("settings", {})
        settings = {**base_settings, **user_settings}

        # Register in database
        registry = ModelRegistryV2()
        display_name = data.get("display_name", entry.get("display_name"))

        model = registry.register_model(
            model_id=model_id,
            model_type=model_type,
            provider_id=entry.get("provider_id", "unknown"),
            model_name=entry.get("model_name", catalog_id),
            display_name=display_name,
            roles=roles,
            parser_id=entry.get("parser_id"),
            settings=settings,
            parser_config=entry.get("parser_config"),
        )

        # Compute new status
        status = is_model_ready(entry, registry)

        return jsonify({
            "model": model.model_dump(),
            "status": status.value if isinstance(status, CatalogStatus) else status,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Register catalog entry failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/catalog/<catalog_id>/download", methods=["POST"])
def download_catalog_entry(catalog_id: str) -> Any:
    """
    Download/prefetch a catalog entry's model files.

    For HuggingFace models: Downloads model files to cache without loading to GPU.
    For Ollama models: Triggers `ollama pull`.
    For Cloud models: No-op (returns success).

    Args:
        catalog_id: Catalog entry ID (URL parameter)

    Returns:
        {
          "success": true,
          "status": "ready",
          "cached": false,
          "cache_dir": "/path/to/cache"
        }
    """
    try:
        entry = get_catalog_entry(catalog_id)
        if not entry:
            return jsonify({"error": f"Unknown catalog entry: {catalog_id}"}), 404

        category = entry.get("category")

        if category == CatalogCategory.HUGGINGFACE:
            # Prefetch HuggingFace model
            model_name = entry.get("model_name")
            if not model_name:
                return jsonify({"error": "No model_name in catalog entry"}), 400

            provider = HFLocalProvider(
                model_id=model_name,
                task_type=entry.get("task_type", "text-classification"),
            )

            result = provider.prefetch()

            # Compute updated status
            registry = ModelRegistryV2()
            status = is_model_ready(entry, registry)

            return jsonify({
                **result,
                "status": status.value if isinstance(status, CatalogStatus) else status,
            })

        elif category == CatalogCategory.OLLAMA:
            # Trigger Ollama pull
            model_name = entry.get("model_name")
            if not model_name:
                return jsonify({"error": "No model_name in catalog entry"}), 400

            base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                connector = OllamaConnector(base_url=base_url)
                result = connector.pull_model(model_name)

                registry = ModelRegistryV2()
                status = is_model_ready(entry, registry)

                return jsonify({
                    "success": True,
                    "result": result,
                    "status": status.value if isinstance(status, CatalogStatus) else status,
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "instructions": f"ollama pull {model_name}",
                }), 502

        elif category == CatalogCategory.CLOUD:
            # Cloud models don't need download
            registry = ModelRegistryV2()
            status = is_model_ready(entry, registry)

            return jsonify({
                "success": True,
                "cached": True,  # No download needed
                "status": status.value if isinstance(status, CatalogStatus) else status,
                "message": "Cloud models do not require download",
            })

        elif category == CatalogCategory.CLASSIC_ML:
            # Download ML model from URL
            from aegis.tools.builtin.sklearn_tool import SklearnTool

            settings = entry.get("settings", {})
            tool = SklearnTool(config=settings)

            result = tool.prefetch()

            registry = ModelRegistryV2()
            status = is_model_ready(entry, registry)

            return jsonify({
                **result,
                "status": status.value if isinstance(status, CatalogStatus) else status,
            })

        else:
            return jsonify({
                "success": False,
                "error": f"Download not supported for category: {category}",
            }), 400

    except Exception as e:
        logger.error(f"Download catalog entry failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/discovered/ollama", methods=["GET"])
def discover_ollama_models() -> Any:
    """
    Discover available Ollama models.

    Query Parameters:
        refresh: Force refresh cache (true/false)

    Returns:
        {
          "models": [
            {
              "name": "qwen2.5-coder:7b",
              "model_type": "ollama_local",
              "provider": "ollama",
              "size_bytes": 4900000000,
              "metadata": {...}
            }
          ]
        }
    """
    try:
        base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
        force_refresh = request.args.get("refresh", "false").lower() == "true"

        client = OllamaDiscoveryClient(base_url)
        models = client.discover_models_sync(force_refresh=force_refresh)

        # Update availability for registered Ollama models
        registry = ModelRegistryV2()
        discovered_names = {m.name for m in models}
        registered = registry.list_models(model_type=ModelType.OLLAMA_LOCAL)
        available_ids = [m.model_id for m in registered if m.model_name in discovered_names]
        unavailable_ids = [m.model_id for m in registered if m.model_name not in discovered_names]
        if available_ids:
            registry.update_availability(available_ids, ModelAvailability.AVAILABLE, datetime.utcnow())
        if unavailable_ids:
            registry.update_availability(unavailable_ids, ModelAvailability.UNAVAILABLE, datetime.utcnow())

        return jsonify({
            "models": [m.model_dump() for m in models]
        })

    except Exception as e:
        logger.error(f"Ollama discovery failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/ollama/pull", methods=["POST"])
def pull_ollama_model() -> Any:
    """Trigger an Ollama pull or return CLI instructions."""
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name") or data.get("name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400

    base_url = data.get("base_url") or current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        connector = OllamaConnector(base_url=base_url)
        result = connector.pull_model(model_name)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        logger.error(f"Ollama pull failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "instructions": f"ollama pull {model_name}"
        }), 502


@models_bp.route("/registry", methods=["POST"])
@models_bp.route("/register", methods=["POST"])  # Backward compatibility
def register_model() -> Any:
    """
    Register a model for use in scans.

    Request Body:
        {
          "model_id": "ollama:qwen2.5-coder",
          "model_type": "ollama_local",
          "provider_id": "ollama",
          "model_name": "qwen2.5-coder:7b",
          "display_name": "Qwen 2.5 Coder 7B",
          "roles": ["deep_scan", "judge"],
          "parser_id": "json_schema",
          "settings": {"temperature": 0.1, "max_tokens": 2048}
        }

    Returns:
        {"model": <ModelRecord>}
    """
    try:
        data = request.get_json()

        # Validate required fields
        required = ["model_type", "provider_id", "model_name", "display_name", "roles"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Derive model_id if omitted
        model_id = data.get("model_id") or f"{data['provider_id']}:{data['model_name']}"

        # Parse enums with legacy support
        model_type = ModelType(data["model_type"])
        roles = parse_roles(data["roles"])
        availability = ModelAvailability(data.get("availability", "unknown"))

        # Register
        registry = ModelRegistryV2()
        model = registry.register_model(
            model_id=model_id,
            model_type=model_type,
            provider_id=data["provider_id"],
            model_name=data["model_name"],
            display_name=data["display_name"],
            roles=roles,
            parser_id=data.get("parser_id"),
            settings=data.get("settings", {}),
            parser_config=data.get("parser_config"),
            availability=availability,
        )

        return jsonify({"model": model.model_dump()})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("", methods=["GET"])  # Used by legacy scan UI
@models_bp.route("/registry", methods=["GET"])
@models_bp.route("/registered", methods=["GET"])  # Backward compatibility
def list_registered_models() -> Any:
    """
    List all registered models.

    Query Parameters:
        type: Filter by model_type
        role: Filter by role
        status: Filter by status

    Returns:
        {"models": [<ModelRecord>, ...]}
    """
    try:
        registry = ModelRegistryV2()

        # Parse filters
        model_type = None
        if request.args.get("type"):
            model_type = ModelType(request.args["type"])

        role = None
        if request.args.get("role"):
            role = parse_role(request.args["role"])

        status = None
        if request.args.get("status"):
            status = ModelStatus(request.args["status"])

        availability = None
        if request.args.get("availability"):
            availability = ModelAvailability(request.args["availability"])

        models = registry.list_models(
            model_type=model_type,
            role=role,
            status=status
        )

        if availability:
            models = [m for m in models if m.availability == availability]

        return jsonify({
            "models": [m.model_dump() for m in models]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"List models failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/registry/<model_id>", methods=["PUT", "PATCH"])
def update_registered_model(model_id: str) -> Any:
    """
    Update a registered model's settings or metadata.

    Request Body (partial):
        {
          "display_name": "New name",
          "roles": ["triage", "deep_scan"],
          "parser_id": "json_schema",
          "settings": { "runtime": { "device": "cpu" } },
          "merge_settings": true
        }
    """
    def _merge_settings(base: dict, updates: dict) -> dict:
        merged = dict(base or {})
        updates = updates or {}
        for key in ("runtime", "hf_kwargs", "generation_kwargs", "parser_config"):
            if isinstance(updates.get(key), dict):
                merged[key] = {**merged.get(key, {}), **updates[key]}
        for key, value in updates.items():
            if key in ("runtime", "hf_kwargs", "generation_kwargs", "parser_config") and isinstance(value, dict):
                continue
            merged[key] = value
        return merged

    try:
        data = request.get_json(silent=True) or {}
        registry = ModelRegistryV2()
        existing = registry.get_model(model_id)
        if not existing:
            return jsonify({"error": "Model not found"}), 404

        display_name = data.get("display_name", existing.display_name)
        model_name = data.get("model_name", existing.model_name)
        parser_id = data.get("parser_id", existing.parser_id)

        roles = existing.roles
        if "roles" in data:
            roles = parse_roles(data.get("roles") or [])
            if not roles:
                return jsonify({"error": "roles must not be empty"}), 400

        settings = existing.settings or {}
        if "settings" in data:
            if data.get("merge_settings", True):
                settings = _merge_settings(settings, data.get("settings") or {})
            else:
                settings = data.get("settings") or {}

        status = existing.status
        if "status" in data:
            status = ModelStatus(data.get("status"))

        availability = existing.availability
        if "availability" in data:
            availability = ModelAvailability(data.get("availability"))

        model = registry.register_model(
            model_id=existing.model_id,
            model_type=existing.model_type,
            provider_id=existing.provider_id,
            model_name=model_name,
            display_name=display_name,
            roles=roles,
            parser_id=parser_id,
            settings=settings,
            status=status,
            availability=availability,
        )
        DEFAULT_RUNTIME_MANAGER.clear_model(model_id)

        return jsonify({"model": model.model_dump()})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Update model failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>", methods=["DELETE"])
def delete_model(model_id: str) -> Any:
    """Delete a registered model."""
    try:
        registry = ModelRegistryV2()
        success = registry.delete_model(model_id)

        if not success:
            return jsonify({"error": "Model not found"}), 404
        DEFAULT_RUNTIME_MANAGER.clear_model(model_id)

        return jsonify({"message": "Model deleted successfully"})

    except Exception as e:
        logger.error(f"Delete model failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/status", methods=["PUT"])
def update_model_status(model_id: str) -> Any:
    """
    Update model status.

    Request Body:
        {"status": "registered|disabled|unavailable"}
    """
    try:
        data = request.get_json()
        status = ModelStatus(data["status"])

        registry = ModelRegistryV2()
        success = registry.update_status(model_id, status)

        if not success:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({"message": "Status updated successfully"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Update status failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/test", methods=["POST"])
def test_model() -> Any:
    """
    Test a model with a sample prompt.

    Request Body:
        {
          "model_id": "ollama:qwen2.5-coder",
          "prompt": "def add(a, b): return a + b"
        }

    Returns:
        {
          "output": "<raw model output>",
          "success": true
        }
    """
    try:
        data = request.get_json()
        model_id = data.get("model_id")
        prompt = data.get("prompt")

        if not model_id or not prompt:
            return jsonify({"error": "model_id and prompt required"}), 400

        # Get model from registry
        registry = ModelRegistryV2()
        model = registry.get_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        engine = ModelExecutionEngine(registry)
        try:
            result = engine.run_model_sync(
                model=model,
                code=prompt,
                file_path=data.get("file_path", "sample.py"),
                role=model.roles[0] if model.roles else None,
            )
            return jsonify({
                "success": True,
                "result": result.model_dump(),
                "model": model.model_dump(),
            })
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        logger.error(f"Test model failed: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# Runtime management routes (defined before /<model_id> routes for proper matching)
# ==============================================================================

@models_bp.route("/runtime/loaded", methods=["GET"])
def list_loaded_models() -> Any:
    """
    List all currently loaded models with their runtime status.

    Returns:
        {
          "models": [
            {
              "model_id": "hf:codebert-insecure",
              "model_name": "mrm8488/codebert-base-finetuned-detect-insecure-code",
              "model_type": "hf_local",
              "device": "cuda:0",
              "vram_mb": 512,
              "last_used": 1706123456.789
            },
            ...
          ],
          "total_vram_mb": 1024
        }
    """
    try:
        models = DEFAULT_RUNTIME_MANAGER.list_loaded_models()
        gpu_info = DEFAULT_RUNTIME_MANAGER.get_gpu_info()

        return jsonify({
            "models": models,
            "total_vram_mb": gpu_info.get("used_vram_mb", 0),
            "gpu": gpu_info
        })

    except Exception as e:
        logger.error(f"List loaded models failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/runtime/clear", methods=["POST"])
def clear_all_runtimes() -> Any:
    """
    Unload all models and clear the runtime cache.

    This is useful for:
    - Freeing all GPU memory
    - Resetting the model cache after configuration changes

    Returns:
        {
          "success": true,
          "message": "All model runtimes cleared"
        }
    """
    try:
        DEFAULT_RUNTIME_MANAGER.clear_all()
        return jsonify({
            "success": True,
            "message": "All model runtimes cleared"
        })

    except Exception as e:
        logger.error(f"Clear runtimes failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/hf/presets", methods=["GET"])
def list_hf_presets() -> Any:
    """
    List HuggingFace model presets.

    Returns:
        {
          "presets": [
            {
              "model_id": "mrm8488/codebert-base-finetuned-detect-insecure-code",
              "task_type": "text-classification",
              "name": "CodeBERT Insecure Code Detector",
              "description": "...",
              "recommended_roles": ["triage"],
              "recommended_parser": "hf_classification"
            }
          ]
        }
    """
    presets = [
        # Classification models (triage)
        {
            **CODEBERT_INSECURE,
            "preset_id": "codebert_insecure",
            "recommended_roles": ["triage"],
            "recommended_parser": "hf_classification",
            "parser_config": {
                "positive_labels": ["LABEL_1", "VULNERABLE", "INSECURE"],
                "negative_labels": ["LABEL_0", "SAFE", "SECURE"],
                "threshold": 0.5,
            },
        },
        {
            **CODEBERT_PRIMEVUL,
            "preset_id": "codebert_primevul",
            "recommended_roles": ["triage"],
            "recommended_parser": "hf_classification",
            "parser_config": {
                "positive_labels": ["LABEL_1", "1", "vulnerable"],
                "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
                "threshold": 0.5,
            },
        },
        {
            **VULBERTA_DEVIGN,
            "preset_id": "vulberta_devign",
            "recommended_roles": ["triage"],
            "recommended_parser": "hf_classification",
            "parser_config": {
                "positive_labels": ["LABEL_1"],
                "negative_labels": ["LABEL_0"],
                "threshold": 0.5,
            },
        },
        {
            **UNIXCODER_PRIMEVUL,
            "preset_id": "unixcoder_primevul",
            "recommended_roles": ["triage"],
            "recommended_parser": "hf_classification",
            "parser_config": {
                "positive_labels": ["LABEL_1", "1", "vulnerable"],
                "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
                "threshold": 0.5,
            },
        },
        # Generative models (deep_scan)
        {
            **CODEASTRA_7B,
            "preset_id": "codeastra_7b",
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        },
        {
            **QWEN25_CODER_7B,
            "preset_id": "qwen25_coder_7b",
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        },
        {
            **DEEPSEEK_CODER_V2_LITE,
            "preset_id": "deepseek_coder_v2_lite",
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        },
        {
            **STARCODER2_15B,
            "preset_id": "starcoder2_15b",
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        },
        {
            **PHI35_MINI,
            "preset_id": "phi35_mini",
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        },
        {
            **QWEN25_05B,
            "preset_id": "qwen25_05b",
            "recommended_roles": ["triage", "deep_scan"],
            "recommended_parser": "json_schema",
        },
    ]

    return jsonify({"presets": presets})


@models_bp.route("/hf/register_preset", methods=["POST"])
def register_hf_preset() -> Any:
    """
    Register a HuggingFace preset model.

    Request Body:
        {
          "preset_id": "codebert_insecure",  // accepts aliases (see below)
          "display_name": "Custom Name (optional)"
        }

    Returns:
        {"model": <ModelRecord>}
    """
    def normalize_preset_id(preset_id: str) -> str:
        """Map UI/alias IDs to canonical preset IDs."""
        pid = (preset_id or "").strip().lower()
        alias_map = {
            # CodeBERT Insecure
            "codebert_insecure": "codebert_insecure",
            "mrm8488/codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
            "codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
            "codebert_base_finetuned_detect_insecure_code": "codebert_insecure",
            # CodeAstra 7B
            "codeastra_7b": "codeastra_7b",
            "rootxhacker/codeastra-7b": "codeastra_7b",
            "codeastra-7b": "codeastra_7b",
            # CodeBERT PrimeVul (custom architecture with custom_loading)
            "codebert_primevul": "codebert_primevul",
            "mahdin70/codebert-primevul-bigvul": "codebert_primevul",
            "codebert-primevul-bigvul": "codebert_primevul",
            "codebert_primevul_bigvul": "codebert_primevul",
            # VulBERTa Devign
            "vulberta_devign": "vulberta_devign",
            "claudios/vulberta-mlp-devign": "vulberta_devign",
            "vulberta-mlp-devign": "vulberta_devign",
            "vulberta_mlp_devign": "vulberta_devign",
            # UnixCoder PrimeVul
            "unixcoder_primevul": "unixcoder_primevul",
            "mahdin70/unixcoder-primevul-bigvul": "unixcoder_primevul",
            "unixcoder-primevul-bigvul": "unixcoder_primevul",
            "unixcoder_primevul_bigvul": "unixcoder_primevul",
            # Qwen 2.5 Coder 7B
            "qwen25_coder_7b": "qwen25_coder_7b",
            "qwen/qwen2.5-coder-7b-instruct": "qwen25_coder_7b",
            "qwen2.5-coder-7b-instruct": "qwen25_coder_7b",
            "qwen2.5-coder-7b": "qwen25_coder_7b",
            # DeepSeek Coder V2 Lite
            "deepseek_coder_v2_lite": "deepseek_coder_v2_lite",
            "deepseek-ai/deepseek-coder-v2-lite-instruct": "deepseek_coder_v2_lite",
            "deepseek-coder-v2-lite-instruct": "deepseek_coder_v2_lite",
            "deepseek-coder-v2-lite": "deepseek_coder_v2_lite",
            # StarCoder2 15B
            "starcoder2_15b": "starcoder2_15b",
            "bigcode/starcoder2-15b-instruct-v0.1": "starcoder2_15b",
            "starcoder2-15b-instruct-v0.1": "starcoder2_15b",
            "starcoder2-15b-instruct": "starcoder2_15b",
            "starcoder2-15b": "starcoder2_15b",
            # Phi-3.5 Mini
            "phi35_mini": "phi35_mini",
            "microsoft/phi-3.5-mini-instruct": "phi35_mini",
            "phi-3.5-mini-instruct": "phi35_mini",
            "phi-3.5-mini": "phi35_mini",
            # Qwen 2.5 0.5B
            "qwen25_05b": "qwen25_05b",
            "qwen/qwen2.5-0.5b-instruct": "qwen25_05b",
            "qwen2.5-0.5b-instruct": "qwen25_05b",
            "qwen2.5-0.5b": "qwen25_05b",
        }
        return alias_map.get(pid, pid)

    try:
        data = request.get_json()
        preset_id = normalize_preset_id(data.get("preset_id"))
        parser_config = data.get("parser_config") or {}
        user_settings = data.get("settings") or {}

        if preset_id == "codebert_insecure":
            preset = CODEBERT_INSECURE
            roles = [ModelRole.TRIAGE]
            parser_id = "hf_classification"
            parser_config = {
                "positive_labels": ["LABEL_1", "VULNERABLE", "INSECURE"],
                "negative_labels": ["LABEL_0", "SAFE", "SECURE"],
                "threshold": 0.5,
            }
        elif preset_id == "codeastra_7b":
            preset = CODEASTRA_7B
            roles = [ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        elif preset_id == "codebert_primevul":
            preset = CODEBERT_PRIMEVUL
            roles = [ModelRole.TRIAGE]
            parser_id = "hf_classification"
            parser_config = {
                "positive_labels": ["LABEL_1", "1", "vulnerable"],
                "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
                "threshold": 0.5,
            }
        elif preset_id == "vulberta_devign":
            preset = VULBERTA_DEVIGN
            roles = [ModelRole.TRIAGE]
            parser_id = "hf_classification"
            parser_config = {
                # Standard LABEL_0/LABEL_1 format
                "positive_labels": ["LABEL_1"],
                "negative_labels": ["LABEL_0"],
                "threshold": 0.5,
            }
        elif preset_id == "unixcoder_primevul":
            preset = UNIXCODER_PRIMEVUL
            roles = [ModelRole.TRIAGE]
            parser_id = "hf_classification"
            parser_config = {
                "positive_labels": ["LABEL_1", "1", "vulnerable"],
                "negative_labels": ["LABEL_0", "0", "non-vulnerable"],
                "threshold": 0.5,
            }
        elif preset_id == "qwen25_coder_7b":
            preset = QWEN25_CODER_7B
            roles = [ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        elif preset_id == "deepseek_coder_v2_lite":
            preset = DEEPSEEK_CODER_V2_LITE
            roles = [ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        elif preset_id == "starcoder2_15b":
            preset = STARCODER2_15B
            roles = [ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        elif preset_id == "phi35_mini":
            preset = PHI35_MINI
            roles = [ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        elif preset_id == "qwen25_05b":
            preset = QWEN25_05B
            roles = [ModelRole.TRIAGE, ModelRole.DEEP_SCAN]
            parser_id = "json_schema"
        else:
            return jsonify({"error": f"Unknown preset: {preset_id}"}), 400

        display_name = data.get("display_name", preset["name"])
        model_id = f"hf:{preset_id}"

        base_settings = {
            "task_type": preset["task_type"],
            "prompt_template": data.get("prompt_template"),
        }

        # Include adapter/base/hf_kwargs if provided by preset
        if preset_id == "codeastra_7b":
            if preset.get("adapter_id"):
                base_settings["adapter_id"] = preset["adapter_id"]
            if preset.get("base_model_id"):
                base_settings["base_model_id"] = preset["base_model_id"]
            if preset.get("hf_kwargs"):
                base_settings["hf_kwargs"] = preset.get("hf_kwargs", {})
            if preset.get("generation_kwargs"):
                base_settings["generation_kwargs"] = preset.get("generation_kwargs", {})
        else:
            # Generic include if present (including custom_loading for custom architectures)
            for key in ["adapter_id", "base_model_id", "hf_kwargs", "generation_kwargs", "custom_loading", "small_model"]:
                if preset.get(key) is not None:
                    base_settings[key] = preset.get(key)

        # Merge settings, allowing user overrides and hf_kwargs merging
        settings = {**base_settings, **user_settings}
        if user_settings.get("hf_kwargs") and base_settings.get("hf_kwargs"):
            merged_hf_kwargs = {
                **base_settings["hf_kwargs"],
                **user_settings["hf_kwargs"],
            }
            settings["hf_kwargs"] = merged_hf_kwargs
        if user_settings.get("generation_kwargs") and base_settings.get("generation_kwargs"):
            merged_gen_kwargs = {
                **base_settings["generation_kwargs"],
                **user_settings["generation_kwargs"],
            }
            settings["generation_kwargs"] = merged_gen_kwargs

        # Ensure HuggingFace provider exists
        registry = ModelRegistryV2()
        model = registry.register_model(
            model_id=model_id,
            model_type=ModelType.HF_LOCAL,
            provider_id="huggingface",
            model_name=preset["model_id"],
            display_name=display_name,
            roles=roles,
            parser_id=parser_id,
            settings=settings,
            parser_config=parser_config,
        )

        return jsonify({"model": model.model_dump()})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"HF preset registration failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/ml/presets", methods=["GET"])
def list_ml_presets() -> Any:
    """
    List ML model presets.

    Returns:
        {
          "presets": [
            {
              "preset_id": "kaggle_rf_cfunctions",
              "name": "Kaggle RF C-Functions Predictor",
              "description": "...",
              "model_url": "...",
              "recommended_roles": ["triage"]
            }
          ]
        }
    """
    ml_entries = get_ml_catalog_entries()
    presets = []

    for entry in ml_entries:
        presets.append({
            "preset_id": entry.get("catalog_id"),
            "name": entry.get("display_name"),
            "description": entry.get("description"),
            "model_url": entry.get("settings", {}).get("model_url"),
            "recommended_roles": entry.get("roles", ["triage"]),
            "size_mb": entry.get("size_mb", 0),
            "requires_gpu": entry.get("requires_gpu", False),
        })

    return jsonify({"presets": presets})


@models_bp.route("/ml/register_preset", methods=["POST"])
def register_ml_preset() -> Any:
    """
    Register an ML model preset.

    This downloads the model and registers it in the database.

    Request Body:
        {
          "preset_id": "kaggle_rf_cfunctions",
          "display_name": "Custom Name (optional)",
          "download": true  // Whether to download immediately (default: true)
        }

    Returns:
        {"model": <ModelRecord>, "download_result": {...}}
    """
    try:
        data = request.get_json(silent=True) or {}
        preset_id = data.get("preset_id")
        download = data.get("download", True)

        if not preset_id:
            return jsonify({"error": "preset_id is required"}), 400

        entry = get_catalog_entry(preset_id)
        if not entry:
            return jsonify({"error": f"Unknown preset: {preset_id}"}), 404

        category = entry.get("category")
        if category != CatalogCategory.CLASSIC_ML:
            return jsonify({"error": f"Preset {preset_id} is not an ML model"}), 400

        # Build model_id
        model_id = f"ml:{preset_id}"

        # Parse roles
        role_strings = entry.get("roles", ["triage"])
        roles = parse_roles(role_strings)

        # Build settings
        settings = {
            "task_type": entry.get("task_type"),
            "tool_id": entry.get("tool_id"),
        }
        if entry.get("settings"):
            settings.update(entry["settings"])

        # Merge user settings
        if data.get("settings"):
            settings.update(data["settings"])

        # Register in database
        registry = ModelRegistryV2()
        display_name = data.get("display_name", entry.get("display_name"))

        model = registry.register_model(
            model_id=model_id,
            model_type=ModelType.TOOL_ML,
            provider_id=entry.get("provider_id", "tool_ml"),
            model_name=entry.get("model_name", preset_id),
            display_name=display_name,
            roles=roles,
            parser_id=entry.get("parser_id", "tool_result"),
            settings=settings,
            parser_config=entry.get("parser_config"),
        )

        # Download if requested
        download_result = None
        if download:
            from aegis.tools.builtin.sklearn_tool import SklearnTool
            tool = SklearnTool(config=settings)
            download_result = tool.prefetch()

        # Compute status
        status = is_model_ready(entry, registry)

        return jsonify({
            "model": model.model_dump(),
            "status": status.value if isinstance(status, CatalogStatus) else status,
            "download_result": download_result,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"ML preset registration failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/health", methods=["POST"])
def health_check_model(model_id: str) -> Any:
    """
    Perform health check on a specific model.

    Args:
        model_id: Model identifier (URL parameter)

    Body (optional):
        {
          "timeout": 30  # Health check timeout in seconds (default: 30)
        }

    Returns:
        {
          "model_id": "ollama:qwen2.5-coder:7b",
          "healthy": true,
          "response_time_ms": 1234.56,
          "error": null
        }
    """
    try:
        registry = ModelRegistryV2()
        model = registry.get_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json(silent=True) or {}
        timeout = data.get("timeout", 30)

        engine = ModelExecutionEngine(registry)
        health_status = engine.health_check_model_sync(model, timeout=timeout)

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"Health check failed for {model_id}: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/health/all", methods=["POST"])
def health_check_all_models() -> Any:
    """
    Perform health checks on all registered models.

    Body (optional):
        {
          "timeout": 30,  # Health check timeout per model (default: 30)
          "status_filter": "registered"  # Only check models with this status
        }

    Returns:
        {
          "results": [
            {
              "model_id": "ollama:qwen2.5-coder:7b",
              "healthy": true,
              "response_time_ms": 1234.56,
              "error": null
            },
            ...
          ],
          "summary": {
            "total": 3,
            "healthy": 2,
            "unhealthy": 1
          }
        }
    """
    try:
        data = request.get_json(silent=True) or {}
        timeout = data.get("timeout", 30)
        status_filter = data.get("status_filter")

        registry = ModelRegistryV2()
        engine = ModelExecutionEngine(registry)

        # Get models to check
        if status_filter:
            try:
                status = ModelStatus(status_filter)
                models = registry.list_models(status=status)
            except ValueError:
                return jsonify({"error": f"Invalid status: {status_filter}"}), 400
        else:
            models = registry.list_models()

        if not models:
            return jsonify({
                "results": [],
                "summary": {"total": 0, "healthy": 0, "unhealthy": 0}
            })

        # Run health checks concurrently
        import asyncio

        async def check_all():
            tasks = [engine.health_check_model(model, timeout) for model in models]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(check_all())

        # Filter out exceptions and build response
        health_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check exception: {result}")
                continue
            health_results.append(result)

        # Calculate summary
        healthy_count = sum(1 for r in health_results if r.get("healthy"))
        unhealthy_count = len(health_results) - healthy_count

        return jsonify({
            "results": health_results,
            "summary": {
                "total": len(health_results),
                "healthy": healthy_count,
                "unhealthy": unhealthy_count
            }
        })

    except Exception as e:
        logger.error(f"Health check all failed: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/warmup", methods=["POST"])
def warmup_model(model_id: str) -> Any:
    """
    Pre-load a model into GPU/CPU memory and run warmup inference.

    This is useful for:
    - Pre-loading models before scans to reduce first-request latency
    - Compiling CUDA kernels ahead of time
    - Validating model availability and device allocation

    Args:
        model_id: Model identifier (URL parameter)

    Body (optional):
        {
          "dummy_input": "def test(): pass"  # Custom warmup input
        }

    Returns:
        {
          "model_id": "hf:codebert-insecure",
          "success": true,
          "cached": false,
          "device": "cuda:0",
          "vram_mb": 512,
          "load_time_ms": 3500,
          "warmup_time_ms": 250,
          "error": null
        }
    """
    try:
        registry = ModelRegistryV2()
        model = registry.get_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json(silent=True) or {}
        dummy_input = data.get("dummy_input")

        result = DEFAULT_RUNTIME_MANAGER.warmup_model(model, dummy_input)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Warmup failed for {model_id}: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/unload", methods=["POST"])
def unload_model(model_id: str) -> Any:
    """
    Unload a model from memory to free GPU/CPU resources.

    This is useful for:
    - Freeing GPU VRAM when switching between models
    - Managing memory on resource-constrained systems
    - Forcing a model reload with updated settings

    Args:
        model_id: Model identifier (URL parameter)

    Body (optional):
        {
          "clear_cuda_cache": true  # Also clear CUDA cache (default: true)
        }

    Returns:
        {
          "model_id": "hf:codebert-insecure",
          "success": true,
          "freed_vram_mb": 512,
          "runtimes_removed": 1,
          "error": null
        }
    """
    try:
        data = request.get_json(silent=True) or {}
        clear_cuda_cache = data.get("clear_cuda_cache", True)

        result = DEFAULT_RUNTIME_MANAGER.unload_model(model_id, clear_cuda_cache)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Unload failed for {model_id}: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/runtime-status", methods=["GET"])
def get_model_runtime_status(model_id: str) -> Any:
    """
    Get the current runtime status of a model.

    Returns whether the model is loaded, which device it's on,
    and current VRAM usage.

    Args:
        model_id: Model identifier (URL parameter)

    Returns:
        {
          "model_id": "hf:codebert-insecure",
          "loaded": true,
          "device": "cuda:0",
          "vram_mb": 512,
          "last_used": 1706123456.789,
          "runtime_count": 1
        }
    """
    try:
        status = DEFAULT_RUNTIME_MANAGER.get_model_status(model_id)
        return jsonify(status)

    except Exception as e:
        logger.error(f"Get status failed for {model_id}: {e}")
        return jsonify({"error": str(e)}), 500
