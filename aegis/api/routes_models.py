"""API routes for model management dashboard."""

import logging
from datetime import datetime
from typing import Any, List
from flask import Blueprint, jsonify, request, current_app

from aegis.models.schema import ModelType, ModelRole, ModelStatus, ModelAvailability
from aegis.models.registry import ModelRegistryV2
from aegis.models.discovery.ollama import OllamaDiscoveryClient
from aegis.models.executor import ModelExecutionEngine
from aegis.providers.hf_local import create_hf_provider, CODEBERT_INSECURE, CODEASTRA_7B
from aegis.connectors.ollama_connector import OllamaConnector

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__, url_prefix="/api/models")


# Legacy role mapping for backward compatibility
ROLE_MAPPING = {
    "scan": ModelRole.DEEP_SCAN,
    "triage": ModelRole.TRIAGE,
    "deep_scan": ModelRole.DEEP_SCAN,
    "judge": ModelRole.JUDGE,
    "explain": ModelRole.EXPLAIN,
    "custom": ModelRole.CUSTOM,
}


def parse_role(role_str: str) -> ModelRole:
    """Parse role string with legacy mapping support."""
    try:
        return ModelRole(role_str)
    except ValueError:
        mapped = ROLE_MAPPING.get(role_str.lower())
        if mapped:
            return mapped
        raise ValueError(f"Invalid role: {role_str}")


def parse_roles(role_strings: List[str]) -> List[ModelRole]:
    """Parse list of role strings with legacy mapping support."""
    return [parse_role(r) for r in role_strings]


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
    data = request.get_json() or {}
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


@models_bp.route("/<model_id>", methods=["DELETE"])
def delete_model(model_id: str) -> Any:
    """Delete a registered model."""
    try:
        registry = ModelRegistryV2()
        success = registry.delete_model(model_id)

        if not success:
            return jsonify({"error": "Model not found"}), 404

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
        {
            **CODEBERT_INSECURE,
            "recommended_roles": ["triage"],
            "recommended_parser": "hf_classification",
            "parser_config": {
                "positive_labels": ["LABEL_1", "VULNERABLE", "INSECURE"],
                "negative_labels": ["LABEL_0", "SAFE", "SECURE"],
                "threshold": 0.5,
            },
        },
        {
            **CODEASTRA_7B,
            "recommended_roles": ["deep_scan"],
            "recommended_parser": "json_schema",
        }
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
            "codebert_insecure": "codebert_insecure",
            "mrm8488/codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
            "codebert-base-finetuned-detect-insecure-code": "codebert_insecure",
            "codebert_base_finetuned_detect_insecure_code": "codebert_insecure",
            "codeastra_7b": "codeastra_7b",
            "rootxhacker/codeastra-7b": "codeastra_7b",
            "codeastra-7b": "codeastra_7b",
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
        else:
            # Generic include if present
            for key in ["adapter_id", "base_model_id", "hf_kwargs"]:
                if preset.get(key):
                    base_settings[key] = preset.get(key)

        # Merge settings, allowing user overrides and hf_kwargs merging
        settings = {**base_settings, **user_settings}
        if user_settings.get("hf_kwargs") and base_settings.get("hf_kwargs"):
            merged_hf_kwargs = {
                **base_settings["hf_kwargs"],
                **user_settings["hf_kwargs"],
            }
            settings["hf_kwargs"] = merged_hf_kwargs

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

        data = request.get_json() or {}
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
        data = request.get_json() or {}
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
