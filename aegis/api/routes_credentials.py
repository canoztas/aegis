"""API routes for credential management."""

import logging
from flask import Blueprint, jsonify, request
from typing import Any, Dict

logger = logging.getLogger(__name__)

bp = Blueprint("credentials", __name__, url_prefix="/api/credentials")


@bp.route("", methods=["GET"])
def list_credentials() -> Any:
    """
    List stored credentials (without values).

    Query params:
        provider: Filter by provider (optional)

    Returns:
        200: List of credentials
    """
    try:
        from aegis.models.credentials import DEFAULT_CREDENTIAL_MANAGER

        provider = request.args.get("provider")
        credentials = DEFAULT_CREDENTIAL_MANAGER.list_credentials(provider=provider)

        return jsonify({
            "credentials": credentials,
            "count": len(credentials),
        }), 200

    except Exception as e:
        logger.error(f"Failed to list credentials: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("", methods=["POST"])
def store_credential() -> Any:
    """
    Store API credential.

    Body:
        {
            "provider": "openai|anthropic|google",
            "key_name": "api_key",
            "key_value": "sk-...",
            "encrypt": true
        }

    Returns:
        201: Credential stored
        400: Invalid request
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        provider = data.get("provider")
        key_name = data.get("key_name")
        key_value = data.get("key_value")
        encrypt = data.get("encrypt", True)

        if not provider or not key_name or not key_value:
            return jsonify({"error": "provider, key_name, and key_value required"}), 400

        from aegis.models.credentials import DEFAULT_CREDENTIAL_MANAGER

        DEFAULT_CREDENTIAL_MANAGER.store_credential(
            provider=provider,
            key_name=key_name,
            key_value=key_value,
            encrypt=encrypt,
        )

        return jsonify({
            "message": "Credential stored successfully",
            "provider": provider,
            "key_name": key_name,
        }), 201

    except Exception as e:
        logger.error(f"Failed to store credential: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/<provider>/<key_name>", methods=["DELETE"])
def delete_credential(provider: str, key_name: str) -> Any:
    """
    Delete API credential.

    Path params:
        provider: Provider name
        key_name: Credential key name

    Returns:
        200: Credential deleted
        404: Credential not found
    """
    try:
        from aegis.models.credentials import DEFAULT_CREDENTIAL_MANAGER

        # Check if exists
        existing = DEFAULT_CREDENTIAL_MANAGER.get_credential(provider, key_name)
        if not existing:
            return jsonify({"error": "Credential not found"}), 404

        DEFAULT_CREDENTIAL_MANAGER.delete_credential(provider, key_name)

        return jsonify({
            "message": "Credential deleted successfully",
            "provider": provider,
            "key_name": key_name,
        }), 200

    except Exception as e:
        logger.error(f"Failed to delete credential: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/validate", methods=["POST"])
def validate_credential() -> Any:
    """
    Validate API credential.

    Body:
        {
            "provider": "openai|anthropic|google",
            "api_key": "sk-..."
        }

    Returns:
        200: Validation result
        400: Invalid request
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        provider = data.get("provider")
        api_key = data.get("api_key")

        if not provider or not api_key:
            return jsonify({"error": "provider and api_key required"}), 400

        from aegis.models.credentials import DEFAULT_CREDENTIAL_MANAGER

        result = DEFAULT_CREDENTIAL_MANAGER.validate_credential(provider, api_key)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Failed to validate credential: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/rotate", methods=["POST"])
def rotate_credential() -> Any:
    """
    Rotate API credential (validate new, then update).

    Body:
        {
            "provider": "openai|anthropic|google",
            "key_name": "api_key",
            "new_value": "sk-new..."
        }

    Returns:
        200: Credential rotated
        400: Invalid request or validation failed
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        provider = data.get("provider")
        key_name = data.get("key_name")
        new_value = data.get("new_value")

        if not provider or not key_name or not new_value:
            return jsonify({"error": "provider, key_name, and new_value required"}), 400

        from aegis.models.credentials import DEFAULT_CREDENTIAL_MANAGER

        try:
            DEFAULT_CREDENTIAL_MANAGER.rotate_credential(provider, key_name, new_value)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        return jsonify({
            "message": "Credential rotated successfully",
            "provider": provider,
            "key_name": key_name,
        }), 200

    except Exception as e:
        logger.error(f"Failed to rotate credential: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/usage", methods=["GET"])
def get_usage() -> Any:
    """
    Get API usage and costs.

    Query params:
        provider: Filter by provider (optional)
        start_date: Start date ISO format (optional)
        end_date: End date ISO format (optional)

    Returns:
        200: Usage statistics
    """
    try:
        from aegis.models.cost_tracker import DEFAULT_COST_TRACKER

        provider = request.args.get("provider")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        usage = DEFAULT_COST_TRACKER.get_provider_usage(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
        )

        return jsonify(usage), 200

    except Exception as e:
        logger.error(f"Failed to get usage: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/usage/scan/<scan_id>", methods=["GET"])
def get_scan_usage(scan_id: str) -> Any:
    """
    Get API usage and cost for a specific scan.

    Path params:
        scan_id: Scan ID

    Returns:
        200: Scan usage statistics
    """
    try:
        from aegis.models.cost_tracker import DEFAULT_COST_TRACKER

        usage = DEFAULT_COST_TRACKER.get_scan_cost(scan_id)

        return jsonify(usage), 200

    except Exception as e:
        logger.error(f"Failed to get scan usage: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/budget", methods=["POST"])
def check_budget() -> Any:
    """
    Check budget status.

    Body:
        {
            "budget_usd": 100.0,
            "start_date": "2025-01-01T00:00:00" (optional)
        }

    Returns:
        200: Budget status
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        budget_usd = data.get("budget_usd")
        start_date = data.get("start_date")

        if budget_usd is None:
            return jsonify({"error": "budget_usd required"}), 400

        from aegis.models.cost_tracker import DEFAULT_COST_TRACKER

        status = DEFAULT_COST_TRACKER.get_budget_status(
            budget_usd=float(budget_usd),
            start_date=start_date,
        )

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Failed to check budget: {e}")
        return jsonify({"error": str(e)}), 500
