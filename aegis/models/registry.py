"""Model Registry V2 - Enhanced model lifecycle management."""

import json
import logging
from typing import List, Optional, Dict, Any, Iterable
from datetime import datetime

from aegis.database import get_db
from aegis.models.schema import (
    ModelType,
    ModelRole,
    ModelRecord,
    ModelStatus,
    ModelAvailability,
)

logger = logging.getLogger(__name__)


class ModelRegistryV2:
    """
    Enhanced model registry with support for:
    - Multiple roles per model
    - Parser assignment
    - Model type tracking
    - Status management
    """

    def __init__(self):
        self.db = get_db()
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure new registry columns exist (idempotent)."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(models)")
            columns = {row["name"] for row in cursor.fetchall()}

            # Add Model Registry V2 columns if missing (idempotent)
            if "roles_json" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN roles_json TEXT DEFAULT '[]'")
            if "parser_id" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN parser_id TEXT DEFAULT NULL")
            if "model_type" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN model_type TEXT DEFAULT 'ollama_local'")
            if "status" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN status TEXT DEFAULT 'registered'")
            if "availability" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN availability TEXT DEFAULT 'unknown'")
            if "availability_checked_at" not in columns:
                cursor.execute("ALTER TABLE models ADD COLUMN availability_checked_at TIMESTAMP NULL")

            conn.commit()

    def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        provider_id: str,
        model_name: str,
        display_name: str,
        roles: List[ModelRole],
        parser_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        status: ModelStatus = ModelStatus.REGISTERED,
        availability: ModelAvailability = ModelAvailability.UNKNOWN,
        parser_config: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """
        Register a new model or update existing registration.

        Args:
            model_id: Unique identifier (e.g., 'ollama:qwen2.5-coder')
            model_type: Type of model (OLLAMA_LOCAL, HF_LOCAL, etc.)
            provider_id: Provider identifier
            model_name: Actual model name at provider
            display_name: Human-readable name
            roles: List of roles this model can fulfill
            parser_id: Optional parser identifier
            settings: Optional model settings (temperature, max_tokens, etc.)
            status: Registration status
            availability: Availability at provider
            parser_config: Optional parser configuration to store alongside settings

        Returns:
            ModelRecord of the registered model

        Raises:
            ValueError: If roles list is empty or invalid provider
        """
        if not roles:
            raise ValueError("Model must have at least one role")

        settings = settings or {}
        if parser_config:
            # Keep parser config nested to avoid schema churn
            settings = {**settings, "parser_config": parser_config}
        roles_json = json.dumps([r.value if isinstance(r, ModelRole) else r for r in roles])
        config_json = json.dumps(settings)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get or create provider
            cursor.execute("SELECT id, type FROM providers WHERE name = ?", (provider_id,))
            provider_row = cursor.fetchone()

            if not provider_row:
                logger.info(f"Provider '{provider_id}' missing. Creating with inferred type.")
                cursor.execute(
                    """
                    INSERT INTO providers (name, type, base_url, enabled)
                    VALUES (?, ?, ?, 1)
                    """,
                    (
                        provider_id,
                        self._infer_provider_type(model_type),
                        settings.get("base_url"),
                    ),
                )
                provider_db_id = cursor.lastrowid
            else:
                provider_db_id = provider_row["id"]

            # Check if model already exists
            cursor.execute("SELECT id FROM models WHERE model_id = ?", (model_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute(
                    """
                    UPDATE models
                    SET model_type = ?,
                        provider_id = ?,
                        model_name = ?,
                        display_name = ?,
                        roles_json = ?,
                        parser_id = ?,
                        config_json = ?,
                        status = ?,
                        availability = ?,
                        availability_checked_at = CURRENT_TIMESTAMP,
                        role = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE model_id = ?
                    """,
                    (
                        model_type.value,
                        provider_db_id,
                        model_name,
                        display_name,
                        roles_json,
                        parser_id,
                        config_json,
                        status.value,
                        availability.value,
                        roles[0].value if roles else None,
                        model_id,
                    ),
                )
                record_id = existing["id"]
            else:
                # Insert new
                cursor.execute(
                    """
                    INSERT INTO models (
                        provider_id, model_id, model_name, display_name,
                        model_type, roles_json, parser_id, config_json, status,
                        availability, availability_checked_at, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """,
                    (
                        provider_db_id,
                        model_id,
                        model_name,
                        display_name,
                        model_type.value,
                        roles_json,
                        parser_id,
                        config_json,
                        status.value,
                        availability.value,
                        roles[0].value if roles else None,
                    ),
                )
                record_id = cursor.lastrowid

            conn.commit()

        logger.info(f"Registered model: {model_id} with roles {roles}")
        return self.get_model(model_id)

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Get a registered model by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.*, p.name as provider_name, p.config_json as provider_config_json, p.base_url as provider_base_url
                FROM models m
                JOIN providers p ON m.provider_id = p.id
                WHERE m.model_id = ?
                """,
                (model_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_record(row)

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        role: Optional[ModelRole] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelRecord]:
        """
        List registered models with optional filters.

        Args:
            model_type: Filter by model type
            role: Filter by role (models that support this role)
            status: Filter by status

        Returns:
            List of ModelRecord objects
        """
        query = """
            SELECT m.*, p.name as provider_name, p.config_json as provider_config_json, p.base_url as provider_base_url
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE 1=1
        """
        params = []

        if model_type:
            query += " AND m.model_type = ?"
            params.append(model_type.value)

        if status:
            query += " AND m.status = ?"
            params.append(status.value)

        query += " ORDER BY m.created_at DESC"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        models = [self._row_to_record(row) for row in rows]

        # Filter by role if specified (requires JSON parsing)
        if role:
            models = [m for m in models if role in m.roles]

        return models

    def get_models_for_role(self, role: ModelRole, enabled_only: bool = True) -> List[ModelRecord]:
        """
        Get all models that can fulfill a specific role.

        Args:
            role: The role to filter by
            enabled_only: Only return enabled models

        Returns:
            List of models supporting the role, sorted by creation date
        """
        status = ModelStatus.REGISTERED if enabled_only else None
        all_models = self.list_models(status=status)
        return [m for m in all_models if role in m.roles]

    def get_best_model_for_role(self, role: ModelRole) -> Optional[ModelRecord]:
        """
        Get the most recently registered model for a role.

        Args:
            role: Role to resolve

        Returns:
            ModelRecord or None
        """
        models = self.get_models_for_role(role, enabled_only=True)
        return models[0] if models else None

    def update_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update the status of a model."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE models
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE model_id = ?
                """,
                (status.value, model_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_availability(
        self,
        model_ids: Iterable[str],
        availability: ModelAvailability,
        checked_at: Optional[datetime] = None,
    ) -> None:
        """Update provider availability for a set of model IDs."""
        checked_ts = checked_at or datetime.utcnow()
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                UPDATE models
                SET availability = ?, availability_checked_at = ?
                WHERE model_id = ?
                """,
                [(availability.value, checked_ts, model_id) for model_id in model_ids],
            )
            conn.commit()

    def delete_model(self, model_id: str) -> bool:
        """Delete a registered model."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_record(self, row: Any) -> ModelRecord:
        """Convert database row to ModelRecord."""
        row_data = row if isinstance(row, dict) else dict(row)

        def _get_value(source: Any, key: str, default: Any = None) -> Any:
            if source is None:
                return default
            getter = getattr(source, "get", None)
            if callable(getter):
                return getter(key, default)
            try:
                return source[key]
            except Exception:
                return default
        # Mapping for legacy role values to new ModelRole enums
        ROLE_MAPPING = {
            "scan": ModelRole.DEEP_SCAN,  # Old 'scan' maps to deep_scan
            "triage": ModelRole.TRIAGE,
            "deep_scan": ModelRole.DEEP_SCAN,
            "judge": ModelRole.JUDGE,
            "explain": ModelRole.EXPLAIN,
            "custom": ModelRole.CUSTOM,
        }

        # Parse roles from JSON
        roles_json = _get_value(row_data, "roles_json") or _get_value(row_data, "role")
        if isinstance(roles_json, str):
            try:
                roles_list = json.loads(roles_json) if roles_json.startswith("[") else [roles_json]
            except json.JSONDecodeError:
                roles_list = [roles_json] if roles_json else []
        else:
            roles_list = []

        # Convert role strings to ModelRole enums with legacy mapping
        roles = []
        for r in roles_list:
            if not r:
                continue
            # Try direct enum conversion first
            try:
                roles.append(ModelRole(r))
            except ValueError:
                # Fall back to mapping for legacy values
                mapped_role = ROLE_MAPPING.get(r.lower())
                if mapped_role:
                    roles.append(mapped_role)
                else:
                    logger.warning(f"Unknown role '{r}' for model {_get_value(row_data, 'model_id')}, skipping")

        # Parse settings from config_json
        config_json = _get_value(row_data, "config_json")
        settings = json.loads(config_json) if config_json else {}

        provider_config = {}
        provider_cfg_json = _get_value(row_data, "provider_config_json")
        if provider_cfg_json:
            try:
                provider_config = json.loads(provider_cfg_json)
            except Exception:
                provider_config = {}
        provider_base_url = _get_value(row_data, "provider_base_url")
        if provider_base_url:
            provider_config.setdefault("base_url", provider_base_url)

        return ModelRecord(
            id=_get_value(row_data, "id"),
            model_id=_get_value(row_data, "model_id"),
            model_type=ModelType(_get_value(row_data, "model_type", "ollama_local")),
            provider_id=_get_value(row_data, "provider_name", "unknown"),
            provider_config=provider_config,
            model_name=_get_value(row_data, "model_name"),
            display_name=_get_value(row_data, "display_name"),
            roles=roles,
            parser_id=_get_value(row_data, "parser_id"),
            settings=settings,
            status=ModelStatus(_get_value(row_data, "status", "registered")),
            availability=ModelAvailability(_get_value(row_data, "availability", "unknown")),
            last_checked=_get_value(row_data, "availability_checked_at"),
            created_at=_get_value(row_data, "created_at"),
            updated_at=_get_value(row_data, "updated_at"),
        )

    @staticmethod
    def _infer_provider_type(model_type: ModelType) -> str:
        """Map ModelType to provider.type for auto-created providers."""
        if model_type == ModelType.OLLAMA_LOCAL:
            return "llm"
        if model_type == ModelType.HF_LOCAL:
            return "llm"
        if model_type == ModelType.OPENAI_COMPATIBLE:
            return "llm"
        return "classic"
