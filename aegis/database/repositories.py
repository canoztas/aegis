"""Repository pattern for database access."""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

from aegis.database import get_db
from aegis.data_models import Finding

logger = logging.getLogger(__name__)


class ProviderRepository:
    """Repository for provider CRUD operations."""

    def create(self, name: str, type: str, config: Dict[str, Any]) -> int:
        """
        Create a new provider.

        Args:
            name: Provider name (ollama, openai, etc.)
            type: Provider type (llm, classic, rest)
            config: Provider configuration dict

        Returns:
            int: Provider ID
        """
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO providers (name, type, base_url, config_json,
                                       rate_limit_per_second, timeout_seconds,
                                       retry_max_attempts, retry_backoff_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, type, config.get('base_url'),
                json.dumps(config),
                config.get('rate_limit_per_second', 10.0),
                config.get('timeout_seconds', 300),
                config.get('retry_max_attempts', 3),
                config.get('retry_backoff_factor', 2.0)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_by_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get provider by ID."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM providers WHERE id = ?", (provider_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                return result
            return None

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider by name."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM providers WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                return result
            return None

    def list_enabled(self) -> List[Dict[str, Any]]:
        """List all enabled providers."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM providers WHERE enabled = 1")
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                results.append(result)
            return results

    def update(self, provider_id: int, **kwargs):
        """Update provider fields."""
        db = get_db()
        with db.get_connection() as conn:
            set_clauses = []
            values = []
            for key, value in kwargs.items():
                if key == 'config':
                    set_clauses.append("config_json = ?")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            set_clauses.append("updated_at = ?")
            values.append(datetime.now())
            values.append(provider_id)

            query = f"UPDATE providers SET {', '.join(set_clauses)} WHERE id = ?"
            conn.execute(query, values)
            conn.commit()


class ModelRepository:
    """Repository for model CRUD operations."""

    def create(self, provider_id: int, model_id: str, display_name: str,
               model_name: str, role: str, config: Dict[str, Any]) -> int:
        """
        Create a new model.

        Args:
            provider_id: Foreign key to providers table
            model_id: Unique model identifier (e.g., 'ollama:qwen2.5-coder:7b')
            display_name: Human-readable name
            model_name: Actual model name for API calls
            role: Model role (triage, deep_scan, judge, explain, scan)
            config: Model-specific configuration

        Returns:
            int: Model database ID
        """
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO models (provider_id, model_id, display_name, model_name,
                                    role, config_json, weight, supports_streaming,
                                    supports_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider_id, model_id, display_name, model_name, role,
                json.dumps(config),
                config.get('weight', 1.0),
                config.get('supports_streaming', False),
                config.get('supports_json', True)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_by_model_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model with provider info by model_id."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.*, p.name as provider_name, p.type as provider_type,
                       p.base_url, p.config_json as provider_config
                FROM models m
                JOIN providers p ON m.provider_id = p.id
                WHERE m.model_id = ? AND m.enabled = 1
            """, (model_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                if result.get('provider_config'):
                    result['provider_config'] = json.loads(result['provider_config'])
                return result
            return None

    def list_by_role(self, role: str) -> List[Dict[str, Any]]:
        """List all models with a specific role."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.*, p.name as provider_name
                FROM models m
                JOIN providers p ON m.provider_id = p.id
                WHERE m.role = ? AND m.enabled = 1
            """, (role,))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                results.append(result)
            return results

    def list_all(self) -> List[Dict[str, Any]]:
        """List all enabled models."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.*, p.name as provider_name
                FROM models m
                JOIN providers p ON m.provider_id = p.id
                WHERE m.enabled = 1
            """)
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('config_json'):
                    result['config'] = json.loads(result['config_json'])
                results.append(result)
            return results

    def update(self, model_id: str, **kwargs):
        """Update model fields."""
        db = get_db()
        with db.get_connection() as conn:
            set_clauses = []
            values = []
            for key, value in kwargs.items():
                if key == 'config':
                    set_clauses.append("config_json = ?")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            set_clauses.append("updated_at = ?")
            values.append(datetime.now())
            values.append(model_id)

            query = f"UPDATE models SET {', '.join(set_clauses)} WHERE model_id = ?"
            conn.execute(query, values)
            conn.commit()


class ScanRepository:
    """Repository for scan CRUD operations."""

    def create(self, scan_id: str, pipeline_config: Dict[str, Any],
               consensus_strategy: str, upload_filename: str = None) -> int:
        """Create a new scan record."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scans (scan_id, status, upload_filename, pipeline_config_json,
                                   consensus_strategy, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                scan_id, 'pending', upload_filename, json.dumps(pipeline_config),
                consensus_strategy, datetime.now()
            ))
            conn.commit()
            return cursor.lastrowid

    def update_status(self, scan_id: str, status: str, error: Optional[str] = None):
        """Update scan status."""
        db = get_db()
        with db.get_connection() as conn:
            updates = ["status = ?", "updated_at = ?"]
            values = [status, datetime.now()]

            if status in ('completed', 'failed', 'cancelled'):
                updates.append("completed_at = ?")
                values.append(datetime.now())

            if error:
                updates.append("error_message = ?")
                values.append(error)

            values.append(scan_id)

            conn.execute(f"""
                UPDATE scans SET {', '.join(updates)}
                WHERE scan_id = ?
            """, values)
            conn.commit()

    def update_progress(self, scan_id: str, processed_files: int = None,
                        processed_chunks: int = None, total_files: int = None,
                        total_chunks: int = None):
        """Update scan progress."""
        db = get_db()
        with db.get_connection() as conn:
            updates = ["updated_at = ?"]
            values = [datetime.now()]

            if processed_files is not None:
                updates.append("processed_files = ?")
                values.append(processed_files)
            if processed_chunks is not None:
                updates.append("processed_chunks = ?")
                values.append(processed_chunks)
            if total_files is not None:
                updates.append("total_files = ?")
                values.append(total_files)
            if total_chunks is not None:
                updates.append("total_chunks = ?")
                values.append(total_chunks)

            values.append(scan_id)

            conn.execute(f"""
                UPDATE scans SET {', '.join(updates)}
                WHERE scan_id = ?
            """, values)
            conn.commit()

    def get_by_scan_id(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan by scan_id."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('pipeline_config_json'):
                    result['pipeline_config'] = json.loads(result['pipeline_config_json'])
                return result
            return None

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent scans."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scans
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('pipeline_config_json'):
                    result['pipeline_config'] = json.loads(result['pipeline_config_json'])
                results.append(result)
            return results

    def list_by_statuses(self, statuses: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """List scans matching any of the provided statuses."""
        if not statuses:
            return []
        placeholders = ",".join(["?"] * len(statuses))
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM scans
                WHERE status IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
            """, (*statuses, limit))
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('pipeline_config_json'):
                    result['pipeline_config'] = json.loads(result['pipeline_config_json'])
                results.append(result)
            return results

    def add_file(self, scan_id: str, file_path: str, content: str, language: str):
        """Add source file to scan."""
        db = get_db()
        with db.get_connection() as conn:
            conn.execute("""
                INSERT INTO scan_files (scan_id, file_path, content, language, lines_count)
                VALUES (?, ?, ?, ?, ?)
            """, (scan_id, file_path, content, language, len(content.split('\n'))))
            conn.commit()

    def get_file(self, scan_id: str, file_path: str) -> Optional[str]:
        """Get source file content."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content FROM scan_files
                WHERE scan_id = ? AND file_path = ?
            """, (scan_id, file_path))
            row = cursor.fetchone()
            return row['content'] if row else None

    def list_files(self, scan_id: str) -> List[Dict[str, Any]]:
        """List all files in a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT file_path, language, lines_count
                FROM scan_files
                WHERE scan_id = ?
            """, (scan_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_file(self, scan_id: str, file_path: str):
        """Delete a file from a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM scan_files
                WHERE scan_id = ? AND file_path = ?
            """, (scan_id, file_path))
            conn.commit()

    def delete_scan(self, scan_id: str):
        """Delete a scan and all associated files."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Delete scan files first
            cursor.execute("DELETE FROM scan_files WHERE scan_id = ?", (scan_id,))
            # Delete scan record
            cursor.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,))
            conn.commit()


class FindingRepository:
    """Repository for finding CRUD operations."""

    def create_batch(self, findings: List[Finding], scan_id: str,
                     model_id: Optional[str] = None, is_consensus: bool = False):
        """Create multiple findings in batch."""
        if not findings:
            return

        db = get_db()
        with db.get_connection() as conn:
            conn.executemany("""
                INSERT INTO findings (scan_id, model_id, is_consensus, fingerprint,
                                      name, severity, cwe, file, start_line, end_line,
                                      message, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (scan_id, model_id, is_consensus, f.fingerprint, f.name, f.severity,
                 f.cwe, f.file, f.start_line, f.end_line, f.message, f.confidence)
                for f in findings
            ])
            conn.commit()

    def get_consensus_findings(self, scan_id: str) -> List[Finding]:
        """Get consensus findings for a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM findings
                WHERE scan_id = ? AND is_consensus = 1
                ORDER BY severity DESC, file, start_line
            """, (scan_id,))
            return [Finding.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_by_model(self, scan_id: str, model_id: str) -> List[Finding]:
        """Get findings from a specific model."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM findings
                WHERE scan_id = ? AND model_id = ? AND is_consensus = 0
                ORDER BY file, start_line
            """, (scan_id, model_id))
            return [Finding.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_all_findings(self, scan_id: str, include_consensus: bool = True) -> List[Dict[str, Any]]:
        """Get all findings for a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if include_consensus:
                cursor.execute("""
                    SELECT * FROM findings
                    WHERE scan_id = ?
                    ORDER BY is_consensus DESC, severity DESC, file, start_line
                """, (scan_id,))
            else:
                cursor.execute("""
                    SELECT * FROM findings
                    WHERE scan_id = ? AND is_consensus = 0
                    ORDER BY severity DESC, file, start_line
                """, (scan_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_by_scan_id(self, scan_id: str):
        """Delete all findings for a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM findings WHERE scan_id = ?", (scan_id,))
            conn.commit()


class TelemetryRepository:
    """Repository for model execution telemetry."""

    def record_execution(self, scan_id: str, model_id: str, file_path: str,
                         chunk_index: int, status: str, latency_ms: int,
                         retry_count: int = 0, usage: Optional[Dict] = None,
                         cost_usd: Optional[float] = None,
                         error: Optional[str] = None):
        """Record a model execution."""
        db = get_db()
        with db.get_connection() as conn:
            conn.execute("""
                INSERT INTO model_executions
                (scan_id, model_id, file_path, chunk_index, status, latency_ms,
                 retry_count, token_usage_json, cost_usd, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_id, model_id, file_path, chunk_index, status, latency_ms,
                retry_count, json.dumps(usage) if usage else None, cost_usd, error
            ))
            conn.commit()

    def get_model_stats(self, scan_id: str, model_id: str) -> Dict[str, Any]:
        """Get aggregated stats for a model in a scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_calls,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count
                FROM model_executions
                WHERE scan_id = ? AND model_id = ?
            """, (scan_id, model_id))
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_scan_stats(self, scan_id: str) -> List[Dict[str, Any]]:
        """Get per-model stats for entire scan."""
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    model_id,
                    COUNT(*) as total_calls,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
                FROM model_executions
                WHERE scan_id = ?
                GROUP BY model_id
            """, (scan_id,))
            return [dict(row) for row in cursor.fetchall()]


