"""Cost tracking and token usage logging for cloud API providers."""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Token usage and cost record."""
    timestamp: datetime
    provider: str
    model_name: str
    scan_id: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    request_id: Optional[str] = None


class CostTracker:
    """Track token usage and costs for API providers."""

    def __init__(self, db_path: str = "data/aegis.db"):
        """
        Initialize cost tracker.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()

        # In-memory cache for current session
        self.session_usage: Dict[str, float] = {}
        self.session_tokens: Dict[str, int] = {}

    def _init_database(self):
        """Initialize cost tracking table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    scan_id TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    request_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_scan_id
                ON api_usage(scan_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_provider
                ON api_usage(provider, model_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp
                ON api_usage(timestamp)
            """)

    def log_usage(
        self,
        provider: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        scan_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """
        Log API usage and cost.

        Args:
            provider: Provider name (openai, anthropic, google)
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            scan_id: Associated scan ID
            request_id: API request ID
        """
        timestamp = datetime.utcnow().isoformat()
        total_tokens = input_tokens + output_tokens

        # Update session cache
        provider_key = f"{provider}:{model_name}"
        with self.lock:
            self.session_usage[provider_key] = self.session_usage.get(provider_key, 0.0) + cost_usd
            self.session_tokens[provider_key] = self.session_tokens.get(provider_key, 0) + total_tokens

        # Insert into database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO api_usage (
                        timestamp, provider, model_name, scan_id,
                        input_tokens, output_tokens, total_tokens, cost_usd, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        provider,
                        model_name,
                        scan_id,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost_usd,
                        request_id,
                    ),
                )

            logger.info(
                f"API usage logged: {provider}/{model_name} - "
                f"tokens={total_tokens} (in={input_tokens}, out={output_tokens}), "
                f"cost=${cost_usd:.6f}, scan={scan_id}"
            )

        except Exception as e:
            logger.error(f"Failed to log API usage: {e}")

    def get_scan_cost(self, scan_id: str) -> Dict[str, any]:
        """
        Get total cost and token usage for a scan.

        Args:
            scan_id: Scan ID

        Returns:
            Dictionary with cost and token breakdown
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    provider,
                    model_name,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as total_cost,
                    COUNT(*) as request_count
                FROM api_usage
                WHERE scan_id = ?
                GROUP BY provider, model_name
                """,
                (scan_id,),
            )

            models = []
            total_cost = 0.0
            total_tokens = 0

            for row in cursor:
                model_data = {
                    "provider": row["provider"],
                    "model_name": row["model_name"],
                    "input_tokens": row["total_input"],
                    "output_tokens": row["total_output"],
                    "total_tokens": row["total_tokens"],
                    "cost_usd": row["total_cost"],
                    "request_count": row["request_count"],
                }
                models.append(model_data)
                total_cost += row["total_cost"]
                total_tokens += row["total_tokens"]

            return {
                "scan_id": scan_id,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "models": models,
            }

    def get_provider_usage(
        self,
        provider: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Get usage statistics by provider.

        Args:
            provider: Filter by provider (None = all providers)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Dictionary with usage statistics
        """
        query = """
            SELECT
                provider,
                model_name,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                COUNT(*) as request_count
            FROM api_usage
            WHERE 1=1
        """
        params = []

        if provider:
            query += " AND provider = ?"
            params.append(provider)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " GROUP BY provider, model_name ORDER BY total_cost DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            models = []
            total_cost = 0.0
            total_tokens = 0
            total_requests = 0

            for row in cursor:
                model_data = {
                    "provider": row["provider"],
                    "model_name": row["model_name"],
                    "input_tokens": row["total_input"],
                    "output_tokens": row["total_output"],
                    "total_tokens": row["total_tokens"],
                    "cost_usd": row["total_cost"],
                    "request_count": row["request_count"],
                }
                models.append(model_data)
                total_cost += row["total_cost"]
                total_tokens += row["total_tokens"]
                total_requests += row["request_count"]

            return {
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "models": models,
            }

    def get_session_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Get current session usage (in-memory cache).

        Returns:
            Dictionary mapping provider:model to usage stats
        """
        with self.lock:
            return {
                provider_key: {
                    "cost_usd": self.session_usage.get(provider_key, 0.0),
                    "total_tokens": self.session_tokens.get(provider_key, 0),
                }
                for provider_key in set(list(self.session_usage.keys()) + list(self.session_tokens.keys()))
            }

    def clear_session_usage(self):
        """Clear session usage cache."""
        with self.lock:
            self.session_usage.clear()
            self.session_tokens.clear()

    def get_budget_status(self, budget_usd: float, start_date: Optional[str] = None) -> Dict[str, any]:
        """
        Check budget status.

        Args:
            budget_usd: Budget limit in USD
            start_date: Budget start date (ISO format, None = all time)

        Returns:
            Dictionary with budget status
        """
        usage = self.get_provider_usage(start_date=start_date)
        total_cost = usage["total_cost_usd"]

        return {
            "budget_usd": budget_usd,
            "spent_usd": total_cost,
            "remaining_usd": budget_usd - total_cost,
            "percent_used": (total_cost / budget_usd * 100) if budget_usd > 0 else 0.0,
            "over_budget": total_cost > budget_usd,
        }


# Global cost tracker instance
DEFAULT_COST_TRACKER = CostTracker()
