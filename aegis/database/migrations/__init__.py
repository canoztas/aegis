"""Database migrations for Aegis.

This module provides a simple migration system for SQLite schema changes.
Migrations are applied in order based on version number.

Usage:
    from aegis.database.migrations import run_migrations
    run_migrations()
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Callable, Tuple

logger = logging.getLogger(__name__)

# Migration registry: List of (version, description, migration_func)
_MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = []


def migration(version: int, description: str):
    """Decorator to register a migration function."""
    def decorator(func: Callable[[sqlite3.Connection], None]):
        _MIGRATIONS.append((version, description, func))
        return func
    return decorator


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from database."""
    try:
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        # Table doesn't exist, schema version is 0
        return 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set schema version in database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def run_migrations(db_path: Path = None) -> int:
    """
    Run pending migrations on the database.

    Args:
        db_path: Path to database. If None, uses default location.

    Returns:
        Number of migrations applied
    """
    if db_path is None:
        from aegis.database import get_db
        db = get_db()
        db_path = db.db_path

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        current_version = get_schema_version(conn)
        logger.info(f"Current schema version: {current_version}")

        # Sort migrations by version
        pending = sorted(
            [(v, d, f) for v, d, f in _MIGRATIONS if v > current_version],
            key=lambda x: x[0]
        )

        if not pending:
            logger.info("No pending migrations")
            return 0

        applied = 0
        for version, description, migrate_func in pending:
            logger.info(f"Applying migration {version}: {description}")
            try:
                migrate_func(conn)
                set_schema_version(conn, version)
                conn.commit()
                applied += 1
                logger.info(f"Migration {version} applied successfully")
            except Exception as e:
                conn.rollback()
                logger.error(f"Migration {version} failed: {e}")
                raise

        return applied

    finally:
        conn.close()


# Import migrations to register them
from aegis.database.migrations import v001_add_roles_json
