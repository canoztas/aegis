"""Migration 001: Add roles_json column to models table.

This migration adds the roles_json column for multi-role support.
The deprecated 'role' column is kept for backward compatibility.
"""

import sqlite3
from aegis.database.migrations import migration


@migration(version=1, description="Add roles_json column to models table")
def migrate(conn: sqlite3.Connection) -> None:
    """Add roles_json column if it doesn't exist."""
    cursor = conn.execute("PRAGMA table_info(models)")
    columns = [row[1] for row in cursor.fetchall()]

    if "roles_json" not in columns:
        conn.execute("""
            ALTER TABLE models
            ADD COLUMN roles_json TEXT DEFAULT '[]'
        """)

        # Migrate existing role values to roles_json
        conn.execute("""
            UPDATE models
            SET roles_json = json_array(role)
            WHERE role IS NOT NULL AND role != ''
        """)
