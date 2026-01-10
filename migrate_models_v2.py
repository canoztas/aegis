#!/usr/bin/env python3
"""Migration script for Model Registry V2."""

import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def apply_migration(db_path: str):
    """Apply Model Registry V2 migration."""
    print(f"Applying migration to: {db_path}")

    migration_sql = Path("aegis/database/migrations/003_model_registry_v2.sql").read_text()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Execute migration
        cursor.executescript(migration_sql)
        conn.commit()
        print("[OK] Migration 003_model_registry_v2 applied successfully")

        # Verify columns exist
        cursor.execute("PRAGMA table_info(models)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {"roles_json", "parser_id", "model_type", "status"}
        missing = required_columns - columns

        if missing:
            print(f"[ERROR] Missing columns after migration: {missing}")
            return False

        print(f"[OK] All required columns present: {required_columns}")

        # Create huggingface provider if not exists
        cursor.execute("SELECT id FROM providers WHERE name = 'huggingface'")
        if not cursor.fetchone():
            cursor.execute(
                """
                INSERT INTO providers (name, type, base_url, enabled)
                VALUES ('huggingface', 'llm', 'local', 1)
                """
            )
            conn.commit()
            print("[OK] Created huggingface provider")

        return True

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Migration failed: {e}")
        return False

    finally:
        conn.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Aegis Model Registry V2 Migration")
    print("=" * 60)

    db_path = "data/aegis.db"

    if not Path(db_path).exists():
        print(f"\n[ERROR] Database not found at: {db_path}")
        return 1

    print(f"\nDatabase: {db_path}\n")

    if apply_migration(db_path):
        print("\n[SUCCESS] Migration completed successfully!")
        return 0
    else:
        print("\n[FAILED] Migration failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
