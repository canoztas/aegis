"""
Database Cleanup Script - Remove All Models

This script removes ALL registered models from the database to provide a clean slate.
Use this when you want to start fresh with model registration.

WARNING: This will delete ALL models. Make sure you want to do this before running!

Usage:
    python clean_models.py --confirm    # Delete all models
    python clean_models.py              # Show models only (no deletion)
"""

import sqlite3
import sys
from pathlib import Path


def clean_models_database(confirm=False):
    """Remove all models from the database."""
    # Find database path
    db_path = Path("data/aegis.db")

    if not db_path.exists():
        print(f"[X] Database not found at: {db_path.absolute()}")
        print("    Creating fresh database...")
        # Database will be created when app starts
        return

    print("=" * 60)
    print("Aegis Model Cleanup Script")
    print("=" * 60)
    print()
    print(f"Database: {db_path.absolute()}")
    print()

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Count existing models
        cursor.execute("SELECT COUNT(*) FROM models")
        count = cursor.fetchone()[0]

        if count == 0:
            print("[OK] No models found in database (already clean)")
            print()
            return

        print(f"Found {count} model(s) in database")
        print()

        # Show models before deletion
        cursor.execute("SELECT model_id, display_name, model_type, status FROM models")
        models = cursor.fetchall()

        print("Models in database:")
        for model_id, display_name, model_type, status in models:
            print(f"  - {model_id:<30} | {display_name:<40} | {model_type:<15} | {status}")
        print()

        if not confirm:
            print("[INFO] Run with --confirm flag to delete all models")
            print("       Example: python clean_models.py --confirm")
            return

        print("Deleting models...")

        # Delete all models
        cursor.execute("DELETE FROM models")
        deleted_count = cursor.rowcount

        # Commit changes
        conn.commit()

        print(f"[OK] Deleted {deleted_count} model(s) successfully")
        print()
        print("Database is now clean. You can register new models via:")
        print("  1. Web UI: http://localhost:5000/models")
        print("  2. API: POST /api/models/register")
        print("  3. Discovery: GET /api/models/discovered/ollama")
        print()

    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            print("[OK] Models table doesn't exist yet (database is clean)")
            print("     Table will be created when app starts")
        else:
            print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    confirm = "--confirm" in sys.argv
    clean_models_database(confirm=confirm)
