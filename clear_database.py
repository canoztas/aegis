#!/usr/bin/env python3
"""Script to clear all scan data from the Aegis database."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aegis.database import Database, get_db


def clear_all_scans():
    """Clear all scans and findings from the database."""
    db = get_db()

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Count records before deletion
        cursor.execute("SELECT COUNT(*) FROM findings")
        findings_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scan_files")
        files_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM scans")
        scans_count = cursor.fetchone()[0]

        print(f"Found {scans_count} scans, {files_count} files, {findings_count} findings")

        # Confirm deletion
        if scans_count > 0:
            response = input(f"\nAre you sure you want to delete ALL {scans_count} scan(s)? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                return

        # Delete in order (respect foreign keys)
        print("\nDeleting findings...")
        cursor.execute("DELETE FROM findings")

        print("Deleting scan files...")
        cursor.execute("DELETE FROM scan_files")

        print("Deleting scans...")
        cursor.execute("DELETE FROM scans")

        conn.commit()

        print(f"\n✅ Successfully deleted:")
        print(f"   - {findings_count} findings")
        print(f"   - {files_count} scan files")
        print(f"   - {scans_count} scans")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Aegis Database Cleanup Tool")
    print("=" * 60)

    # Check if database exists
    db_path = os.environ.get('AEGIS_DB_PATH', 'data/aegis.db')
    if not os.path.exists(db_path):
        print(f"\n❌ Database not found at: {db_path}")
        print("Make sure AEGIS_DB_PATH environment variable is set correctly.")
        return 1

    print(f"\nDatabase: {db_path}")

    try:
        clear_all_scans()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
