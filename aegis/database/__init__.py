"""Database initialization and connection management for Aegis."""
from pathlib import Path
import sqlite3
from typing import Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager with automatic schema initialization."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. If None, uses data/aegis.db
        """
        if db_path is None:
            # Default to data/aegis.db in project root
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "aegis.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing database at {self.db_path}")
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize database schema if not exists."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            logger.warning(f"Schema file not found at {schema_path}")
            return

        try:
            with self.get_connection() as conn:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                conn.executescript(schema_sql)
                conn.commit()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection with Row factory enabled

        Example:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM models")
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()


# Global database instance (singleton pattern)
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """
    Get the global database instance (singleton).

    Returns:
        Database: The global database instance

    Example:
        from aegis.database import get_db
        db = get_db()
        with db.get_connection() as conn:
            # ... use connection
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def init_db(db_path: Optional[str] = None):
    """
    Initialize the global database instance with a custom path.

    Args:
        db_path: Path to SQLite database file

    Example:
        init_db("/custom/path/aegis.db")
    """
    global _db_instance
    _db_instance = Database(db_path)
    return _db_instance
