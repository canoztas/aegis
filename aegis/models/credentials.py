"""Secure credential management for API providers."""

import logging
import os
import sqlite3
import threading
from base64 import b64decode, b64encode
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CredentialManager:
    """Manage API credentials with optional encryption."""

    def __init__(self, db_path: str = "data/aegis.db", use_encryption: bool = False):
        """
        Initialize credential manager.

        Args:
            db_path: Path to SQLite database
            use_encryption: Enable encryption (requires cryptography package)
        """
        self.db_path = db_path
        self.use_encryption = use_encryption
        self.lock = threading.Lock()

        if use_encryption:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(self._get_or_create_key())
            except ImportError:
                logger.warning("cryptography package not installed, falling back to base64 encoding")
                self.use_encryption = False
                self.cipher = None
        else:
            self.cipher = None

        self._init_database()

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_path = "data/.aegis_key"
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            os.makedirs("data", exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Restrict to owner only
            logger.info(f"Created new encryption key: {key_path}")
            return key

    def _init_database(self):
        """Initialize credentials table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    encrypted BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(provider, key_name)
                )
            """)

            # Create index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_credentials_provider
                ON api_credentials(provider, key_name)
            """)

    def _encrypt(self, value: str) -> str:
        """Encrypt value."""
        if self.use_encryption and self.cipher:
            return self.cipher.encrypt(value.encode()).decode()
        else:
            # Fallback to base64 encoding (NOT SECURE, just obfuscation)
            return b64encode(value.encode()).decode()

    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt value."""
        if self.use_encryption and self.cipher:
            return self.cipher.decrypt(encrypted_value.encode()).decode()
        else:
            # Fallback to base64 decoding
            return b64decode(encrypted_value.encode()).decode()

    def store_credential(
        self,
        provider: str,
        key_name: str,
        key_value: str,
        encrypt: bool = True,
    ):
        """
        Store API credential.

        Args:
            provider: Provider name (openai, anthropic, google)
            key_name: Credential key name (api_key, organization, etc.)
            key_value: Credential value
            encrypt: Whether to encrypt (default: True)
        """
        with self.lock:
            encrypted_value = self._encrypt(key_value) if encrypt else key_value
            timestamp = datetime.utcnow().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO api_credentials (provider, key_name, key_value, encrypted, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(provider, key_name)
                    DO UPDATE SET
                        key_value = excluded.key_value,
                        encrypted = excluded.encrypted,
                        updated_at = excluded.updated_at
                    """,
                    (provider, key_name, encrypted_value, encrypt, timestamp),
                )

            logger.info(f"Stored credential: {provider}/{key_name} (encrypted={encrypt})")

    def get_credential(
        self,
        provider: str,
        key_name: str,
        fallback_env_var: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get API credential.

        Args:
            provider: Provider name
            key_name: Credential key name
            fallback_env_var: Environment variable to check if not in DB

        Returns:
            Decrypted credential value or None
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT key_value, encrypted FROM api_credentials WHERE provider = ? AND key_name = ?",
                    (provider, key_name),
                )
                row = cursor.fetchone()

                if row:
                    value = row["key_value"]
                    encrypted = bool(row["encrypted"])
                    return self._decrypt(value) if encrypted else value

            # Fallback to environment variable
            if fallback_env_var:
                env_value = os.getenv(fallback_env_var)
                if env_value:
                    logger.debug(f"Using credential from env var: {fallback_env_var}")
                    return env_value

            return None

    def delete_credential(self, provider: str, key_name: str):
        """
        Delete API credential.

        Args:
            provider: Provider name
            key_name: Credential key name
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM api_credentials WHERE provider = ? AND key_name = ?",
                    (provider, key_name),
                )
            logger.info(f"Deleted credential: {provider}/{key_name}")

    def list_credentials(self, provider: Optional[str] = None) -> list:
        """
        List stored credentials (without values).

        Args:
            provider: Filter by provider (None = all)

        Returns:
            List of credential metadata
        """
        query = "SELECT provider, key_name, created_at, updated_at FROM api_credentials"
        params = []

        if provider:
            query += " WHERE provider = ?"
            params.append(provider)

        query += " ORDER BY provider, key_name"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def validate_credential(self, provider: str, api_key: str) -> Dict[str, any]:
        """
        Validate API credential by making a test request.

        Args:
            provider: Provider name (openai, anthropic, google)
            api_key: API key to validate

        Returns:
            Dictionary with validation result
        """
        try:
            if provider == "openai":
                import openai
                client = openai.OpenAI(api_key=api_key, timeout=10)
                # Test with models.list() - lightweight API call
                models = client.models.list()
                return {
                    "valid": True,
                    "provider": provider,
                    "message": f"Valid OpenAI API key (found {len(list(models.data))} models)",
                }

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key, timeout=10)
                # Test with a minimal message
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}],
                )
                return {
                    "valid": True,
                    "provider": provider,
                    "message": "Valid Anthropic API key",
                }

            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Test with list_models()
                models = list(genai.list_models())
                return {
                    "valid": True,
                    "provider": provider,
                    "message": f"Valid Google API key (found {len(models)} models)",
                }

            else:
                return {
                    "valid": False,
                    "provider": provider,
                    "message": f"Unknown provider: {provider}",
                }

        except Exception as e:
            logger.error(f"Credential validation failed for {provider}: {e}")
            return {
                "valid": False,
                "provider": provider,
                "message": str(e),
            }

    def rotate_credential(self, provider: str, key_name: str, new_value: str):
        """
        Rotate API credential (validate new, then update).

        Args:
            provider: Provider name
            key_name: Credential key name (usually "api_key")
            new_value: New credential value
        """
        # Validate new credential first
        if key_name == "api_key":
            result = self.validate_credential(provider, new_value)
            if not result["valid"]:
                raise ValueError(f"New credential validation failed: {result['message']}")

        # Store new credential
        self.store_credential(provider, key_name, new_value, encrypt=True)
        logger.info(f"Rotated credential: {provider}/{key_name}")


# Global credential manager instance
DEFAULT_CREDENTIAL_MANAGER = CredentialManager()
