#!/usr/bin/env python3
"""Migration script from in-memory to SQLite-based Aegis V2."""
import sys
import os
from pathlib import Path

# Add parent directory to path to import aegis modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from aegis.database import get_db, init_db
from aegis.database.repositories import ProviderRepository, ModelRepository
from aegis.config_loader import ConfigLoader

def migrate():
    """Initialize database and create default configuration."""
    print("=" * 60)
    print("Aegis V2 Migration Script")
    print("=" * 60)
    print()

    # Initialize database
    print("[1/4] Initializing database...")
    db = init_db()
    print(f"      [OK] Database initialized at: {db.db_path}")
    print()

    # Create default providers
    print("[2/4] Creating default providers...")
    provider_repo = ProviderRepository()

    providers_config = [
        {
            "name": "ollama",
            "type": "llm",
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "rate_limit_per_second": 10.0,
            "timeout_seconds": 600,
            "retry_max_attempts": 3,
            "retry_backoff_factor": 2.0,
        },
        {
            "name": "openai",
            "type": "llm",
            "base_url": "https://api.openai.com/v1",
            "rate_limit_per_second": 5.0,
            "timeout_seconds": 60,
            "retry_max_attempts": 3,
            "retry_backoff_factor": 2.0,
        },
        {
            "name": "anthropic",
            "type": "llm",
            "base_url": "https://api.anthropic.com/v1",
            "rate_limit_per_second": 5.0,
            "timeout_seconds": 60,
            "retry_max_attempts": 3,
            "retry_backoff_factor": 2.0,
        },
    ]

    created_providers = {}
    for prov_config in providers_config:
        try:
            # Check if provider already exists
            existing = provider_repo.get_by_name(prov_config["name"])
            if existing:
                print(f"      [SKIP] Provider '{prov_config['name']}' already exists (ID: {existing['id']})")
                created_providers[prov_config["name"]] = existing['id']
            else:
                provider_id = provider_repo.create(
                    name=prov_config["name"],
                    type=prov_config["type"],
                    config=prov_config
                )
                created_providers[prov_config["name"]] = provider_id
                print(f"      [OK] Created provider '{prov_config['name']}' (ID: {provider_id})")
        except Exception as e:
            print(f"      [ERR] Failed to create provider '{prov_config['name']}': {e}")

    print()

    # Create default models
    print("[3/4] Creating default models...")
    model_repo = ModelRepository()

    models_config = [
        {
            "provider_name": "ollama",
            "model_id": "ollama:qwen2.5-coder:7b",
            "display_name": "Qwen 2.5 Coder 7B",
            "model_name": "qwen2.5-coder:7b",
            "role": "scan",
            "config": {"weight": 1.0, "supports_streaming": False, "supports_json": True},
        },
        {
            "provider_name": "ollama",
            "model_id": "ollama:codellama:7b",
            "display_name": "CodeLlama 7B",
            "model_name": "codellama:7b",
            "role": "scan",
            "config": {"weight": 1.0, "supports_streaming": False, "supports_json": True},
        },
    ]

    for model_config in models_config:
        try:
            provider_name = model_config["provider_name"]
            if provider_name not in created_providers:
                print(f"      [ERR] Provider '{provider_name}' not found, skipping model '{model_config['model_id']}'")
                continue

            # Check if model already exists
            existing = model_repo.get_by_model_id(model_config["model_id"])
            if existing:
                print(f"      [SKIP] Model '{model_config['model_id']}' already exists")
            else:
                model_repo.create(
                    provider_id=created_providers[provider_name],
                    model_id=model_config["model_id"],
                    display_name=model_config["display_name"],
                    model_name=model_config["model_name"],
                    role=model_config["role"],
                    config=model_config["config"]
                )
                print(f"      [OK] Created model '{model_config['model_id']}'")
        except Exception as e:
            print(f"      [ERR] Failed to create model '{model_config['model_id']}': {e}")

    print()

    # Load from YAML config (optional)
    print("[4/5] Loading models from YAML config (if available)...")
    try:
        project_root = Path(__file__).parent.parent
        yaml_path = project_root / "config" / "models_v2.yaml"

        if yaml_path.exists():
            print(f"      [INFO] Found config at: {yaml_path}")
            ConfigLoader.bootstrap_from_yaml(yaml_path)
            print("      [OK] YAML config loaded successfully")
        else:
            print(f"      [SKIP] No YAML config found at: {yaml_path}")
            print("      [INFO] Using default providers and models from migration script")
    except Exception as e:
        print(f"      [WARN] Failed to load YAML config: {e}")
        print("      [INFO] Continuing with default configuration")

    print()

    # Summary
    print("[5/5] Migration complete!")
    print()
    print("Next steps:")
    print("  1. Review the database at: {}".format(db.db_path))
    print("  2. (Optional) Edit config/models_v2.yaml to add/modify models")
    print("  3. (Optional) Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY")
    print("  4. Set environment variable: AEGIS_USE_V2=true")
    print("  5. Start Aegis: python app.py")
    print()
    print("=" * 60)

if __name__ == "__main__":
    try:
        migrate()
    except KeyboardInterrupt:
        print("\n\nMigration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
