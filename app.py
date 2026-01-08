#!/usr/bin/env python3

import os
import sys
from aegis import create_app

if __name__ == "__main__":
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Initialize database (V2)
    use_v2 = os.environ.get("AEGIS_USE_V2", "true").lower() == "true"
    if use_v2:
        print("Initializing Aegis V2 (SQLite database)...")
        from aegis.database import get_db, init_db
        from aegis.database.repositories import ProviderRepository, ModelRepository

        db = get_db()
        print(f"  Database: {db.db_path}")

        # Check if database has any providers
        provider_repo = ProviderRepository()
        providers = provider_repo.list_enabled()

        if not providers:
            print("  No providers found. Run 'python scripts/migrate_to_v2.py' first.")
            print("  Continuing with V1 (in-memory) mode...")
            use_v2 = False
        else:
            model_repo = ModelRepository()
            models = model_repo.list_all()
            print(f"  Loaded {len(providers)} providers, {len(models)} models")
    else:
        print("Using Aegis V1 (in-memory mode)")

    app = create_app()

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"

    print(f"Starting aegis on http://{host}:{port}")
    print("Upload your source code ZIP files for security analysis")
    print("Press CTRL+C to stop the server")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\nShutting down aegis...")
        sys.exit(0)
