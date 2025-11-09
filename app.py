#!/usr/bin/env python3

import os
import sys
from aegis import create_app

if __name__ == "__main__":
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

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
