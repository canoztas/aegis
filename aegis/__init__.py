from flask import Flask
from aegis.routes import main_bp, init_scan_worker
from aegis.api.routes_models import models_bp
from aegis.api.routes_credentials import bp as credentials_bp
from aegis.config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(main_bp)
    app.register_blueprint(models_bp)  # Model management API
    app.register_blueprint(credentials_bp)  # Credential management API
    init_scan_worker(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=7766)
