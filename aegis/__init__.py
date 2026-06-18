from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from aegis.routes import main_bp, init_scan_worker
from aegis.api.routes_models import models_bp
from aegis.api.routes_credentials import bp as credentials_bp
from aegis.config import Config


def _wants_json() -> bool:
    """Whether the current request expects a JSON response rather than HTML."""
    if request.path.startswith("/api/"):
        return True
    accept = request.accept_mimetypes
    best = accept.best_match(["application/json", "text/html"])
    return best == "application/json" and accept[best] >= accept["text/html"]


def _register_error_handlers(app: Flask) -> None:
    """Return JSON (not Flask's default HTML page) for errors on API requests.

    Without this, any unhandled exception in an API route returns an HTML error
    page, and the frontend's `response.json()` then throws
    "Unexpected token '<'", swallowing the real error message.
    """

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        if _wants_json():
            response = jsonify({
                "error": error.description or error.name,
                "status": error.code,
            })
            response.status_code = error.code or 500
            return response
        return error  # default HTML page for normal page routes

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error: Exception):
        if isinstance(error, HTTPException):
            return error  # handled above
        app.logger.exception("Unhandled exception during request")
        if _wants_json():
            return jsonify({
                "error": str(error) or "Internal Server Error",
                "status": 500,
            }), 500
        raise error  # let Flask render its default page for page routes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    _register_error_handlers(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(models_bp)  # Model management API
    app.register_blueprint(credentials_bp)  # Credential management API
    init_scan_worker(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=7766)
