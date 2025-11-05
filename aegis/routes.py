import os
import zipfile
from typing import Any, Dict, List
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    flash,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from aegis.analyzer import SecurityAnalyzer
from aegis.utils import allowed_file, extract_source_files

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index() -> str:
    return render_template("index.html")


@main_bp.route("/upload", methods=["POST"])
def upload_file() -> Any:
    if "file" not in request.files:
        flash("No file selected")
        return redirect(request.url)

    file: FileStorage = request.files["file"]

    if file.filename == "":
        flash("No file selected")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            source_files = extract_source_files(filepath)
            analyzer = SecurityAnalyzer()
            results = analyzer.analyze_files(source_files)

            os.remove(filepath)

            return render_template("results.html", results=results)

        except Exception as e:
            flash(f"Error analyzing file: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for("main.index"))

    flash("Invalid file type. Please upload a ZIP file.")
    return redirect(url_for("main.index"))


@main_bp.route("/api/analyze", methods=["POST"])
def api_analyze() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file: FileStorage = request.files["file"]

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        source_files = extract_source_files(filepath)
        analyzer = SecurityAnalyzer()
        results = analyzer.analyze_files(source_files)

        os.remove(filepath)

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/health")
def health_check() -> Any:
    return "OK", 200
