"""
Web T.R.A.C.E — Flask Web Application & Inference Server.

Flat Architecture: All templates and static assets reside directly in the
project root. Only outputs/ and cited_research_papers/ exist as folders.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import model_service

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR),
    static_url_path=""
)


# ---------------------------------------------------------------------------
# Web UI Entrypoints (Flat Root Templates)
# ---------------------------------------------------------------------------
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/login")
@app.route("/login.html")
def login_page():
    return render_template("login.html")


@app.route("/signup")
@app.route("/signup.html")
def signup_page():
    return render_template("signup.html")


# ---------------------------------------------------------------------------
# Static Assets Serving & Legacy Path Compatibility
# ---------------------------------------------------------------------------
@app.route("/static/<path:filename>")
@app.route("/css/<path:filename>")
@app.route("/js/<path:filename>")
def serve_asset_fallback(filename):
    """Serves static CSS/JS files from root regardless of legacy path prefixes."""
    base_name = Path(filename).name
    file_path = BASE_DIR / base_name
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(BASE_DIR), base_name)
    return "", 404


# ---------------------------------------------------------------------------
# Static Figures Serving (outputs/figures/)
# ---------------------------------------------------------------------------
@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(str(FIGURE_DIR), filename)


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Web T.R.A.C.E Zero-Scraping Phishing Detector",
        "model_version": "2.0",
        "feature_count": len(model_service.FEATURE_COLUMNS),
        "active_models_count": len(model_service.MODEL_METADATA),
        "zero_scraping_mode": True,
        "calibration": "Isotonic Regression",
        "supported_roles": ["User", "Administrator"],
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns metadata for all benchmarked ML and DL models."""
    return jsonify(model_service.list_available_models())


@app.route("/api/scan", methods=["POST"])
def scan_url():
    """
    Performs real-time zero-scraping feature extraction and model inference
    for a single URL. Returns risk tier, probabilities, and diagnostics.
    """
    payload = request.get_json(force=True, silent=True) or {}
    url = str(payload.get("url", "")).strip()
    model_name = str(payload.get("model", "Ensemble_Calibrated_SoftVoting")).strip()
    if not model_name:
        model_name = "Ensemble_Calibrated_SoftVoting"

    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL exceeds maximum allowed length of 2048 characters"}), 400

    try:
        result = model_service.predict_single_url(url, model_name=model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@app.route("/api/batch-scan", methods=["POST"])
def batch_scan_urls():
    """Evaluates a list of URLs and returns a triage report table."""
    payload = request.get_json(force=True, silent=True) or {}
    urls = payload.get("urls", [])
    model_name = str(payload.get("model", "Ensemble_Calibrated_SoftVoting")).strip()
    if not model_name:
        model_name = "Ensemble_Calibrated_SoftVoting"

    if not urls or not isinstance(urls, list):
        return jsonify({"error": "URL list cannot be empty"}), 400

    try:
        results = model_service.predict_batch_urls(urls, model_name=model_name)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Batch scan error: {str(e)}"}), 500


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns benchmark comparison metrics across all 10 models."""
    csv_file = REPORT_DIR / "final_model_comparison.csv"
    if not csv_file.exists():
        csv_file = REPORT_DIR / "tp_tn_evaluation_unseen_domain.csv"
    if not csv_file.exists():
        return jsonify({"error": "Benchmark report not found"}), 404

    df = pd.read_csv(csv_file)
    return jsonify(df.to_dict(orient="records"))


def main():
    print("=" * 70)
    print("STARTING WEB T.R.A.C.E APPLICATION SERVER (FLAT ARCHITECTURE)")
    print("=" * 70)
    print("Local Web Dashboard: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
