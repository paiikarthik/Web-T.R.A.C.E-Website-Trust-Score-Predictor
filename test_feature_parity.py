"""
Web T.R.A.C.E — Feature Parity & Safety Integrity Verification Suite.

Tests:
1. Feature column parity: Ensures app (model_service) and training (feature_list.json)
   consume the exact same feature list and order.
2. Shortcut & leak ban: Asserts that 'www.', 'https', and path/query features
   are strictly ABSENT from the general production model.
3. Free-host flagging & score capping: Asserts that URLs on free-hosting platforms
   (e.g., github.io, web.app, firebaseapp.com) trigger the required disclaimer
   and trust score cap.
4. End-to-end inference verification: Tests legitimate and phishing test cases.
5. Flask API health & scan endpoint contracts.
"""

import json
from pathlib import Path
import pytest
import features
import model_service
from app import app

BASE_DIR = Path(__file__).resolve().parent
FEATURE_LIST_PATH = BASE_DIR / "outputs" / "models" / "feature_list.json"

FORBIDDEN_SHORTCUTS = [
    "IsHTTPS",
    "is_https",
    "https",
    "www",
    "starts_with_www",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "URLLength",
    "path_depth",
    "path_len",
    "path_entropy",
    "query_len",
    "query_param_count",
    "has_query",
    "brand_in_path",
]


def test_feature_list_artifact_exists():
    """Verifies that the trained feature schema artifact exists."""
    assert FEATURE_LIST_PATH.exists(), f"Missing artifact: {FEATURE_LIST_PATH}"
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)
    assert isinstance(cols, list)
    assert len(cols) == 38, f"Expected 38 features, found {len(cols)}"


def test_app_and_training_feature_parity():
    """Verifies 100% parity between training feature schema and app inference schema."""
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        training_features = json.load(f)

    app_features = model_service.FEATURE_COLUMNS
    assert app_features == training_features, (
        f"Train/Serve feature mismatch!\n"
        f"Training: {training_features}\n"
        f"App: {app_features}"
    )


def test_no_shortcut_features_in_general_model():
    """Asserts that shortcut, leaky, https, www, and path features are banned from the general model."""
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        features_in_model = json.load(f)

    for forbidden in FORBIDDEN_SHORTCUTS:
        assert forbidden not in features_in_model, (
            f"VIOLATION: Forbidden shortcut feature '{forbidden}' found in general model feature list!"
        )


def test_free_host_detection_and_capping():
    """Verifies that free-host URLs receive the security warning and trust score cap."""
    free_url = "https://legit-looking-service.firebaseapp.com/login"
    result = model_service.predict_single_url(free_url)

    assert result["is_free_host"] is True, "Failed to identify free-host URL"
    assert result["free_host_warning"] == "Hosted on a free platform; the platform's reputation does not vouch for this site."
    assert result["trust_score"] <= 50.0, f"Trust score ({result['trust_score']}) was not capped at 50.0 for free host"


def test_end_to_end_predictions():
    """Verifies that legitimate domains receive high trust and phishing receives low trust."""
    legit_res = model_service.predict_single_url("https://www.wikipedia.org")
    assert legit_res["prediction_label"] == "LEGITIMATE"
    assert legit_res["trust_score"] >= 70.0
    assert legit_res["risk_tier"] == "LOW"

    # Known homoglyph typosquat
    phish_res = model_service.predict_single_url("http://paypa1-account-update.tk/signin")
    assert phish_res["prediction_label"] == "PHISHING"
    assert phish_res["trust_score"] < 40.0
    assert phish_res["risk_tier"] == "HIGH"
    # Verify SHAP plain-English explanations present
    assert len(phish_res["risk_factors"]) > 0
    assert all("detail" in factor and "title" in factor for factor in phish_res["risk_factors"])


def test_flask_api_health_endpoint():
    """Verifies Flask /api/health endpoint reporting model version and feature count."""
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "online"
    assert data["model_version"] == "2.0"
    assert data["feature_count"] == 38
    assert data["active_models_count"] == 11


def test_flask_api_input_validation():
    """Verifies input validation on /api/scan."""
    client = app.test_client()
    # Empty URL
    resp_empty = client.post("/api/scan", json={"url": ""})
    assert resp_empty.status_code == 400

    # Overlong URL (>2048 chars)
    resp_long = client.post("/api/scan", json={"url": "http://example.com/" + "a" * 2100})
    assert resp_long.status_code == 400

    # Valid scan
    resp_valid = client.post("/api/scan", json={"url": "https://www.wikipedia.org"})
    assert resp_valid.status_code == 200
    data = resp_valid.get_json()
    assert "trust_score" in data
    assert "prediction_label" in data


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING FEATURE PARITY & INTEGRITY TESTS")
    print("=" * 65)
    test_feature_list_artifact_exists()
    print("[PASS] Feature list artifact exists.")
    test_app_and_training_feature_parity()
    print("[PASS] Train/Serve feature schema parity (38 features).")
    test_no_shortcut_features_in_general_model()
    print("[PASS] Forbidden shortcut, www, https, and path features strictly absent.")
    test_free_host_detection_and_capping()
    print("[PASS] Free-host platform detection, disclaimer, and 50% cap verified.")
    test_end_to_end_predictions()
    print("[PASS] End-to-end model predictions & SHAP explanations verified.")
    test_flask_api_health_endpoint()
    print("[PASS] Flask /api/health contract verified.")
    test_flask_api_input_validation()
    print("[PASS] Flask /api/scan input validation contract verified.")
    print("=" * 65)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)
