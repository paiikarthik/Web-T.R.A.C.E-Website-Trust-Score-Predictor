from app import app
import json

client = app.test_client()

print("=" * 70)
print("TESTING PHISHGUARD AI FLASK APPLICATION ENDPOINTS")
print("=" * 70)

# 1. Root / UI
res = client.get("/")
print(f"[1] GET / -> HTTP {res.status_code} ({len(res.data)} bytes HTML)")
assert res.status_code == 200

# 2. Health check
res = client.get("/api/health")
print(f"[2] GET /api/health -> HTTP {res.status_code} | {res.get_json()['status']}")
assert res.status_code == 200

# 3. Models list
res = client.get("/api/models")
models = res.get_json()
print(f"[3] GET /api/models -> HTTP {res.status_code} ({len(models)} models available)")
assert res.status_code == 200
assert len(models) >= 10

# 4. Benchmarks
res = client.get("/api/benchmarks")
bench = res.get_json()
print(f"[4] GET /api/benchmarks -> HTTP {res.status_code} ({len(bench)} benchmark rows)")
assert res.status_code == 200

# 5. Live Scan: Legitimate URL
res = client.post("/api/scan", json={"url": "https://www.wikipedia.org", "model": "ML_RandomForest"})
data = res.get_json()
print(f"[5] POST /api/scan (Wikipedia): {data['prediction_label']} ({data['phishing_probability']*100:.1f}%) | Tier: {data['risk_tier_label']}")
assert res.status_code == 200
assert data["is_phishing"] is False

# 6. Live Scan: Phishing URL
res = client.post("/api/scan", json={"url": "http://paypal-verification-update.com.tk/login.php", "model": "ML_RandomForest"})
data = res.get_json()
print(f"[6] POST /api/scan (Phishing): {data['prediction_label']} ({data['phishing_probability']*100:.1f}%) | Tier: {data['risk_tier_label']}")
assert res.status_code == 200
assert data["is_phishing"] is True

# 7. Batch Scan
batch_urls = [
    "https://www.wikipedia.org",
    "http://192.168.1.1/admin/login",
    "https://www.southbankmosaics.com"
]
res = client.post("/api/batch-scan", json={"urls": batch_urls, "model": "ML_RandomForest"})
batch_data = res.get_json()
print(f"[7] POST /api/batch-scan -> HTTP {res.status_code} ({len(batch_data)} URLs evaluated)")
assert res.status_code == 200
assert len(batch_data) == 3

# 8. Static Figure Serving
res = client.get("/figures/model_comparison_unseen_domain.png")
print(f"[8] GET /figures/model_comparison_unseen_domain.png -> HTTP {res.status_code} ({len(res.data):,} bytes)")
assert res.status_code == 200

print("=" * 70)
print("ALL 8 TESTS PASSED WITH 100% SUCCESS!")
print("=" * 70)
