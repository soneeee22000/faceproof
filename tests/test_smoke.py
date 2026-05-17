"""Phase 0 smoke test — verifies the scaffold is wired correctly."""

from fastapi.testclient import TestClient

from faceproof.api import app

client = TestClient(app)


def test_health_ok() -> None:
    """The health endpoint responds 200 with status ``ok``."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "faceproof"
