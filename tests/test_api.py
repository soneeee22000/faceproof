"""Tests for the FastAPI verification endpoints.

Error-path tests run without the CV stack — decoding fails before any model
loads. Happy-path tests need the optional ``[ml]`` extra and are skipped
without it.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from faceproof.api import app

client = TestClient(app)


def _png(color: tuple[int, int, int] = (120, 130, 140)) -> bytes:
    """Encode a small solid-colour PNG."""
    buffer = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(data: bytes, name: str = "image.png", mime: str = "image/png") -> tuple:
    """Build a multipart file tuple for the test client."""
    return (name, data, mime)


def test_verify_rejects_a_non_image_upload() -> None:
    """A non-image upload yields a 415 INVALID_IMAGE envelope."""
    response = client.post(
        "/api/verify",
        files={
            "id_portrait": _upload(b"not an image", "a.txt", "text/plain"),
            "selfie": _upload(b"not an image", "b.txt", "text/plain"),
        },
    )
    assert response.status_code == 415
    body = response.json()
    assert body["data"] is None
    assert body["error"]["code"] == "INVALID_IMAGE"


def test_match_rejects_a_non_image_upload() -> None:
    """The match endpoint rejects non-image uploads the same way."""
    response = client.post(
        "/api/match",
        files={
            "image_a": _upload(b"nope", "a.txt", "text/plain"),
            "image_b": _upload(b"nope", "b.txt", "text/plain"),
        },
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_liveness_rejects_a_non_image_upload() -> None:
    """The liveness endpoint rejects non-image uploads."""
    response = client.post(
        "/api/liveness",
        files={"selfie": _upload(b"nope", "s.txt", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_verify_requires_both_files() -> None:
    """A missing upload yields a 422 INVALID_REQUEST envelope."""
    response = client.post("/api/verify", files={"id_portrait": _upload(_png())})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_match_happy_path_returns_envelope() -> None:
    """Two identical real faces match, returning the data envelope."""
    pytest.importorskip("insightface")
    import cv2
    from insightface.data import get_image

    encoded = bytes(cv2.imencode(".png", get_image("t1"))[1])
    response = client.post(
        "/api/match",
        files={"image_a": _upload(encoded), "image_b": _upload(encoded)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["data"]["is_match"] is True
    assert 0.0 <= body["data"]["similarity"] <= 1.0


def test_verify_happy_path_returns_envelope() -> None:
    """The full verify pipeline returns a well-formed envelope."""
    pytest.importorskip("insightface")
    pytest.importorskip("torch")
    import cv2
    from insightface.data import get_image

    encoded = bytes(cv2.imencode(".png", get_image("t1"))[1])
    response = client.post(
        "/api/verify",
        files={"id_portrait": _upload(encoded), "selfie": _upload(encoded)},
    )
    if response.status_code == 503:
        pytest.skip("Silent-Face weights not downloaded")
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert isinstance(body["data"]["verified"], bool)
    assert len(body["data"]["reasons"]) >= 2
