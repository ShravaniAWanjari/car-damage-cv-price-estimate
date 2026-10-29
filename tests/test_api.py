"""API surface: demo exposes the website + annotated images, prod exposes neither."""
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "s01.jpg"


def _client(monkeypatch, mode):
    monkeypatch.setenv("CARDAMAGE_MODE", mode)
    monkeypatch.setenv("CARDAMAGE_DEVICE", "cpu")
    import cardamage.api as api
    api = importlib.reload(api)  # mounts are decided at import time from the mode
    return TestClient(api.app)


@pytest.fixture(scope="module")
def sample_bytes():
    return SAMPLE.read_bytes()


def test_demo_site_and_annotations(monkeypatch, sample_bytes):
    with _client(monkeypatch, "demo") as c:
        assert "Car damage analysis" in c.get("/").text
        info = c.get("/api/info").json()
        assert info["mode"] == "demo" and info["damage_model"]["name"]
        assert c.get("/api/samples").json()["samples"]
        r = c.post("/analyze", files=[("files", ("s01.jpg", sample_bytes, "image/jpeg"))])
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "demo" and body["images"][0]["image_id"] == "s01"
        assert c.get(body["demo"]["annotated"][0]).status_code == 200


def test_prod_has_no_site_or_images(monkeypatch, sample_bytes):
    with _client(monkeypatch, "prod") as c:
        assert c.get("/").status_code == 404
        assert c.get("/api/info").status_code == 404
        assert c.get("/demo/x").status_code == 404
        r = c.post("/analyze", files=[("files", ("s01.jpg", sample_bytes, "image/jpeg"))])
        assert r.status_code == 200 and "demo" not in r.json()
        assert c.get("/schema").json()["schema_version"]


def test_bad_upload_rejected(monkeypatch):
    with _client(monkeypatch, "prod") as c:
        r = c.post("/analyze", files=[("files", ("x.jpg", b"not an image", "image/jpeg"))])
        assert r.status_code == 422


def test_empty_upload_is_422(monkeypatch):
    with _client(monkeypatch, "prod") as c:
        assert c.post("/analyze", files=[("files", ("e.jpg", b"", "image/jpeg"))]).status_code == 422


def test_duplicate_and_unsafe_filenames(monkeypatch, sample_bytes):
    with _client(monkeypatch, "demo") as c:
        files = [("files", ("photo.jpg", sample_bytes, "image/jpeg")),
                 ("files", ("photo.jpg", sample_bytes, "image/jpeg")),
                 ("files", ("my car #1 (front)?.jpg", sample_bytes, "image/jpeg"))]
        body = c.post("/analyze", files=files).json()
        ids = [i["image_id"] for i in body["images"]]
        assert ids[:2] == ["photo", "photo-2"] and len(set(ids)) == 3
        urls = body["demo"]["annotated"]
        assert len(set(urls)) == 3
        assert all(c.get(u).status_code == 200 for u in urls)   # incl. the one with '#' and '?' in its name
        dmg_ids = [d["id"] for i in body["images"] for d in i["damages"]]
        assert len(dmg_ids) == len(set(dmg_ids))                # damage ids stay unique across images
