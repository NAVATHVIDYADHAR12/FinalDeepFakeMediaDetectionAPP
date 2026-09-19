"""HTTP-level tests against the FastAPI app, using Starlette's TestClient.

These run the real application including its startup lifespan, so a broken
model load or a bad route shows up here rather than at demo time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config as cfg
import main


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _upload(path, field="file", content_type="image/jpeg"):
    return {field: (path.name, path.read_bytes(), content_type)}


# ---------------------------------------------------------------------- system
class TestSystemEndpoints:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["face_analyzer"] is True

    def test_system_info(self, client):
        body = client.get("/api/system/info").json()
        assert "detector" in body
        assert body["thresholds"]["fake"] > body["thresholds"]["suspicious"]
        assert ".jpg" in body["supported"]["image"]
        assert ".jfif" in body["supported"]["image"]
        assert ".mp4" in body["supported"]["video"]

    def test_models_endpoint(self, client):
        body = client.get("/api/models").json()
        assert "ready" in body and "models" in body
        if body["ready"]:
            for m in body["models"]:
                assert "arch" in m and "name" in m

    def test_openapi_schema_builds(self, client):
        assert client.get("/openapi.json").status_code == 200


# ----------------------------------------------------------------------- scans
class TestScanEndpoints:
    def test_scan_image(self, client, face_image_path):
        r = client.post("/api/scan/image", files=_upload(face_image_path))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["media_type"] == "image"
        assert body["verdict"] in {"AUTHENTIC", "SUSPICIOUS", "FAKE"}
        assert body["faces_detected"] == 1
        assert 0.0 <= body["fake_probability"] <= 1.0

    def test_scan_response_hides_embeddings(self, client, face_image_path):
        body = client.post("/api/scan/image", files=_upload(face_image_path)).json()
        assert "_embedding" not in body["faces"][0]

    def test_auto_route_picks_image(self, client, face_image_path):
        body = client.post("/api/scan", files=_upload(face_image_path)).json()
        assert body["media_type"] == "image"

    def test_jfif_extension_is_accepted_as_jpeg(self, client, face_image_path):
        response = client.post(
            "/api/scan",
            files={"file": ("shared-photo.jfif", face_image_path.read_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["media_type"] == "image"
        assert body["filename"] == "shared-photo.jfif"

    def test_auto_route_picks_video(self, client, video_path):
        """Regression: /api/scan used to call the /api/scan/video handler
        directly, so FastAPI never resolved its `Form(None)` default and
        max_frames arrived as a Form object, blowing up inside the sampler.
        Every video dropped on the dashboard returned a 500."""
        r = client.post("/api/scan",
                        files={"file": (video_path.name, video_path.read_bytes(), "video/mp4")})
        assert r.status_code == 200, r.text
        assert r.json()["media_type"] == "video"

    def test_video_report_has_model_rollup(self, client, video_path):
        body = client.post("/api/scan",
                           files={"file": (video_path.name, video_path.read_bytes(), "video/mp4")}).json()
        assert len(body["models"]) >= 1
        for m in body["models"]:
            assert 0.0 <= m["fake_probability"] <= 1.0
            assert m["verdict"] in {"AUTHENTIC", "SUSPICIOUS", "FAKE"}

    def test_uploads_are_not_left_on_disk(self, client, face_image_path, video_path):
        """Regression: uploads accumulated forever - 49 files / 12 MB after a
        single audit run."""
        before = set(cfg.UPLOAD_DIR.glob("*"))
        client.post("/api/scan", files=_upload(face_image_path))
        client.post("/api/scan",
                    files={"file": (video_path.name, video_path.read_bytes(), "video/mp4")})
        client.post("/api/identity/enroll", data={"name": "Cleanup Test"},
                    files=_upload(face_image_path))
        client.post("/api/identity/match", files=_upload(face_image_path))
        assert set(cfg.UPLOAD_DIR.glob("*")) == before

    def test_previews_are_size_capped(self, client, face_image_path):
        """Previews ride inside the JSON, so they must stay small."""
        body = client.post("/api/scan", files=_upload(face_image_path)).json()
        for face in body["faces"]:
            for key in ("crop_preview", "heatmap_preview"):
                if face.get(key):
                    assert len(face[key]) < 120_000, f"{key} is {len(face[key])} bytes"

    def test_scan_video(self, client, video_path):
        r = client.post("/api/scan/video",
                        files={"file": (video_path.name, video_path.read_bytes(), "video/mp4")},
                        data={"max_frames": 6})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["media_type"] == "video"
        assert body["frames_analyzed"] <= 6

    def test_unsupported_extension_rejected(self, client):
        r = client.post("/api/scan",
                        files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"]

    def test_empty_file_rejected(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

    def test_corrupt_image_rejected(self, client):
        r = client.post("/api/scan/image",
                        files={"file": ("bad.jpg", b"not an image at all", "image/jpeg")})
        assert r.status_code == 400


# ------------------------------------------------------------------- retrieval
class TestHistoryEndpoints:
    def test_scan_is_retrievable_afterwards(self, client, face_image_path):
        scan_id = client.post("/api/scan/image",
                              files=_upload(face_image_path)).json()["scan_id"]
        r = client.get(f"/api/scan/{scan_id}")
        assert r.status_code == 200
        assert r.json()["scan_id"] == scan_id

    def test_unknown_scan_returns_404(self, client):
        assert client.get("/api/scan/DOES_NOT_EXIST").status_code == 404

    def test_list_scans(self, client, face_image_path):
        client.post("/api/scan/image", files=_upload(face_image_path))
        body = client.get("/api/scans?limit=5").json()
        assert isinstance(body["scans"], list)
        assert len(body["scans"]) >= 1

    def test_stats_shape(self, client):
        body = client.get("/api/stats").json()
        for key in ("total_scans", "authentic", "suspicious", "fake",
                    "authentic_pct", "by_media_type", "trend"):
            assert key in body

    def test_delete_scan(self, client, face_image_path):
        scan_id = client.post("/api/scan/image",
                              files=_upload(face_image_path)).json()["scan_id"]
        assert client.delete(f"/api/scan/{scan_id}").status_code == 200
        assert client.get(f"/api/scan/{scan_id}").status_code == 404

    def test_delete_unknown_scan_404(self, client):
        assert client.delete("/api/scan/NOPE").status_code == 404


# ------------------------------------------------------------------ identities
class TestIdentityEndpoints:
    def test_enroll_then_match(self, client, face_image_path):
        r = client.post("/api/identity/enroll",
                        data={"name": "Test Person"},
                        files=_upload(face_image_path))
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "Test Person"
        assert r.json()["dimensions"] == 128

        m = client.post("/api/identity/match", files=_upload(face_image_path))
        assert m.status_code == 200
        body = m.json()
        assert body["faces_detected"] == 1
        assert body["matches"][0]["identity"] == "Test Person"
        assert body["matches"][0]["similarity"] > 0.9

    def test_enroll_rejects_image_without_face(self, client, noise_image_path):
        r = client.post("/api/identity/enroll",
                        data={"name": "Nobody"},
                        files={"file": (noise_image_path.name,
                                        noise_image_path.read_bytes(), "image/png")})
        assert r.status_code == 422

    def test_list_identities(self, client, face_image_path):
        client.post("/api/identity/enroll", data={"name": "Listed"},
                    files=_upload(face_image_path))
        names = [i["name"] for i in client.get("/api/identities").json()["identities"]]
        assert "Listed" in names

    def test_delete_identity(self, client, face_image_path):
        client.post("/api/identity/enroll", data={"name": "Temp"},
                    files=_upload(face_image_path))
        assert client.delete("/api/identity/Temp").status_code == 200
        assert client.delete("/api/identity/Temp").status_code == 404

    def test_match_against_empty_gallery_is_safe(self, client, face_image_path):
        for i in client.get("/api/identities").json()["identities"]:
            client.delete(f"/api/identity/{i['name']}")
        body = client.post("/api/identity/match", files=_upload(face_image_path)).json()
        assert body["gallery_size"] == 0
        assert body["matches"][0]["matched"] is False
