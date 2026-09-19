"""End-to-end tests for the image pipeline, the video pipeline and the database."""

from __future__ import annotations

import numpy as np
import pytest

import config as cfg
import database as db
import pipeline
import video as video_mod


# ------------------------------------------------------------------ image path
class TestImagePipeline:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls, request):
        detector = request.getfixturevalue("detector")
        faces = request.getfixturevalue("face_analyzer")
        path = request.getfixturevalue("face_image_path")
        return pipeline.analyze_image(path, detector, faces)

    def test_top_level_fields(self, report):
        for key in ("scan_id", "media_type", "filename", "verdict", "risk_level",
                    "fake_probability", "authenticity_score", "faces", "findings",
                    "timeline", "metadata", "c2pa", "ela", "models"):
            assert key in report, f"missing {key}"

    def test_media_type_and_scan_id(self, report):
        assert report["media_type"] == "image"
        assert report["scan_id"].startswith("SCN")

    def test_score_ranges(self, report):
        assert 0.0 <= report["fake_probability"] <= 1.0
        assert 0.0 <= report["authenticity_score"] <= 100.0

    def test_verdict_is_consistent_with_score(self, report):
        assert report["verdict"] == cfg.verdict_from_score(report["fake_probability"])

    def test_face_was_detected(self, report):
        assert report["faces_detected"] == 1
        assert len(report["faces"]) == 1
        assert report["faces"][0]["is_full_frame"] is False

    def test_dimensions_reported(self, report):
        assert report["dimensions"] == "512x512"

    def test_previews_are_data_uris(self, report):
        face = report["faces"][0]
        assert face["crop_preview"].startswith("data:image/jpeg;base64,")
        assert face["heatmap_preview"].startswith("data:image/jpeg;base64,")

    def test_timeline_is_monotonic(self, report):
        elapsed = [step["elapsed_ms"] for step in report["timeline"]]
        assert elapsed == sorted(elapsed)
        assert len(elapsed) >= 5

    def test_overall_score_is_the_worst_face(self, report):
        worst = max(f["fake_probability"] for f in report["faces"])
        assert report["fake_probability"] == pytest.approx(worst, abs=1e-4)

    def test_model_rollup_covers_every_model(self, report, detector):
        assert len(report["models"]) == len(detector.models)
        for m in report["models"]:
            assert 0.0 <= m["fake_probability"] <= 1.0

    def test_findings_present(self, report):
        assert len(report["findings"]) >= 2

    def test_embedding_is_captured_for_identity_use(self, report):
        assert report["faces"][0]["has_embedding"] is True


class TestImagePipelineEdgeCases:
    def test_face_only_model_does_not_score_no_face_full_frame(self, noise_image_path,
                                                               detector, face_analyzer):
        report = pipeline.analyze_image(noise_image_path, detector, face_analyzer)
        assert report["faces_detected"] == 0
        assert len(report["faces"]) == 1
        assert report["faces"][0]["is_full_frame"] is True
        assert report["verdict"] == "UNVERIFIED"
        assert report["risk_level"] == "UNKNOWN"
        assert report["fake_probability"] is None
        assert report["confidence"] is None
        assert report["faces"][0]["verdict"] == "UNVERIFIED"
        assert any("no face" in f["text"].lower() for f in report["findings"])

    def test_ai_generation_model_scores_no_face_full_frame(self, noise_image_path,
                                                            detector, face_analyzer):
        class AIGenerationDetector:
            ready = True
            task = "ai_generation"
            models = detector.models

            @staticmethod
            def supports(task):
                return task == "ai_generation"

            @staticmethod
            def predict(image, want_heatmap=True):
                result = detector.predict(image, want_heatmap=want_heatmap)
                result["confidence_kind"] = "threshold_calibrated_model_score"
                result["confidence"] = result["fake_probability"]
                return result

        report = pipeline.analyze_image(
            noise_image_path, detector, face_analyzer,
            ai_detector=AIGenerationDetector(),
        )

        assert report["faces_detected"] == 0
        assert report["fake_probability"] is not None
        # The 256px noise fixture is below the 384px classifier input size.
        # It is scored for diagnostics, but the final verdict must abstain.
        assert report["verdict"] == "UNVERIFIED"
        assert report["decision_reliability"] == "LIMITED"
        assert report["analysis_scope"] == ["ai_generation"]
        assert report["generation_analysis"] is not None
        assert report["confidence_kind"] == "threshold_calibrated_model_score"
        assert report["confidence"] == report["fake_probability"]
        assert all(model["arch"].startswith("ai:") for model in report["models"])

    def test_corrupt_file_raises_value_error(self, tmp_path, detector, face_analyzer):
        bad = tmp_path / "broken.jpg"
        bad.write_bytes(b"this is not an image")
        with pytest.raises(ValueError):
            pipeline.analyze_image(bad, detector, face_analyzer)

    def test_scan_ids_are_unique(self, face_image_path, detector, face_analyzer):
        a = pipeline.analyze_image(face_image_path, detector, face_analyzer, False)
        b = pipeline.analyze_image(face_image_path, detector, face_analyzer, False)
        assert a["scan_id"] != b["scan_id"]

    def test_previews_can_be_disabled(self, face_image_path, detector, face_analyzer):
        report = pipeline.analyze_image(face_image_path, detector, face_analyzer,
                                        want_previews=False)
        assert "crop_preview" not in report["faces"][0]


# ------------------------------------------------------------------ video path
class TestVideoPipeline:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls, request):
        return video_mod.analyze_video(
            request.getfixturevalue("video_path"),
            request.getfixturevalue("detector"),
            request.getfixturevalue("face_analyzer"),
            max_frames=8,
        )

    def test_top_level_fields(self, report):
        for key in ("scan_id", "media_type", "verdict", "fake_probability",
                    "frame_scores", "tracks", "temporal_std", "duration_sec",
                    "frames_analyzed", "findings", "timeline"):
            assert key in report, f"missing {key}"

    def test_media_type(self, report):
        assert report["media_type"] == "video"

    def test_respects_the_frame_budget(self, report):
        assert 0 < report["frames_analyzed"] <= 8

    def test_frames_were_sampled_across_the_clip(self, report):
        frames = [f["frame"] for f in report["frame_scores"]]
        assert frames == sorted(frames)
        assert len(set(frames)) == len(frames)

    def test_faces_found_in_the_clip(self, report):
        assert report["frames_with_faces"] > 0

    def test_person_was_tracked(self, report):
        assert report["people_detected"] >= 1
        track = report["tracks"][0]
        assert track["frames_seen"] >= 1
        assert 0.0 <= track["mean_fake_probability"] <= 1.0

    def test_most_suspicious_frame_is_captured(self, report):
        frame = report["most_suspicious_frame"]
        assert frame is not None
        assert frame["preview"].startswith("data:image/jpeg;base64,")

    def test_score_ranges(self, report):
        assert 0.0 <= report["fake_probability"] <= 1.0
        assert report["temporal_std"] >= 0.0

    def test_verdict_is_consistent(self, report):
        assert report["verdict"] == cfg.verdict_from_score(report["fake_probability"])

    def test_unreadable_video_raises(self, tmp_path, detector, face_analyzer):
        bad = tmp_path / "broken.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(ValueError):
            video_mod.analyze_video(bad, detector, face_analyzer)


# -------------------------------------------------------------------- database
class TestDatabase:
    def _report(self, scan_id="SCN000001", verdict="FAKE", prob=0.9):
        return {
            "scan_id": scan_id, "media_type": "image", "filename": "x.jpg",
            "verdict": verdict, "risk_level": "HIGH", "fake_probability": prob,
            "authenticity_score": (1 - prob) * 100, "confidence": 0.8,
            "faces_detected": 1, "file_size_bytes": 1234, "processing_ms": 42.0,
            "faces": [{"face_id": 1, "_embedding": [0.1] * 128}],
        }

    def test_roundtrip(self):
        db.save_scan(self._report())
        loaded = db.get_scan("SCN000001")
        assert loaded["verdict"] == "FAKE"
        assert loaded["fake_probability"] == 0.9

    def test_embeddings_are_not_persisted(self):
        db.save_scan(self._report())
        assert "_embedding" not in db.get_scan("SCN000001")["faces"][0]

    def test_missing_scan_returns_none(self):
        assert db.get_scan("NOPE") is None

    def test_recent_scans_are_newest_first(self):
        for i in range(3):
            db.save_scan(self._report(scan_id=f"SCN00000{i}"))
        rows = db.recent_scans(limit=10)
        assert len(rows) == 3

    def test_filter_by_verdict(self):
        db.save_scan(self._report("SCN_A", "FAKE", 0.9))
        db.save_scan(self._report("SCN_B", "AUTHENTIC", 0.1))
        assert len(db.recent_scans(verdict="FAKE")) == 1
        assert len(db.recent_scans(verdict="AUTHENTIC")) == 1

    def test_delete(self):
        db.save_scan(self._report("SCN_DEL"))
        assert db.delete_scan("SCN_DEL") is True
        assert db.delete_scan("SCN_DEL") is False

    def test_stats_percentages_add_up(self):
        db.save_scan(self._report("S1", "FAKE", 0.9))
        db.save_scan(self._report("S2", "AUTHENTIC", 0.1))
        db.save_scan(self._report("S3", "AUTHENTIC", 0.2))
        s = db.stats()
        assert s["total_scans"] == 3
        assert s["fake"] == 1 and s["authentic"] == 2
        assert s["authentic_pct"] + s["suspicious_pct"] + s["fake_pct"] == pytest.approx(100.0, abs=0.2)

    def test_stats_on_empty_database(self):
        s = db.stats()
        assert s["total_scans"] == 0
        assert s["authentic_pct"] == 0.0

    def test_enroll_and_load_identity(self):
        vec = np.ones(128, dtype=np.float32)
        db.enroll_identity("alice", vec)
        gallery = db.load_gallery()
        assert "alice" in gallery
        assert np.allclose(gallery["alice"], vec)

    def test_reenrolling_averages_samples(self):
        db.enroll_identity("bob", np.zeros(128, dtype=np.float32))
        result = db.enroll_identity("bob", np.ones(128, dtype=np.float32))
        assert result["sample_count"] == 2
        assert np.allclose(db.load_gallery()["bob"], 0.5)

    def test_delete_identity(self):
        db.enroll_identity("carol", np.ones(128, dtype=np.float32))
        assert db.delete_identity("carol") is True
        assert db.delete_identity("carol") is False
