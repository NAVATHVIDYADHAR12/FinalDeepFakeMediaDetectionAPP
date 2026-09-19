"""Unit tests for configuration, face handling, the detector and forensics."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

import config as cfg
import forensics
from detector import DeepfakeDetector, colorize_heatmap, _preprocess, _softmax
from faces import FaceAnalyzer, FaceTracker, _iou


# ------------------------------------------------------------------ thresholds
class TestVerdicts:
    @pytest.mark.parametrize("score,expected", [
        (0.00, "AUTHENTIC"), (0.39, "AUTHENTIC"),
        (0.40, "SUSPICIOUS"), (0.64, "SUSPICIOUS"),
        (0.65, "FAKE"), (1.00, "FAKE"),
    ])
    def test_bands(self, score, expected):
        assert cfg.verdict_from_score(score) == expected

    def test_bands_are_ordered(self):
        assert cfg.SUSPICIOUS_THRESHOLD < cfg.FAKE_THRESHOLD

    def test_risk_increases_with_score(self):
        levels = [cfg.risk_from_score(s) for s in (0.1, 0.5, 0.7, 0.9)]
        assert levels == ["MINIMAL", "LOW", "MEDIUM", "HIGH"]

    def test_class_index_convention(self):
        # The whole system assumes index 1 means "fake". Guard it.
        assert (cfg.REAL_IDX, cfg.FAKE_IDX) == (0, 1)


# ----------------------------------------------------------------------- faces
class TestFaceDetection:
    def test_finds_the_face(self, face_analyzer, face_image):
        faces = face_analyzer.detect(face_image)
        assert len(faces) == 1
        assert faces[0]["score"] > cfg.FACE_SCORE_THRESHOLD

    def test_landmarks_shape(self, face_analyzer, face_image):
        lm = face_analyzer.detect(face_image)[0]["landmarks"]
        assert np.array(lm).shape == (5, 2)

    def test_no_face_in_noise(self, face_analyzer):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        assert face_analyzer.detect(noise) == []

    def test_faces_sorted_largest_first(self, face_analyzer, face_image):
        faces = face_analyzer.detect(face_image)
        areas = [f["bbox"][2] * f["bbox"][3] for f in faces]
        assert areas == sorted(areas, reverse=True)


class TestFaceCrop:
    def test_crop_is_larger_than_bbox(self, face_analyzer, face_image):
        bbox = face_analyzer.detect(face_image)[0]["bbox"]
        crop = face_analyzer.crop(face_image, bbox, margin=0.2)
        assert crop.shape[0] > bbox[3] * 0.9
        assert crop.size > 0

    def test_crop_clamps_to_image_bounds(self, face_analyzer, face_image):
        h, w = face_image.shape[:2]
        crop = face_analyzer.crop(face_image, [-50, -50, w + 200, h + 200])
        assert crop.shape[0] <= h and crop.shape[1] <= w

    def test_degenerate_bbox_falls_back_to_full_image(self, face_analyzer, face_image):
        crop = face_analyzer.crop(face_image, [10, 10, 0, 0], margin=0.0)
        assert crop.shape == face_image.shape


class TestEmbeddings:
    def test_embedding_dimensions(self, face_analyzer, face_image):
        face = face_analyzer.detect(face_image)[0]
        emb = face_analyzer.embed(face_image, face["_raw"])
        assert emb is not None and emb.shape == (128,)

    def test_same_face_is_self_similar(self, face_analyzer, face_image):
        face = face_analyzer.detect(face_image)[0]
        a = face_analyzer.embed(face_image, face["_raw"])
        b = face_analyzer.embed(face_image, face["_raw"])
        assert FaceAnalyzer.cosine(a, b) > 0.99

    def test_cosine_bounds(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert FaceAnalyzer.cosine(v, v) == pytest.approx(1.0)
        assert FaceAnalyzer.cosine(v, -v) == pytest.approx(-1.0)
        assert FaceAnalyzer.cosine(v, np.zeros(3, dtype=np.float32)) == 0.0

    def test_match_returns_none_below_threshold(self, face_analyzer):
        rng = np.random.default_rng(1)
        probe = rng.standard_normal(128).astype(np.float32)
        gallery = {"someone": rng.standard_normal(128).astype(np.float32)}
        name, _ = face_analyzer.match(probe, gallery, threshold=0.99)
        assert name is None

    def test_match_finds_identical_vector(self, face_analyzer):
        rng = np.random.default_rng(2)
        vec = rng.standard_normal(128).astype(np.float32)
        name, sim = face_analyzer.match(vec, {"alice": vec, "bob": -vec})
        assert name == "alice" and sim > 0.99

    def test_empty_gallery_is_safe(self, face_analyzer):
        name, sim = face_analyzer.match(np.ones(128, dtype=np.float32), {})
        assert name is None and sim == 0.0


# -------------------------------------------------------------------- tracking
class TestTracker:
    def test_iou_identical_boxes(self):
        assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        assert _iou([0, 0, 10, 10], [50, 50, 10, 10]) == 0.0

    def test_same_person_becomes_one_track(self):
        emb = np.ones(128, dtype=np.float32).tolist()
        t = FaceTracker()
        for frame in range(5):
            t.update(frame, [{"bbox": [10 + frame, 10, 50, 50],
                              "embedding": emb, "fake_probability": 0.8}])
        summary = t.summary()
        assert len(summary) == 1
        assert summary[0]["frames_seen"] == 5

    def test_two_people_become_two_tracks(self):
        a = np.eye(128, dtype=np.float32)[0].tolist()
        b = np.eye(128, dtype=np.float32)[1].tolist()
        t = FaceTracker()
        for frame in range(3):
            t.update(frame, [
                {"bbox": [0, 0, 40, 40], "embedding": a, "fake_probability": 0.9},
                {"bbox": [200, 0, 40, 40], "embedding": b, "fake_probability": 0.1},
            ])
        summary = t.summary()
        assert len(summary) == 2
        verdicts = {s["verdict"] for s in summary}
        assert verdicts == {"FAKE", "AUTHENTIC"}

    def test_flicker_is_flagged_as_inconsistent(self):
        emb = np.ones(128, dtype=np.float32).tolist()
        t = FaceTracker()
        for frame, score in enumerate([0.05, 0.95, 0.05, 0.95, 0.05]):
            t.update(frame, [{"bbox": [0, 0, 40, 40],
                              "embedding": emb, "fake_probability": score}])
        assert t.summary()[0]["temporally_inconsistent"] is True

    def test_steady_scores_are_not_flagged(self):
        emb = np.ones(128, dtype=np.float32).tolist()
        t = FaceTracker()
        for frame, score in enumerate([0.10, 0.11, 0.09, 0.10, 0.12]):
            t.update(frame, [{"bbox": [0, 0, 40, 40],
                              "embedding": emb, "fake_probability": score}])
        assert t.summary()[0]["temporally_inconsistent"] is False

    def test_falls_back_to_iou_without_embeddings(self):
        t = FaceTracker()
        for frame in range(4):
            t.update(frame, [{"bbox": [10, 10, 50, 50],
                              "embedding": None, "fake_probability": 0.5}])
        assert len(t.summary()) == 1


# -------------------------------------------------------------------- detector
class TestDetector:
    def test_untrained_manifest_is_not_ready_outside_test_override(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path)
        (tmp_path / "manifest.json").write_text(
            '{"dummy": true, "models": [{"arch": "dummy_testnet", "dummy": true}]}',
            encoding="utf-8",
        )
        monkeypatch.delenv("OMNIGUARD_ALLOW_DUMMY_MODELS", raising=False)
        assert not DeepfakeDetector().ready

    def test_loads_at_least_one_model(self, detector):
        assert detector.ready
        assert len(detector.models) >= 1

    def test_face_models_are_not_loaded_as_classifiers(self, detector):
        archs = {m["arch"] for m in detector.models}
        assert "face_detection_yunet" not in archs
        assert "face_recognition_sface" not in archs

    def test_prediction_shape_and_ranges(self, detector, face_image):
        r = detector.predict(face_image)
        assert 0.0 <= r["fake_probability"] <= 1.0
        assert 0.0 <= r["authenticity_score"] <= 100.0
        assert r["verdict"] in {"AUTHENTIC", "SUSPICIOUS", "FAKE"}
        assert r["risk_level"] in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
        assert len(r["models"]) == len(detector.models)

    def test_authenticity_is_the_inverse_of_fake_probability(self, detector, face_image):
        r = detector.predict(face_image)
        assert r["authenticity_score"] == pytest.approx(
            (1 - r["fake_probability"]) * 100, abs=0.15)

    def test_verdict_matches_the_score(self, detector, face_image):
        r = detector.predict(face_image)
        assert r["verdict"] == cfg.verdict_from_score(r["fake_probability"])

    def test_deterministic(self, detector, face_image):
        a = detector.predict(face_image)["fake_probability"]
        b = detector.predict(face_image)["fake_probability"]
        assert a == b

    def test_heatmap_is_normalized(self, detector, face_image):
        hm = detector.predict(face_image, want_heatmap=True).get("heatmap")
        assert hm is not None
        assert hm.shape == face_image.shape[:2]
        assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6

    def test_heatmap_can_be_skipped(self, detector, face_image):
        assert "heatmap" not in detector.predict(face_image, want_heatmap=False)

    def test_tiny_image_does_not_crash(self, detector):
        assert detector.predict(np.zeros((8, 8, 3), dtype=np.uint8))["verdict"]

    def test_calibrated_binary_detector_exposes_ai_likelihood_not_distance(self):
        class Session:
            @staticmethod
            def run(_outputs, _inputs):
                # sigmoid(log(4)) == 0.8
                return [np.array([[np.log(4.0)]], dtype=np.float32)]

        detector = DeepfakeDetector.__new__(DeepfakeDetector)
        detector.manifest = {
            "output_type": "sigmoid_logit",
            "confidence_kind": "threshold_calibrated_model_score",
            "decision_threshold": 0.65,
        }
        detector.models = [{
            "arch": "test_ai_detector",
            "name": "Test AI detector",
            "session": Session(),
            "input_name": "input",
            "classifier_weights": None,
            "metrics": {},
        }]

        result = detector.predict(
            np.zeros((32, 32, 3), dtype=np.uint8), want_heatmap=False
        )
        assert result["fake_probability"] == pytest.approx(0.8, abs=1e-4)
        assert result["confidence"] == pytest.approx(0.8, abs=1e-4)
        assert result["confidence_kind"] == "threshold_calibrated_model_score"
        assert result["decision_threshold"] == 0.65

    def test_colorize_matches_input_size(self, detector, face_image):
        hm = detector.predict(face_image)["heatmap"]
        overlay = colorize_heatmap(hm, face_image)
        assert overlay.shape == face_image.shape


class TestDetectorInternals:
    def test_preprocess_shape_and_normalization(self, face_image):
        batch = _preprocess(face_image)
        assert batch.shape == (1, 3, cfg.CLASSIFIER_INPUT_SIZE, cfg.CLASSIFIER_INPUT_SIZE)
        assert batch.dtype == np.float32
        # ImageNet-normalized data sits roughly in [-2.2, 2.7]
        assert -4 < batch.min() and batch.max() < 4

    def test_softmax_sums_to_one(self):
        out = _softmax(np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32))
        assert np.allclose(out.sum(axis=-1), 1.0)

    def test_cam_handles_mismatched_weights(self):
        feats = np.random.default_rng(0).standard_normal((1, 32, 7, 7)).astype(np.float32)
        wrong = np.zeros((2, 999), dtype=np.float32)
        cam = DeepfakeDetector._cam(feats, wrong)
        assert cam is not None and cam.shape == (7, 7)

    def test_cam_rejects_non_spatial_features(self):
        assert DeepfakeDetector._cam(np.zeros((1, 32), dtype=np.float32), None) is None


# ------------------------------------------------------------------- forensics
class TestForensics:
    def test_metadata_reports_dimensions(self, face_image_path):
        meta = forensics.read_metadata(face_image_path)
        assert meta["dimensions"] == "512x512"
        assert meta["format"] in {"JPEG", "PNG"}

    def test_metadata_on_stripped_file(self, noise_image_path):
        meta = forensics.read_metadata(noise_image_path)
        assert meta["metadata_stripped"] is True
        assert meta["has_gps"] is False

    def test_c2pa_absent_is_not_an_accusation(self, face_image_path):
        c2pa = forensics.check_c2pa(face_image_path)
        assert c2pa["status"] in {"PRESENT", "NOT_FOUND"}
        if not c2pa["signature_found"]:
            assert "not evidence of manipulation" in c2pa["note"]

    def test_ela_returns_numbers(self, face_image_path):
        ela = forensics.error_level_analysis(face_image_path)
        assert "error" not in ela
        assert 0.0 <= ela["mean_error"] <= 1.0
        assert isinstance(ela["suspicious_regions"], bool)

    def test_ela_on_missing_file_is_handled(self, tmp_path):
        assert "error" in forensics.error_level_analysis(tmp_path / "nope.jpg")

    def test_normal_resolution_image_has_quality_report(self, face_image_path, face_image):
        quality = forensics.assess_input_quality(face_image_path, face_image)
        assert quality["short_edge_px"] == 512
        assert quality["reliability"] in {"STANDARD", "LIMITED"}
        assert quality["bits_per_pixel"] > 0

    def test_low_resolution_is_flagged_as_limited(self, tmp_path):
        image = np.full((120, 160, 3), 127, dtype=np.uint8)
        path = tmp_path / "tiny.jfif"
        assert cv2.imwrite(str(path.with_suffix(".jpg")), image)
        jpeg_path = path.with_suffix(".jpg")
        quality = forensics.assess_input_quality(jpeg_path, image)
        assert quality["reliability"] == "LIMITED"
        assert any("upscale" in item for item in quality["limitations"])


class TestFindings:
    def _findings(self, fake_p, **kw):
        meta = {"metadata_stripped": False, "camera_make": "Canon", "software": None}
        meta.update(kw.pop("metadata", {}))
        return forensics.build_findings(
            {"fake_probability": fake_p, "model_agreement": kw.pop("agreement", 1.0)},
            meta,
            kw.pop("c2pa", {"signature_found": True}),
            kw.pop("ela", {"suspicious_regions": False}),
            kw.pop("face_count", 1),
        )

    def test_high_score_yields_high_severity(self):
        assert any(f["severity"] == "high" for f in self._findings(0.95))

    def test_low_score_yields_no_high_severity(self):
        assert not any(f["severity"] == "high" for f in self._findings(0.02))

    def test_disagreement_is_surfaced(self):
        texts = " ".join(f["text"] for f in self._findings(0.5, agreement=0.1))
        assert "disagree" in texts.lower()

    def test_missing_metadata_is_flagged(self):
        texts = " ".join(f["text"] for f in self._findings(
            0.1, metadata={"metadata_stripped": True, "camera_make": None}))
        assert "metadata" in texts.lower()

    def test_no_face_case_is_explained(self):
        texts = " ".join(f["text"] for f in self._findings(0.1, face_count=0))
        assert "no face" in texts.lower()

    def test_every_finding_is_well_formed(self):
        for f in self._findings(0.8):
            assert f["severity"] in {"high", "medium", "info"}
            assert isinstance(f["text"], str) and f["text"]
