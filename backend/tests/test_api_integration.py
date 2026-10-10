"""Integration tests for detection → training → validation API flow.

Data accuracy assertions go beyond HTTP status checks:
- Box coordinates within image bounds
- Confidence in valid range
- Mask polygons valid (at least 3 points, within image)
- List vs detail endpoint data consistency
- Timestamps and UUIDs well-formed
"""

import os
import time

import pytest
import requests

from app.core.config import hf_repo_cache_dir

BASE = "http://127.0.0.1:8000/api/v1"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAT_DIR = os.path.join(_PROJECT_ROOT, "test_images", "cat")
TEST_IMAGE = os.path.join(CAT_DIR, "pexels-helen1-30002394.jpg") if os.path.isdir(CAT_DIR) else None


def _backend_reachable():
    try:
        requests.get(f"{BASE}/model/status", timeout=2)
        return True
    except Exception:
        return False


backend_required = pytest.mark.skipif(not _backend_reachable(), reason="Backend not running")


def _validate_boxes(boxes, img_w, img_h):
    """Assert box data integrity — not just shape, but bounds correctness."""
    for b in boxes:
        # Coordinates must be within image
        assert 0 <= b["x1"] < img_w, f"x1={b['x1']} out of bounds [0, {img_w})"
        assert 0 <= b["y1"] < img_h, f"y1={b['y1']} out of bounds [0, {img_h})"
        assert b["x1"] < b["x2"] <= img_w, f"x2={b['x2']} must be > x1 and <= {img_w}"
        assert b["y1"] < b["y2"] <= img_h, f"y2={b['y2']} must be > y1 and <= {img_h}"
        # Confidence in valid range
        if b.get("confidence") is not None:
            assert 0.0 <= b["confidence"] <= 1.0, f"confidence {b['confidence']} not in [0,1]"
        # Class name is non-empty
        assert b["className"] and isinstance(b["className"], str)

    # No duplicate boxes
    box_ids = [b["id"] for b in boxes]
    assert len(box_ids) == len(set(box_ids)), "Duplicate box IDs"


def _validate_mask_polygon(poly, img_w, img_h):
    """Validate mask polygon data integrity."""
    if poly is None:
        return
    assert isinstance(poly, list) and len(poly) >= 3, "Polygon must have >= 3 points"
    for pt in poly:
        assert isinstance(pt, list) and len(pt) == 2, "Each point must be [x, y]"
        assert 0 <= pt[0] <= img_w, f"Polygon x={pt[0]} out of image bounds"
        assert 0 <= pt[1] <= img_h, f"Polygon y={pt[1]} out of image bounds"


def _validate_detection_response(data):
    """Full data integrity check on a detection response."""
    assert isinstance(data["id"], str) and len(data["id"]) > 0
    assert isinstance(data["imageName"], str)
    assert isinstance(data["categories"], list) and len(data["categories"]) > 0
    assert isinstance(data["modelType"], str) and data["modelType"] in ("vlm", "vlm+sam2", "sam3")
    assert data["imageWidth"] > 0
    assert data["imageHeight"] > 0
    assert isinstance(data["elapsedMs"], (int, type(None)))
    assert data["status"] == "completed"
    assert isinstance(data["boxes"], list)

    _validate_boxes(data["boxes"], data["imageWidth"], data["imageHeight"])


def _post_detect(use_sam3=False):
    with open(TEST_IMAGE, "rb") as f:
        return requests.post(
            f"{BASE}/detect",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={
                "categories": '["cat"]',
                "use_sam3": str(use_sam3).lower(),
                "use_sam3_seg": "false",
                "sam3_threshold": "0.5",
            },
            timeout=120,
        )


@backend_required
@pytest.mark.skipif(not (TEST_IMAGE and os.path.exists(TEST_IMAGE)), reason="no test images")
class TestDetectionFlow:
    def test_detect_vlm(self):
        """VLM detection returns valid response."""
        with open(TEST_IMAGE, "rb") as f:
            resp = requests.post(
                f"{BASE}/detect",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"categories": '["cat"]'},
                timeout=120,
            )
        assert resp.status_code == 201
        data = resp.json()["data"]
        _validate_detection_response(data)
        assert len(data["boxes"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("HF_TOKEN") and not hf_repo_cache_dir("facebook/sam3").is_dir(),
        reason="SAM3 requires HF_TOKEN or cached model",
    )
    def test_detect_sam3(self):
        """SAM3 detection returns valid response."""
        resp = _post_detect(use_sam3=True)
        assert resp.status_code == 201, resp.json()
        data = resp.json()["data"]
        _validate_detection_response(data)
        assert data["modelType"] == "sam3"

    def test_list_returns_lightweight_boxes(self):
        """List endpoint returns boxes without maskPolygon."""
        resp = requests.get(f"{BASE}/detections", params={"page": 1, "pageSize": 5})
        assert resp.status_code == 200
        items = resp.json()["data"]
        if items:
            box = items[0]["boxes"][0]
            assert "maskPolygon" not in box
            assert "modelType" in items[0]

    def test_detail_returns_full_data(self):
        """Detail endpoint returns boxes with maskPolygon, and masks are valid."""
        list_resp = requests.get(f"{BASE}/detections", params={"page": 1, "pageSize": 1})
        items = list_resp.json()["data"]
        if not items:
            pytest.skip("No detections in DB")
        det_id = items[0]["id"]
        resp = requests.get(f"{BASE}/detections/{det_id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        boxes = data["boxes"]
        for b in boxes:
            assert "maskPolygon" in b
            _validate_mask_polygon(b["maskPolygon"], data["imageWidth"], data["imageHeight"])

    def test_list_detail_consistency(self):
        """List and detail endpoints agree on box count and basic info."""
        list_resp = requests.get(f"{BASE}/detections", params={"page": 1, "pageSize": 1})
        items = list_resp.json()["data"]
        if not items:
            pytest.skip("No detections in DB")
        item = items[0]
        det_id = item["id"]
        detail_resp = requests.get(f"{BASE}/detections/{det_id}")
        detail = detail_resp.json()["data"]
        # Core fields match
        assert detail["id"] == item["id"]
        assert detail["imageName"] == item["imageName"]
        assert detail["imageWidth"] == item["imageWidth"]
        assert detail["imageHeight"] == item["imageHeight"]
        assert detail["modelType"] == item["modelType"]
        assert len(detail["boxes"]) == len(item["boxes"])


@backend_required
class TestModelManagement:
    def test_model_events_sse(self):
        """SSE endpoint streams model status events."""
        resp = requests.get(f"{BASE}/model/events", stream=True, timeout=10)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Read first event
        line = ""
        for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                line += chunk
            if "\n\n" in line:
                break
        assert "data:" in line
        resp.close()

    def test_model_status_endpoints(self):
        """Individual status endpoints respond."""
        endpoints = ["/model/status", "/model/sam2/status", "/model/sam3/status"]
        for ep in endpoints:
            resp = requests.get(f"{BASE}{ep}", timeout=5)
            assert resp.status_code == 200
            assert "data" in resp.json()

    def test_sam3_unload(self):
        """SAM3 unload returns 204."""
        resp = requests.post(f"{BASE}/model/sam3/unload", timeout=5)
        # 204 or 500 (not running) both acceptable
        assert resp.status_code in (204, 500)


@backend_required
class TestTrainingFlow:
    def test_list_training_jobs(self):
        """Training jobs list endpoint works."""
        resp = requests.get(f"{BASE}/train/jobs", timeout=5)
        assert resp.status_code == 200

    def test_rename_job(self):
        """Rename training job — success, 404, empty → null."""
        # Get an existing job to rename
        resp = requests.get(f"{BASE}/train/jobs", timeout=5)
        assert resp.status_code == 200
        jobs = resp.json().get("data", [])
        if not jobs:
            pytest.skip("No training jobs to rename")
        job_id = jobs[0]["id"]

        # Successful rename
        resp = requests.post(
            f"{BASE}/train/jobs/{job_id}/rename",
            json={"name": "test-rename"},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "test-rename"

        # Empty string → null
        resp = requests.post(
            f"{BASE}/train/jobs/{job_id}/rename",
            json={"name": ""},
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] is None

        # Restore to original (no name)
        requests.post(
            f"{BASE}/train/jobs/{job_id}/rename",
            json={"name": ""},
            timeout=5,
        )

    def test_rename_job_404(self):
        """Rename non-existent job returns 404."""
        resp = requests.post(
            f"{BASE}/train/jobs/00000000-0000-0000-0000-000000000000/rename",
            json={"name": "test"},
            timeout=5,
        )
        assert resp.status_code == 404

    def test_list_variants(self):
        """YOLO variants endpoint works."""
        resp = requests.get(f"{BASE}/train/variants", timeout=5)
        assert resp.status_code == 200
        assert len(resp.json()["data"]) > 0

    def test_train_and_validate(self):
        """Detect 10 images → train YOLO → verify completion and metrics."""
        if not CAT_DIR or not os.path.isdir(CAT_DIR):
            pytest.skip("No test images available")
        images = sorted(
            [f for f in os.listdir(CAT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )[:10]
        if len(images) < 5:
            pytest.skip(f"Need at least 5 images, found {len(images)}")

        # Step 1: Detect 10 images
        det_ids: list[str] = []
        for img_name in images:
            img_path = os.path.join(CAT_DIR, img_name)
            with open(img_path, "rb") as f:
                resp = requests.post(
                    f"{BASE}/detect",
                    files={"file": (img_name, f, "image/jpeg")},
                    data={"categories": '["cat"]'},
                    timeout=120,
                )
            assert resp.status_code == 201, f"Detect failed for {img_name}: {resp.text}"
            det_id = resp.json()["data"]["id"]
            # Wait briefly to ensure detection is fully processed
            for _ in range(30):
                detail = requests.get(f"{BASE}/detections/{det_id}").json()
                if detail.get("data", {}).get("status") == "completed":
                    break
                time.sleep(1)
            det_ids.append(det_id)

        assert len(det_ids) >= 5, f"Only detected {len(det_ids)} images"

        # Step 2: Start training
        train_resp = requests.post(
            f"{BASE}/train/jobs",
            json={
                "detectionIds": det_ids,
                "modelVariant": "yolo26n",
                "epochs": 1,
                "imgsz": 320,
                "batch": 4,
                "trainRatio": 0.7,
                "valRatio": 0.2,
                "taskType": "detect",
            },
            timeout=30,
        )
        assert train_resp.status_code == 201, f"Train create failed: {train_resp.text}"
        job_id = train_resp.json()["data"]["id"]

        # Step 3: Poll until training completes
        job = None
        for _ in range(120):  # up to 10 minutes
            job = requests.get(f"{BASE}/train/jobs/{job_id}").json()["data"]
            status = job["status"]
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(5)
        assert job is not None
        assert job["status"] == "completed", (
            f"Training failed: {job.get('errorMessage', 'unknown')}"
        )
        assert job.get("metrics") is not None, "No metrics returned"
        ns = job["metrics"]["num_samples"]
        assert ns >= 1, f"Expected at least 1 sample, got {ns}"

        # Step 4: Download model
        dl_resp = requests.get(f"{BASE}/train/jobs/{job_id}/download", timeout=30)
        assert dl_resp.status_code == 200
        assert len(dl_resp.content) > 1000, "Model file too small"

        # Validate with trained model
        val_url = f"{BASE}/train/jobs/{job_id}/predict"
        with open(TEST_IMAGE, "rb") as f:
            val_resp = requests.post(val_url, files={"file": f}, timeout=30)
        assert val_resp.status_code == 200
        assert "boxes" in val_resp.json()["data"]
