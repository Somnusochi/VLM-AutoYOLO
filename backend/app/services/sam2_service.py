"""SAM2 segmentation service — bbox → mask via Segment Anything Model 2."""

from __future__ import annotations

import contextlib
import enum
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..core.config import hf_repo_cache_dir, settings
from ..core.gpu_memory import get_memory_manager

logger = logging.getLogger(__name__)

# ── SAM2 State machine ──────────────────────────────


class Sam2State(enum.StrEnum):
    UNLOADED = "unloaded"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


_sam2_state: dict = {
    "state": Sam2State.UNLOADED,
    "stage": "",
    "progress": 0,
    "error": "",
}
_sam2_state_lock = threading.Lock()

_sam_worker: SegmentAnythingWorker | None = None
_last_activity: float = 0.0
_watchdog_thread: threading.Thread | None = None
_watchdog_stop: threading.Event | None = None
_lock: threading.Lock = threading.Lock()
_load_thread: threading.Thread | None = None
_load_complete = threading.Event()


class SegmentAnythingWorker:
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        progress_cb=None,
    ):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device or settings.resolved_device
        self.model_id = model_id or settings.sam2_model_id
        logger.info("Loading SAM2 (%s) to %s...", self.model_id, self.device)

        # Stage 1: resolve/download model
        self._set_progress(progress_cb, "downloading", "model", 30)
        sam2_model = _build_sam2_model(self.model_id)

        # Stage 2: move to GPU
        self._set_progress(progress_cb, "loading", "gpu", 80)
        if self.device != "cpu":
            sam2_model = sam2_model.to(self.device)
        sam2_model.eval()
        self.predictor = SAM2ImagePredictor(sam2_model)

        logger.info("SAM2 ready on %s", self.device)
        self._set_progress(progress_cb, "loaded", "", 100)

    @staticmethod
    def _set_progress(cb, state: str, stage: str, progress: int):
        if cb:
            with contextlib.suppress(Exception):
                cb(state, stage, progress)

    @torch.no_grad()
    def segment(
        self, image: Image.Image, boxes: list[dict], score_threshold: float = 0.0
    ) -> list[list[list[float]]]:
        """Generate polygon masks for each bounding box.

        Args:
            score_threshold: Skip masks with SAM2 confidence below this (0.0–1.0).
        """
        np_image = np.array(image.convert("RGB"))
        self.predictor.set_image(np_image)

        polygons: list[list[list[float]]] = []
        for box in boxes:
            bbox = np.array([box["x1"], box["y1"], box["x2"], box["y2"]])
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,
            )
            if masks is None or masks.shape[0] == 0:
                polygons.append([])
                continue
            if scores[0] < score_threshold:
                polygons.append([])
                continue
            mask = masks[0].astype(np.uint8)
            mask = _clean_mask(mask)
            contours = _find_contours(mask)
            if contours:
                largest = max(contours, key=lambda c: _contour_area(c))
                if len(largest) >= 3:
                    poly = largest.squeeze(1).tolist()
                    polygons.append([[float(p[0]), float(p[1])] for p in poly])
                    continue
            polygons.append([])

        get_memory_manager().full_cleanup()
        return polygons


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        return mask


def _contour_area(contour: np.ndarray) -> float:
    try:
        import cv2

        return cv2.contourArea(contour)
    except ImportError:
        return len(contour)


def _find_contours(mask: np.ndarray):
    try:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    except ImportError:
        return []


def _hf_repo_cache_dir(model_id: str, hf_home: Path | None = None) -> Path:
    if hf_home is not None:
        return hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    return hf_repo_cache_dir(model_id)


def _find_cached_checkpoint(
    model_id: str, checkpoint_name: str, hf_home: Path | None = None
) -> Path | None:
    repo_dir = _hf_repo_cache_dir(model_id, hf_home=hf_home)
    for path in repo_dir.glob(f"snapshots/*/{checkpoint_name}"):
        if path.is_file():
            return path
    for path in repo_dir.rglob(checkpoint_name):
        if path.is_file():
            return path
    return None


def _build_sam2_model(model_id: str):
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2, build_sam2_hf

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]

    if settings.sam2_checkpoint_path:
        ckpt_path = Path(settings.sam2_checkpoint_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"SAM2_CHECKPOINT_PATH does not exist: {ckpt_path}")
        logger.info("Loading SAM2 checkpoint from SAM2_CHECKPOINT_PATH=%s", ckpt_path)
        return build_sam2(config_file=config_name, ckpt_path=str(ckpt_path), device="cpu")

    cached_checkpoint = _find_cached_checkpoint(model_id, checkpoint_name)
    if cached_checkpoint is not None:
        logger.info("Loading SAM2 checkpoint from local Hugging Face cache: %s", cached_checkpoint)
        return build_sam2(config_file=config_name, ckpt_path=str(cached_checkpoint), device="cpu")

    return build_sam2_hf(model_id, device="cpu")


# ── Module-level singleton ────────────────────────────


def _sam2_progress_callback(state: str, stage: str, progress: int):
    with _sam2_state_lock:
        _sam2_state["state"] = Sam2State(state)
        _sam2_state["stage"] = stage
        _sam2_state["progress"] = progress
        if state == "loaded":
            _sam2_state["error"] = ""


def _load_sam2_sync():
    global _sam_worker
    try:
        _sam2_state["error"] = ""
        _load_complete.clear()
        _sam_worker = SegmentAnythingWorker(progress_cb=_sam2_progress_callback)
        with _sam2_state_lock:
            _sam2_state["state"] = Sam2State.LOADED
            _sam2_state["stage"] = ""
            _sam2_state["progress"] = 100
        _bump_activity()
        _start_watchdog()
    except Exception as exc:
        logger.exception("SAM2 loading failed")
        with _sam2_state_lock:
            _sam2_state["state"] = Sam2State.ERROR
            _sam2_state["stage"] = ""
            _sam2_state["progress"] = 0
            _sam2_state["error"] = str(exc)
    finally:
        _load_complete.set()


def _start_sam2_loading():
    global _load_thread
    with _sam2_state_lock:
        if _sam2_state["state"] in (Sam2State.DOWNLOADING, Sam2State.LOADING):
            return
        _sam2_state["state"] = Sam2State.DOWNLOADING
        _sam2_state["stage"] = "starting"
        _sam2_state["progress"] = 0
        _sam2_state["error"] = ""

    _load_complete.clear()
    _load_thread = threading.Thread(
        target=_load_sam2_sync,
        daemon=True,
        name="sam2-loader",
    )
    _load_thread.start()


def _watchdog_loop():
    while _watchdog_stop is not None and not _watchdog_stop.is_set():
        _watchdog_stop.wait(timeout=30)
        if _watchdog_stop is None or _watchdog_stop.is_set():
            break
        with _lock:
            idle = time.monotonic() - _last_activity
        if idle >= settings.model_idle_timeout_seconds:
            logger.info("SAM2 unloaded after %.0fs idle", idle)
            unload_sam()
            break


def _start_watchdog():
    global _watchdog_thread, _watchdog_stop
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return
    _watchdog_stop = threading.Event()
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop, daemon=True, name="sam-idle-watchdog"
    )
    _watchdog_thread.start()


def _stop_watchdog():
    global _watchdog_thread, _watchdog_stop
    if _watchdog_stop is not None:
        _watchdog_stop.set()
    _watchdog_thread = None
    _watchdog_stop = None


def _bump_activity():
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


def _get_sam_worker() -> SegmentAnythingWorker:
    global _sam_worker
    if _sam_worker is not None:
        return _sam_worker

    # If already loading in background, wait for it
    with _sam2_state_lock:
        if _sam2_state["state"] in (Sam2State.DOWNLOADING, Sam2State.LOADING):
            pass

    if _sam_worker is None:
        _start_sam2_loading()
        _load_complete.wait()

        if _sam_worker is None:
            with _sam2_state_lock:
                err = _sam2_state["error"]
            raise RuntimeError(err or "SAM2 loading failed")

    return _sam_worker


def segment_image(
    image: Image.Image, boxes: list[dict], score_threshold: float = 0.0
) -> list[list[list[float]]]:
    worker = _get_sam_worker()
    _bump_activity()
    get_memory_manager().empty_cache()
    return worker.segment(image, boxes, score_threshold=score_threshold)


def get_sam2_status() -> dict:
    with _sam2_state_lock:
        return dict(_sam2_state)


def is_sam_loaded() -> bool:
    with _sam2_state_lock:
        return _sam2_state["state"] == Sam2State.LOADED


def unload_sam() -> None:
    global _sam_worker, _load_thread
    _stop_watchdog()
    _load_thread = None
    if _sam_worker is not None:
        del _sam_worker.predictor
        _sam_worker = None
    get_memory_manager().full_cleanup()
    with _sam2_state_lock:
        _sam2_state["state"] = Sam2State.UNLOADED
        _sam2_state["stage"] = ""
        _sam2_state["progress"] = 0
        _sam2_state["error"] = ""
    logger.info("SAM2 unloaded")
