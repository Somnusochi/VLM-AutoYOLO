"""VLM model service — wraps LocateAnything-3B for visual grounding."""

from __future__ import annotations

import contextlib
import enum
import logging
import re
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from ..core.config import settings
from ..core.exceptions import InferenceError
from ..core.gpu_memory import get_memory_manager, validate_vlm_device

logger = logging.getLogger(__name__)

# decord is Linux-only; on macOS/Windows we provide a stub
_decord_loaded = False


def _ensure_decord_stub():
    global _decord_loaded
    if _decord_loaded:
        return
    if sys.platform in ("darwin", "win32"):
        try:
            import decord  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            import decord_stub

            # Transformers checks optional imports before loading remote model
            # code. Register the bundled image-only compatibility shim under
            # the package name it expects.
            sys.modules["decord"] = decord_stub
    _decord_loaded = True


# ── Model state machine ────────────────────────────


class ModelState(enum.StrEnum):
    UNLOADED = "unloaded"
    DOWNLOADING = "downloading"  # fetching model files from HuggingFace
    LOADING = "loading"  # loading into memory / moving to GPU
    LOADED = "loaded"
    ERROR = "error"


_model_state: dict = {
    "state": ModelState.UNLOADED,
    "stage": "",  # e.g. "tokenizer", "processor", "model", "gpu"
    "progress": 0,  # 0–100 percentage
    "error": "",  # error message when state == ERROR
}
_state_lock = threading.Lock()


class LocateAnythingWorker:
    def __init__(self, model_path: str, device: str = "mps", progress_cb=None):
        _ensure_decord_stub()

        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.dtype = torch.bfloat16

        gpu_mem = get_memory_manager()

        # ── Download tokenizer ──
        self._set_progress(progress_cb, "downloading", "tokenizer", 10)
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ── Download processor ──
        self._set_progress(progress_cb, "downloading", "processor", 30)
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # ── Download model weights ──
        self._set_progress(progress_cb, "downloading", "model", 50)
        attn_impl = gpu_mem.resolve_attn_impl()
        logger.info("Loading model to %s (attn=%s)...", device, attn_impl)
        load_kwargs = {}
        if device == "cuda":
            total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if total_mb < 10 * 1024:
                # The BF16 model is over 7 GiB. Keeping it entirely on an 8 GiB
                # display GPU leaves no room for activations and triggers very
                # slow WDDM shared-memory paging.
                load_kwargs = {
                    "device_map": "auto",
                    "max_memory": {0: "5GiB", "cpu": "12GiB"},
                }
                logger.info("Low-VRAM CUDA detected; enabling CPU/GPU weight offload")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )

        # ── Move to GPU ──
        self._set_progress(progress_cb, "loading", "gpu", 90)
        if load_kwargs:
            self.model.eval()
            logger.info("Model device map: %s", getattr(self.model, "hf_device_map", {}))
        else:
            self.model = self.model.to(device).eval()

        logger.info("LocateAnything model ready on %s", device)
        self._set_progress(progress_cb, "loaded", "", 100)

    @staticmethod
    def _set_progress(cb, state: str, stage: str, progress: int):
        if cb:
            with contextlib.suppress(Exception):
                cb(state, stage, progress)

    @torch.inference_mode()
    def detect(self, image: Image.Image, categories: list[str]) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self._predict(image, prompt)

    @torch.inference_mode()
    def _predict(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128,
    ) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        del images, videos  # free CPU memory early — no longer needed after processor
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="hybrid",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )

        raw_text = response[0] if isinstance(response, tuple) else response

        # Aggressively free intermediate tensors before returning
        del inputs, response
        get_memory_manager().full_cleanup()

        return {"answer": raw_text}


# ── Module-level singleton ────────────────────────────

_worker: LocateAnythingWorker | None = None
_last_activity: float = 0.0
_watchdog_thread: threading.Thread | None = None
_watchdog_stop: threading.Event | None = None
_lock: threading.Lock = threading.Lock()
_load_thread: threading.Thread | None = None
_load_complete = threading.Event()


def _progress_callback(state: str, stage: str, progress: int):
    """Update the shared model state dict for polling by the status endpoint."""
    with _state_lock:
        _model_state["state"] = ModelState(state)
        _model_state["stage"] = stage
        _model_state["progress"] = progress
        if state == "loaded":
            _model_state["error"] = ""


def _set_model_error(exc: Exception) -> None:
    with _state_lock:
        _model_state["state"] = ModelState.ERROR
        _model_state["stage"] = ""
        _model_state["progress"] = 0
        _model_state["error"] = str(exc)


def _load_worker_sync(model_path: str, device: str):
    """Load the model synchronously (runs in background thread)."""
    global _worker
    try:
        _model_state["error"] = ""
        _load_complete.clear()
        _worker = LocateAnythingWorker(model_path, device=device, progress_cb=_progress_callback)
        with _state_lock:
            _model_state["state"] = ModelState.LOADED
            _model_state["stage"] = ""
            _model_state["progress"] = 100
        _bump_activity()
        _start_watchdog()
    except Exception as exc:
        logger.exception("Model loading failed")
        _set_model_error(exc)
    finally:
        _load_complete.set()


def _start_loading(model_path: str, device: str):
    """Start background model loading if not already in progress."""
    global _load_thread
    with _state_lock:
        if _model_state["state"] in (ModelState.DOWNLOADING, ModelState.LOADING):
            return  # already loading
        _model_state["state"] = ModelState.DOWNLOADING
        _model_state["stage"] = "starting"
        _model_state["progress"] = 0
        _model_state["error"] = ""

    _load_complete.clear()
    _load_thread = threading.Thread(
        target=_load_worker_sync,
        args=(model_path, device),
        daemon=True,
        name="model-loader",
    )
    _load_thread.start()


def _watchdog_loop():
    """Background thread that unloads the model after idle timeout."""
    while _watchdog_stop is not None and not _watchdog_stop.is_set():
        _watchdog_stop.wait(timeout=30)
        if _watchdog_stop is None or _watchdog_stop.is_set():
            break
        with _lock:
            idle = time.monotonic() - _last_activity
        if idle >= settings.model_idle_timeout_seconds:
            logger.info("Model idle for %.0fs, auto-unloading...", idle)
            unload_model()
            break


def _start_watchdog():
    global _watchdog_thread, _watchdog_stop
    if _watchdog_thread is not None and _watchdog_thread.is_alive():
        return
    _watchdog_stop = threading.Event()
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop, daemon=True, name="model-idle-watchdog"
    )
    _watchdog_thread.start()
    logger.info("Idle watchdog started (timeout=%ds)", settings.model_idle_timeout_seconds)


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


def _get_worker() -> LocateAnythingWorker:
    global _worker
    if _worker is not None:
        return _worker

    model_path = settings.resolved_model_dir
    model_dir_path = Path(model_path)
    if not (model_dir_path.exists() and (model_dir_path / "config.json").exists()):
        model_path = settings.model_id

    try:
        device = settings.resolved_device
        validate_vlm_device(device)
    except Exception as exc:
        logger.exception("Model device preflight failed")
        _set_model_error(exc)
        raise

    # If already loading in background, wait for it
    with _state_lock:
        if _model_state["state"] in (ModelState.DOWNLOADING, ModelState.LOADING):
            pass  # wait below

    if _worker is None:
        # Start or wait for background load
        _start_loading(model_path, device)
        _load_complete.wait()  # block until load finishes

        if _worker is None:
            # Loading failed
            with _state_lock:
                err = _model_state["error"]
            raise RuntimeError(err or "Model loading failed")

    return _worker


_REF_PATTERN = re.compile(r"<ref>([^<]+)</ref>")
_BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_CONF_PATTERN = re.compile(r"<conf>([\d.]+)</conf>")
_TOKEN_PATTERN = re.compile(r"<ref>[^<]+</ref>|<box><\d+><\d+><\d+><\d+></box>|<conf>[\d.]+</conf>")


def parse_boxes(raw_text: str, img_w: int, img_h: int) -> list[dict]:
    """Parse model output into boxes with class names and confidence.

    Expected model format:
      <ref>cat</ref><box><100><200><300><400></box><conf>0.92</conf>

    Each <conf> is optional and applies to the immediately preceding <box>.
    <ref> tags set the label for all subsequent boxes until the next <ref>.
    """
    boxes: list[dict] = []
    current_class = ""
    last_conf = None

    for token in _TOKEN_PATTERN.findall(raw_text):
        if token.startswith("<ref>"):
            m = _REF_PATTERN.match(token)
            if m:
                current_class = m.group(1).strip()
        elif token.startswith("<box>"):
            m = _BOX_PATTERN.match(token)
            if m and current_class:
                x1, y1, x2, y2 = map(int, m.groups())
                boxes.append(
                    {
                        "class_name": current_class,
                        "x1": int(x1 / 1000 * img_w),
                        "y1": int(y1 / 1000 * img_h),
                        "x2": int(x2 / 1000 * img_w),
                        "y2": int(y2 / 1000 * img_h),
                        "confidence": last_conf,
                    }
                )
                last_conf = None
        elif token.startswith("<conf>"):
            m = _CONF_PATTERN.match(token)
            if m:
                try:
                    last_conf = float(m.group(1))
                except ValueError:
                    last_conf = None

    return boxes


_max_long_side: int | None = None


def _get_max_long_side() -> int:
    global _max_long_side
    if _max_long_side is None:
        _max_long_side = get_memory_manager().resolve_max_long_side()
        logger.info("Image long-side cap set to %dpx (auto-detected from GPU)", _max_long_side)
    return _max_long_side


def detect(image_path: str | Path, categories: list[str]) -> dict:
    try:
        worker = _get_worker()
    except Exception as exc:
        raise InferenceError(f"Model loading failed: {exc}") from exc
    _bump_activity()

    gpu_mem = get_memory_manager()
    gpu_mem.empty_cache()

    img = Image.open(image_path).convert("RGB")
    try:
        orig_w, orig_h = img.size
        w, h = orig_w, orig_h

        # Downscale large images to avoid GPU OOM (ViT attention is quadratic)
        longest = max(w, h)
        cap = _get_max_long_side()
        if longest > cap:
            scale = cap / longest
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info("Resizing image %dx%d -> %dx%d", w, h, new_w, new_h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        try:
            result = worker.detect(img, categories)
            raw_text = result["answer"]
            logger.debug("Raw model response: %r", raw_text)
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

        boxes = parse_boxes(raw_text, w, h)
        logger.info("Detection: %s -> %d boxes for %s", image_path, len(boxes), categories)

        return {
            "raw_text": raw_text,
            "boxes": boxes,
            "img_w": w,
            "img_h": h,
            "orig_w": orig_w,
            "orig_h": orig_h,
        }
    finally:
        img.close()
        gpu_mem.full_cleanup()


def get_model_status() -> dict:
    """Return current model state for the status endpoint."""
    with _state_lock:
        return dict(_model_state)


def is_model_loaded() -> bool:
    with _state_lock:
        return _model_state["state"] == ModelState.LOADED


def unload_model() -> None:
    global _worker, _load_thread
    _stop_watchdog()
    _load_thread = None
    if _worker is not None:
        del _worker.model
        del _worker.tokenizer
        del _worker.processor
        _worker = None
    get_memory_manager().full_cleanup()
    with _state_lock:
        _model_state["state"] = ModelState.UNLOADED
        _model_state["stage"] = ""
        _model_state["progress"] = 0
        _model_state["error"] = ""
    logger.info("Model unloaded")
